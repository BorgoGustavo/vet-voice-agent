"""Voice agent for a vet clinic: answers the phone, books, moves and cancels appointments.

    uv run --env-file .env agent.py dev      run a worker against LiveKit Cloud
    uv run --env-file .env agent.py start    the same, for production

Every rule about the schedule lives in the SQL functions (supabase/migrations). This file
only turns a phone call into calls to them, and keeps the LLM away from what it must not
choose: who the caller is comes from caller ID, never from a tool argument.
"""

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg
from livekit import api
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    Agent,
    AgentServer,
    AgentSession,
    APIConnectOptions,
    JobContext,
    RunContext,
    ToolError,
    cli,
    function_tool,
    get_job_context,
    inference,
    tts,
)
from livekit.agents.beta import EndCallTool
from livekit.agents.utils import shortuuid
from livekit.plugins import openai
from psycopg.types.json import Jsonb

logger = logging.getLogger("vet-agent")

CLINIC_NAME = "Clínica Veterinária Bicho Feliz"
CLINIC_TZ = ZoneInfo("America/Sao_Paulo")
WEEKDAYS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
RECORDINGS = Path(os.environ.get("RECORDINGS_DIR", "recordings"))  # both voices of each call; personal data, not in git
TTS_VOICE = "nova"  # picked by ear among coral, shimmer, sage and nova; coral sounded harsh
TTS_INSTRUCTIONS = (
    "Fale em português do Brasil com voz calma, suave e gentil, como uma recepcionista simpática. "
    "Ritmo tranquilo, sem ênfases fortes."
)

# USD, from developers.openai.com/api/docs/pricing on 2026-10-19. Per 1M tokens, or per
# minute of audio for the live transcription model. Telephony minutes are not included.
PRICES = {
    "gpt-4.1-mini": {"input": 0.40, "cached": 0.10, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "cached": 0.50, "output": 8.00},
    "gpt-4o-mini-tts": {"input": 0.60, "output": 12.00},
    "tts-1": {"character": 15.00},  # per 1M characters
    "gpt-live-transcribe": {"minute": 0.017},
}
HEDGE_AFTER = 1.5  # seconds without audio from the voice before the backup is asked too


@dataclass
class Call:
    db: psycopg.AsyncConnection
    phone: str | None  # caller ID, the only identity the agent trusts
    client_id: int | None = None
    call_id: str | None = None  # LiveKit room name. None in text evals: no call row to point at.
    caller_identity: str | None = None  # the SIP participant, for the transfer
    outcome: str = "no_action"


async def query(call: Call, sql: str, *args):
    cur = await call.db.execute(sql, args)
    return (await cur.fetchone())[0]


def local_time(text: str) -> datetime:
    """Clinic wall-clock time from the ISO string the LLM wrote.

    The SQL functions take local time without a zone, and Postgres drops the offset of
    "2026-10-26T13:00Z" instead of converting it, which would book 13:00 instead of 10:00.
    """
    try:
        t = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        raise ToolError(f"horário inválido: {text!r}. Use AAAA-MM-DDTHH:MM, no horário local da clínica.") from None
    return t.astimezone(CLINIC_TZ).replace(tzinfo=None) if t.tzinfo else t


def known_client(call: Call) -> int:
    if call.client_id is None:
        raise ToolError("Este número não tem cadastro. Peça o nome do tutor, do pet e a espécie e use register_pet.")
    return call.client_id


def answer(result) -> str:
    return json.dumps(result, ensure_ascii=False)  # the SDK would send repr() of a dict


class HedgedTTS(tts.TTS):
    """Text to speech with a hedged request ("The Tail at Scale", Dean and Barroso).

    The hourly probe caught gpt-4o-mini-tts taking 10 s, 54 s or past a 30 s timeout to start
    4 replies out of 10, and the plugin waits up to 30 s per attempt: the caller hears silence
    and hangs up. Here, if the primary has not started speaking after `hedge_after` seconds
    (or fails), the backup is asked as well and whichever speaks first is played. The extra
    request is paid only on the slow ones.
    """

    def __init__(self, primary: tts.TTS, backup: tts.TTS, hedge_after: float) -> None:
        assert (primary.sample_rate, primary.num_channels) == (backup.sample_rate, backup.num_channels)
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=primary.sample_rate,
            num_channels=primary.num_channels,
        )
        self.primary, self.backup, self.hedge_after = primary, backup, hedge_after
        for t in (primary, backup):  # usage and cost come from the request that ran
            t.on("metrics_collected", lambda m: self.emit("metrics_collected", m))

    @property
    def model(self) -> str:
        return f"hedged:{self.primary.model}"  # unpriced on purpose: the inner metrics carry the cost

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        return _HedgedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        await asyncio.gather(self.primary.aclose(), self.backup.aclose())


class _HedgedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        hedged: HedgedTTS = self._tts
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=hedged.sample_rate,
            num_channels=hedged.num_channels,
            mime_type="audio/pcm",
        )
        once = APIConnectOptions(max_retry=0, timeout=self._conn_options.timeout)  # this stream retries as a whole
        streams = [hedged.primary.synthesize(self._input_text, conn_options=once)]
        firsts = [asyncio.ensure_future(anext(streams[0]))]
        try:
            await asyncio.wait(firsts, timeout=hedged.hedge_after)
            if not firsts[0].done() or firsts[0].exception():
                logger.info("tts hedge: primary silent after %.1f s, asking the backup", hedged.hedge_after)
                streams.append(hedged.backup.synthesize(self._input_text, conn_options=once))
                firsts.append(asyncio.ensure_future(anext(streams[1])))
            async for first in asyncio.as_completed(firsts):
                if first.exception() is None:
                    break
            else:
                raise firsts[-1].exception()
            winner = streams[firsts.index(first)]
            if len(streams) > 1:
                logger.info("tts hedge: %s spoke first", "backup" if winner is streams[1] else "primary")
            output_emitter.push(first.result().frame.data.tobytes())
            async for audio in winner:
                output_emitter.push(audio.frame.data.tobytes())
            output_emitter.flush()
        finally:
            for f in firsts:
                f.cancel()
            await asyncio.gather(*(s.aclose() for s in streams), return_exceptions=True)


def make_tts() -> HedgedTTS:
    # pcm: raw audio, nothing to decode, the format OpenAI recommends for the fastest start.
    # mp3 left gaps ("slow audio generation") that the caller heard as stutter. tts-1 is the
    # backup: slower to start (~2 s) but it never stalled in the probe. Same voice, no tone
    # instructions, so a rescued sentence can sound slightly different.
    return HedgedTTS(
        primary=openai.TTS(voice=TTS_VOICE, instructions=TTS_INSTRUCTIONS, response_format="pcm"),
        backup=openai.TTS(model="tts-1", voice=TTS_VOICE, response_format="pcm"),
        hedge_after=HEDGE_AFTER,
    )


def make_llm(model: str = LLM_MODEL) -> openai.LLM:
    # One tool call at a time: in parallel, the model booked a pet it was registering in the
    # same step, before register_pet had returned the new pet's id.
    return openai.LLM(model=model, parallel_tool_calls=False)


def instructions(context: dict) -> str:
    # Fixed text first, per-call text last. OpenAI only caches an identical prefix of at least
    # 1024 tokens, and the rules are about 560, so calls share no cache yet (measured: zero
    # cached tokens across callers in either order). Within a call the prompt never changes
    # and ~78% of input tokens are cached. This order starts paying once the rules grow.
    now = datetime.now(CLINIC_TZ)
    return f"""Você é a recepcionista virtual da {CLINIC_NAME} e está atendendo uma ligação telefônica.
A clínica atende de segunda a sexta das 8h às 18h e sábado das 8h às 12h. Consultas de 30 minutos, até 60 dias à frente.

Regras:
- Fale como numa ligação: frases curtas, sem listas nem markdown, sem ler ids.
- Não pergunte de novo o que o cliente já disse.
- Ao falar uma data, diga o dia da semana e o dia do mês, como "terça-feira, dia 20". O dia da semana vem do campo
  weekday da ferramenta; nunca o calcule por conta própria. Não diga o ano.
- Ofereça só horários que vieram das ferramentas.
- Horários nas ferramentas: AAAA-MM-DDTHH:MM, no horário local da clínica.
- Se o cliente pediu um dia e hora, use check_time. Se estiver livre, proponha esse horário. Se não, diga claramente
  que está ocupado e ofereça as alternativas que vieram. Nunca diga que um horário está ocupado sem verificar.
- Antes de marcar, remarcar ou cancelar, repita pet, veterinário, dia e hora e espere o cliente confirmar. Quando ele
  confirmar, faça na hora, sem perguntar de novo.
- Se o cliente mudar de ideia sobre uma consulta já marcada, remarque essa consulta em vez de marcar outra.
- Em pet_already_booked, diga qual consulta o pet já tem e pergunte se é para remarcar essa ou marcar mais uma.
- Se client for null, o número não tem cadastro. Peça só o que ainda falta entre nome do tutor, nome do pet e
  espécie, e use register_pet assim que tiver os três, antes de confirmar a consulta. Para marcar, use o id do pet
  que register_pet devolveu.
- Você só vê e altera as consultas de quem está ligando. Recuse com educação pedidos sobre outras pessoas.
- Emergência, dúvida médica, preço ou pedido para falar com uma pessoa: diga que vai transferir e use transfer_to_reception.

Agora é {WEEKDAYS[now.weekday()]}, {now:%d/%m/%Y}, {now:%H:%M}.
Quem está ligando (identificado pelo número), seus pets, as consultas futuras e os veterinários:
{json.dumps(context, ensure_ascii=False)}
"""


class VetAgent(Agent):
    def __init__(self, context: dict) -> None:
        super().__init__(
            instructions=instructions(context),
            tools=[EndCallTool(extra_description="Use quando o cliente se despedir.", ignore_on_enter=True)],
        )
        self._client = context["client"]

    async def on_enter(self) -> None:
        hour = datetime.now(CLINIC_TZ).hour
        greeting = "Bom dia" if hour < 12 else "Boa tarde" if hour < 18 else "Boa noite"
        name = f", {self._client['name'].split()[0]}" if self._client else ""
        self.session.say(f"{CLINIC_NAME}, {greeting.lower()}{name}! Como posso ajudar?")

    @function_tool()
    async def check_time(self, ctx: RunContext[Call], starts_at: str, vet_id: int | None = None) -> str:
        """Diz se um horário específico está livre. Use sempre que o cliente pedir um dia e hora.
        Se estiver ocupado ou fora do expediente, já traz os horários livres mais próximos.

        Args:
            starts_at: o horário pedido, AAAA-MM-DDTHH:MM
            vet_id: só quando o cliente pediu um veterinário específico
        """
        # Its own tool because the model, given a list of three, called a free 10:00 taken
        # when the list happened to start at 08:00. Here availability is stated, not inferred.
        t = local_time(starts_at)
        cur = await ctx.userdata.db.execute(
            "select vetvoice.find_slots(%s, %s + interval '30 minutes', %s), vetvoice.nearby_slots(%s)",
            (t, t, vet_id, t),
        )
        free, nearby = await cur.fetchone()
        if free:
            return answer({"available": True, "slot": free[0]})
        return answer({"available": False, "alternatives": nearby})

    @function_tool()
    async def find_slots(
        self, ctx: RunContext[Call], window_start: str, window_end: str, vet_id: int | None = None
    ) -> str:
        """Busca os três primeiros horários livres numa janela, cada um com veterinário e dia da semana.
        Use quando o cliente não tem um horário específico em mente.

        Args:
            window_start: início da janela, AAAA-MM-DDTHH:MM
            window_end: fim da janela (exclusivo), AAAA-MM-DDTHH:MM
            vet_id: só quando o cliente pediu um veterinário específico
        """
        return answer(
            await query(
                ctx.userdata,
                "select vetvoice.find_slots(%s, %s, %s)",
                local_time(window_start),
                local_time(window_end),
                vet_id,
            )
        )

    @function_tool()
    async def book_appointment(
        self, ctx: RunContext[Call], pet_id: int, vet_id: int, starts_at: str, additional: bool = False
    ) -> str:
        """Marca uma consulta. Só depois que o cliente confirmar pet, veterinário, dia e hora.

        Args:
            pet_id: id do pet de quem está ligando
            vet_id: id do veterinário
            starts_at: início da consulta, AAAA-MM-DDTHH:MM
            additional: true só se o pet já tem consulta e o cliente confirmou que quer mais uma
        """
        call = ctx.userdata
        result = await query(
            call,
            "select vetvoice.book_appointment(%s, %s, %s, %s, %s, %s)",
            known_client(call),
            pet_id,
            vet_id,
            local_time(starts_at),
            call.call_id,
            additional,
        )
        if result["status"] == "booked":
            call.outcome = "booked"
        return answer(result)

    @function_tool()
    async def reschedule_appointment(
        self, ctx: RunContext[Call], appointment_id: int, vet_id: int, starts_at: str
    ) -> str:
        """Move uma consulta já marcada para outro horário. Se o novo estiver ocupado, a antiga continua valendo.

        Args:
            appointment_id: id da consulta a mover
            vet_id: id do veterinário do novo horário
            starts_at: novo início, AAAA-MM-DDTHH:MM
        """
        call = ctx.userdata
        result = await query(
            call,
            "select vetvoice.reschedule_appointment(%s, %s, %s, %s)",
            known_client(call),
            appointment_id,
            vet_id,
            local_time(starts_at),
        )
        if result["status"] == "rescheduled":
            call.outcome = "rescheduled"
        return answer(result)

    @function_tool()
    async def cancel_appointment(self, ctx: RunContext[Call], appointment_id: int) -> str:
        """Cancela uma consulta futura. Só depois que o cliente confirmar.

        Args:
            appointment_id: id da consulta a cancelar
        """
        call = ctx.userdata
        result = await query(call, "select vetvoice.cancel_appointment(%s, %s)", known_client(call), appointment_id)
        if result["status"] == "cancelled":
            call.outcome = "cancelled"
        return answer(result)

    @function_tool()
    async def register_pet(self, ctx: RunContext[Call], owner_name: str | None, pet_name: str, species: str) -> str:
        """Cadastra um pet no número de quem está ligando, e o tutor junto se o número for novo.

        Args:
            owner_name: nome do tutor; só é necessário se o número ainda não tem cadastro
            pet_name: nome do pet
            species: espécie, como cachorro ou gato
        """
        call = ctx.userdata
        result = await query(
            call, "select vetvoice.register_pet(%s, %s, %s, %s)", call.phone, owner_name, pet_name, species
        )
        if result["status"] == "registered":
            call.client_id = result["client"]["id"]
            call.outcome = "registered"
        return answer(result)

    @function_tool()
    async def transfer_to_reception(self, ctx: RunContext[Call]) -> str:
        """Transfere a ligação para a recepção. Avise o cliente antes de chamar."""
        call = ctx.userdata
        await ctx.wait_for_playout()  # let "vou transferir" finish before the line changes
        try:
            # The format Twilio expects ("+55..." or "tel:+55...") is confirmed on a real call.
            # A failed transfer can come back as a status, not an exception.
            result = await get_job_context().transfer_sip_participant(
                call.caller_identity, os.environ["RECEPTION_NUMBER"]
            )
            ok = result.status != api.SIPTransferStatus.STS_TRANSFER_FAILED
        except Exception:
            logger.exception("transfer failed")
            ok = False
        if not ok:
            return answer({"status": "failed"})
        call.outcome = "transferred"
        return answer({"status": "transferred"})


def cost_usd(usage: list[dict], session_minutes: float = 0.0) -> float:
    """Model cost of a call. Per-minute models without an audio duration in their usage
    (live transcription streams the whole call) are billed for the session's length."""
    total = 0.0
    for u in usage:
        price = PRICES.get(u.get("model"))
        if price is None:
            continue  # local models (VAD, turn detection) are free; unknown ones show up in usage
        cached = u.get("input_cached_tokens", 0)
        total += (
            (u.get("input_tokens", 0) - cached) * price.get("input", 0)
            + cached * price.get("cached", 0)
            + u.get("output_tokens", 0) * price.get("output", 0)
        ) / 1e6
        minutes = u["audio_duration"] / 60 if u.get("audio_duration") else session_minutes
        total += minutes * price.get("minute", 0)
        total += u.get("characters_count", 0) * price.get("character", 0) / 1e6
    return round(total, 5)


server = AgentServer()


async def save_call(call: Call, report: dict, session_minutes: float) -> None:
    """Transcript (with per-turn latency), usage and cost of a finished call."""
    usage = report["usage"] or []
    await call.db.execute(
        # A Python float arrives as double precision, which matches no end_call signature.
        "select vetvoice.end_call(%s, %s, %s, %s, %s::numeric)",
        (
            call.call_id,
            call.outcome,
            Jsonb(report["chat_history"]),
            Jsonb(usage),
            cost_usd(usage, session_minutes),
        ),
    )


async def log_call(ctx: JobContext) -> None:
    """Runs after the session closes."""
    call: Call = ctx.primary_session.userdata
    report = ctx.make_session_report()
    minutes = (report.timestamp - report.started_at) / 60 if report.started_at else 0.0
    try:
        await save_call(call, report.to_dict(), minutes)
    finally:
        await call.db.close()
    # The session records both sides to a temp dir that is deleted with the job. Keep a copy,
    # with the moment it started, so demo.py can line the transcript up with the audio.
    if report.audio_recording_path and report.audio_recording_path.exists():
        RECORDINGS.mkdir(exist_ok=True)
        shutil.copy(report.audio_recording_path, RECORDINGS / f"{call.call_id}.ogg")
        (RECORDINGS / f"{call.call_id}.json").write_text(json.dumps({"started_at": report.audio_recording_started_at}))


@server.rtc_session(agent_name="vet-agent", on_session_end=log_call)
async def entrypoint(ctx: JobContext) -> None:
    # autocommit: each SQL function is its own transaction. Without it the booking and
    # the lock on the pet stay open until the call ends, and a hang-up rolls them back.
    db, caller = await asyncio.gather(
        # One of the pooler's three addresses was unreachable from the test network, psycopg
        # always tried it first, and with no timeout the caller heard 76 s of silence (the OS
        # TCP timeout). Shuffled order and a 2 s cap per address: median 0.9 s, worst 3 s.
        psycopg.AsyncConnection.connect(
            os.environ["DATABASE_URL"], autocommit=True, connect_timeout=2, load_balance_hosts="random"
        ),
        ctx.wait_for_participant(),
    )
    logger.info("caller attributes: %s", caller.attributes)  # which keys Twilio sends
    call = Call(
        db=db, phone=caller.attributes.get("sip.phoneNumber"), call_id=ctx.room.name, caller_identity=caller.identity
    )
    context = await query(
        call,
        "select vetvoice.start_call(%s, %s, %s)",
        call.call_id,
        call.phone,
        caller.attributes.get("sip.twilio.callSid"),
    )
    call.client_id = (context["client"] or {}).get("id")

    session = AgentSession[Call](
        stt=openai.STT(model="gpt-live-transcribe", language="pt"),  # streaming: no wait for a batch
        llm=make_llm(),
        tts=make_tts(),
        turn_handling={
            # Pinned: unpinned, `dev` uses the cloud model and `start` the local one, and the
            # latency numbers would not compare.
            "turn_detection": inference.TurnDetector(version="v1-mini"),
            # The first call split "queria marcar uma consulta para o meu... animal" in two:
            # the turn closed 0.3 s into a pause. The final transcript takes ~0.9 s anyway, so
            # waiting 0.6 s costs little latency and keeps a sentence in one turn.
            "endpointing": {"min_delay": 0.6},
            # Only words interrupt. When LiveKit's adaptive detector timed out (408) the session
            # fell back to raw VAD, a sound with no words cancelled the reply being generated
            # after a tool call, and with no transcript to answer the agent went silent.
            "interruption": {"min_words": 1},
        },
        userdata=call,
    )
    await session.start(agent=VetAgent(context), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
