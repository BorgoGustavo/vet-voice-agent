"""What the LLM can do during a call: one tool per SQL function.

Who is calling comes from caller ID, kept in Call. No tool takes a client id, so the LLM
cannot be talked into acting for someone else.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime

import psycopg
from livekit import api
from livekit.agents import Agent, RunContext, ToolError, function_tool, get_job_context
from livekit.agents.beta import EndCallTool

from prompt import CLINIC_NAME, CLINIC_TZ, instructions

logger = logging.getLogger("vet-agent")


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
