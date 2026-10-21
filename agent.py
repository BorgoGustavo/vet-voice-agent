"""Voice agent for a vet clinic: answers the phone, books, moves and cancels appointments.

    uv run --env-file .env agent.py dev      run a worker against LiveKit Cloud
    uv run --env-file .env agent.py start    the same, for production

Every rule about the schedule lives in the SQL functions (supabase/migrations). The agent
only turns a phone call into calls to them, and keeps the LLM away from what it must not
choose: who the caller is comes from caller ID, never from a tool argument.

This file wires a call together. The parts:
    tools.py    what the LLM can do, one SQL function per tool
    prompt.py   the system prompt
    voice.py    text to speech, hedged against a stalled request
    billing.py  model prices and the cost of a call
"""

import asyncio
import base64
import json
import logging
import os
import shutil
from pathlib import Path

import psycopg
from livekit.agents import AgentServer, AgentSession, JobContext, cli, inference
from livekit.agents.telemetry import set_tracer_provider
from livekit.plugins import openai
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from psycopg.types.json import Jsonb

from billing import cost_usd
from tools import Call, VetAgent, query
from voice import make_tts

logger = logging.getLogger("vet-agent")

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
RECORDINGS = Path(os.environ.get("RECORDINGS_DIR", "recordings"))  # both voices of each call; personal data, not in git


def make_llm(model: str = LLM_MODEL) -> openai.LLM:
    # One tool call at a time: in parallel, the model booked a pet it was registering in the
    # same step, before register_pet had returned the new pet's id.
    return openai.LLM(model=model, parallel_tool_calls=False)


def trace_to_langfuse(session_id: str) -> TracerProvider | None:
    """Sends the call's OpenTelemetry spans (turns, LLM, tools, voice) to Langfuse, when its
    keys are set. The spans carry transcripts and the caller's number (allow_pii)."""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    keys = f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}"
    exporter = OTLPSpanExporter(
        endpoint=f"{os.environ['LANGFUSE_BASE_URL'].rstrip('/')}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {base64.b64encode(keys.encode()).decode()}"},
    )
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    set_tracer_provider(provider, metadata={"langfuse.session.id": session_id})
    return provider


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
    if tracing := trace_to_langfuse(ctx.room.name):  # each call runs in its own process

        async def flush() -> None:
            tracing.force_flush()

        ctx.add_shutdown_callback(flush)
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
