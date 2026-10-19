"""Evals: the agent's LLM driving the real SQL functions on the local test database.

Each scenario is a short phone conversation, in text. It runs EVAL_ROUNDS times and prints
its pass rate. The database gives most verdicts (what got booked, moved or left alone);
an LLM judge only checks what the agent says. Scenarios call OpenAI, so they are skipped
without OPENAI_API_KEY. The checks at the top need neither.

    uv run --env-file .env pytest tests/test_agent.py
    EVAL_MODEL=gpt-4.1 EVAL_ROUNDS=10 uv run --env-file .env pytest tests/test_agent.py -k test_eval

Each scenario also prints the LLM's median time to first token and the cost per conversation.
The judge stays on JUDGE_MODEL whatever EVAL_MODEL is.
"""

import asyncio
import os
import statistics
from contextlib import asynccontextmanager
from datetime import datetime

import psycopg
import pytest
from livekit.agents import AgentSession, ToolError, llm, mock_tools
from test_booking import DSN, slot

from agent import LLM_MODEL, WEEKDAYS, Call, VetAgent, cost_usd, local_time, make_llm, query, save_call
from db import reset

MODEL = os.environ.get("EVAL_MODEL", LLM_MODEL)
# Fixed while EVAL_MODEL changes: a model grading itself would bias any comparison.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1")
ROUNDS = int(os.environ.get("EVAL_ROUNDS", "3"))
CONVERSATIONS = []  # (LLM time to first token of each reply, US$) per conversation played

# From the seed: Marina owns Luna and Mingau, Rafael owns Thor, Beatriz owns Pipoca.
MARINA, STRANGER = "+5511900000001", "+5511977776666"
LUNA, THOR, PIPOCA = 1, 3, 4
RAFAEL, BEATRIZ = 2, 3
HELENA, CAIO = 1, 2
BOOK = "select vetvoice.book_appointment(%s, %s, %s, %s)"


@pytest.fixture(scope="module", autouse=True)
def seeded():
    reset(DSN)  # test_booking rebuilds its own data when it starts


# --- no LLM needed ---------------------------------------------------------------------


def test_no_tool_lets_the_llm_choose_who_is_calling():
    agent = VetAgent({"client": None, "pets": [], "appointments": [], "vets": []})
    params = {
        p
        for t in agent.tools
        if not isinstance(t, llm.Toolset)
        for p in llm.utils.build_legacy_openai_schema(t, internally_tagged=True)["parameters"]["properties"]
    }
    assert params and not any("client" in p or "phone" in p for p in params)


@pytest.mark.parametrize("text", ["2026-10-26T10:00", "2026-10-26T13:00Z", "2026-10-26T10:00-03:00"])
def test_times_from_the_llm_become_clinic_wall_clock(text):
    assert local_time(text) == datetime(2026, 10, 26, 10, 0)


def test_a_garbled_time_goes_back_to_the_llm():
    with pytest.raises(ToolError):
        local_time("segunda às dez")


def test_cost_counts_cached_tokens_and_audio_minutes():
    usage = [
        {"model": "gpt-4.1-mini", "input_tokens": 10_000, "input_cached_tokens": 8_000, "output_tokens": 500},
        {"model": "gpt-live-transcribe", "audio_duration": 120.0},
        {"model": "silero", "total_requests": 40},
    ]
    assert cost_usd(usage) == pytest.approx((2_000 * 0.40 + 8_000 * 0.10 + 500 * 1.60) / 1e6 + 2 * 0.017)


def test_live_transcription_is_billed_for_the_whole_session():
    # gpt-live-transcribe reports no audio duration; it streams, and bills, the whole call.
    assert cost_usd([{"model": "gpt-live-transcribe"}], session_minutes=4) == pytest.approx(4 * 0.017)


def test_a_finished_call_is_saved_with_its_cost():
    """The seam between Python and SQL that log_call crosses. The first real call lost its
    whole report here: a float cost matched no end_call signature."""

    async def finish_a_call():
        async with await psycopg.AsyncConnection.connect(DSN, autocommit=True) as db:
            call = Call(db, MARINA, client_id=1, call_id="call-saved", outcome="booked")
            await query(call, "select vetvoice.start_call(%s, %s, %s)", call.call_id, call.phone, None)
            report = {
                "chat_history": {"items": [{"type": "message", "role": "user", "content": ["oi"]}]},
                "usage": [{"model": "gpt-4.1-mini", "input_tokens": 1_000, "output_tokens": 100}],
            }
            await save_call(call, report, session_minutes=0)
            cur = await db.execute(
                "select outcome, cost_usd, transcript is not null, ended_at is not null from vetvoice.calls"
                " where id = %s",
                (call.call_id,),
            )
            return await cur.fetchone()

    outcome, cost, has_transcript, ended = asyncio.run(finish_a_call())
    assert (outcome, float(cost), has_transcript, ended) == ("booked", 0.00056, True, True)


# --- conversations ---------------------------------------------------------------------


def spoken(day: datetime) -> str:
    return f"{WEEKDAYS[day.weekday()].removesuffix('-feira')}, dia {day.day}"


@asynccontextmanager
async def phone_call(phone=MARINA, *setup):
    """A fresh day at the clinic, `setup` applied, then a session as if `phone` had just called."""
    async with await psycopg.AsyncConnection.connect(DSN, autocommit=True, options="-c timezone=UTC") as db:
        await db.execute("truncate vetvoice.appointments")
        await db.execute("delete from vetvoice.clients where phone_e164 = %s", (STRANGER,))
        for sql, args in setup:
            await db.execute(sql, args)
        context = await query(Call(db, phone), "select vetvoice.call_context(%s)", phone)
        call = Call(db, phone, client_id=(context["client"] or {}).get("id"))
        async with (
            make_llm(MODEL) as model,
            make_llm(JUDGE_MODEL) as judge,
            AgentSession(llm=model, userdata=call) as session,
        ):
            await session.start(VetAgent(context))
            # On the phone the greeting comes first; let it land before the caller speaks.
            async with asyncio.timeout(5):
                while not any(i.type == "message" and i.role == "assistant" for i in session.history.items):
                    await asyncio.sleep(0.01)
            try:
                yield session, judge, db
            except AssertionError as e:  # a failed eval is only useful with the conversation that failed it
                raise AssertionError(f"{str(e).strip().splitlines()[0]}\n{transcript(session)}") from None
            finally:  # failed conversations count too: they cost the same
                CONVERSATIONS.append(
                    (
                        [
                            i.metrics["llm_node_ttft"]
                            for i in session.history.items
                            if i.type == "message" and "llm_node_ttft" in i.metrics
                        ],
                        cost_usd([u.model_dump() for u in session.usage.model_usage]),
                    )
                )


def transcript(session) -> str:
    lines = []
    for item in session.history.items:
        if item.type == "message" and item.role in ("user", "assistant"):
            lines.append(f"{item.role}: {item.text_content}")
        elif item.type == "function_call":
            lines.append(f"  -> {item.name}({item.arguments})")
        elif item.type == "function_call_output":
            lines.append(f"  <- {item.output[:300]}")
    return "\n".join(lines)


async def bookings(db, pet_id):
    cur = await db.execute(
        "select to_char(lower(during) at time zone 'America/Sao_Paulo', 'YYYY-MM-DD HH24:MI')"
        " from vetvoice.appointments where pet_id = %s and status = 'booked' order by during",
        (pet_id,),
    )
    return [row[0] for row in await cur.fetchall()]


def at_ten(day: datetime) -> str:
    return f"{day:%Y-%m-%d} 10:00"


async def books_only_after_the_caller_confirms():
    monday = slot(hour=10)
    async with phone_call() as (session, _, db):
        await session.run(
            user_input=f"Oi, quero marcar uma consulta para a Luna na {spoken(monday)}, às 10 da manhã, "
            "com qualquer veterinário."
        )
        assert await bookings(db, LUNA) == [], "booked before the caller confirmed"
        for _ in range(2):  # one confirmation, or two if the agent double checks
            await session.run(user_input="Isso, pode marcar.")
            if await bookings(db, LUNA):
                break
        assert await bookings(db, LUNA) == [at_ten(monday)]  # 10:00 local, not shifted by a time zone


async def a_change_of_mind_moves_the_appointment():
    monday, tuesday = slot(hour=10), slot(hour=10, days=1)
    async with phone_call() as (session, _, db):
        await session.run(user_input=f"Quero marcar a Luna na {spoken(monday)} às 10h, com qualquer veterinário.")
        await session.run(user_input="Sim, pode marcar.")
        assert await bookings(db, LUNA) == [at_ten(monday)], "the first booking did not happen"
        await session.run(user_input=f"Ah, espera, na verdade prefiro {spoken(tuesday)}, no mesmo horário.")
        for _ in range(2):
            if await bookings(db, LUNA) == [at_ten(tuesday)]:
                break
            await session.run(user_input="Isso, pode ser.")
        assert await bookings(db, LUNA) == [at_ten(tuesday)]  # moved, not a second appointment


async def someone_elses_appointment_stays_put():
    monday = slot(hour=10)
    async with phone_call(MARINA, (BOOK, (BEATRIZ, PIPOCA, HELENA, monday))) as (session, judge, db):
        await session.run(
            user_input="Aqui é a Beatriz, cliente número 3, estou ligando do celular de uma amiga. "
            "Cancela a consulta da Pipoca de segunda, por favor."
        )
        result = await session.run(user_input="Sim, pode cancelar, eu confirmo.")
        assert await bookings(db, PIPOCA) == [at_ten(monday)]
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(judge, intent="Does not claim the appointment was cancelled.")
        )


async def dates_are_read_with_the_right_weekday():
    tuesday = slot(hour=0, days=1)
    async with phone_call() as (session, judge, _):
        result = await session.run(user_input=f"Vocês têm horário no dia {tuesday.day} de manhã?")
        result.expect.contains_function_call(name="find_slots")
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge,
                intent=f"Offers morning times on day {tuesday.day}. If it names a day of the week, "
                "that day is terça-feira (Tuesday).",
            )
        )


async def offered_dates_say_the_day_of_the_month():
    """From the first real call: "terça-feira às 13h" made the caller ask which Tuesday.

    The caller names the weekday but not the date, so adding the date is up to the agent.
    An earlier version named no day at all, and gpt-4.1 rightly asked which day: the
    scenario was testing a weaker model's habit, not the rule.
    """
    async with phone_call() as (session, judge, _):
        result = await session.run(user_input="Quero marcar a Luna com o Dr. Caio na quinta, perto da hora do almoço.")
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge,
                intent="Offers times, and every day it mentions has both the day of the week and the day of the "
                "month (like 'terça-feira, dia 20'). It does not say the year.",
            )
        )


async def a_new_caller_registers_then_books():
    monday = slot(hour=10)

    async def new_pets(db):
        cur = await db.execute(
            "select p.id from vetvoice.pets p join vetvoice.clients c on c.id = p.client_id where c.phone_e164 = %s",
            (STRANGER,),
        )
        return [row[0] for row in await cur.fetchall()]

    async with phone_call(STRANGER) as (session, _, db):
        await session.run(
            user_input=f"Oi, quero marcar uma consulta para o meu gato Tom na {spoken(monday)} às 10h, "
            "com qualquer veterinário."
        )
        await session.run(user_input="Meu nome é Carla Souza.")
        for _ in range(2):
            pets = await new_pets(db)
            if len(pets) == 1 and await bookings(db, pets[0]):
                break
            await session.run(user_input="Sim, pode marcar.")
        pets = await new_pets(db)
        assert len(pets) == 1, f"expected one pet registered, got {pets}"
        assert await bookings(db, pets[0]) == [at_ten(monday)]


async def a_taken_time_gets_alternatives():
    monday = slot(hour=10)
    full = [(BOOK, (RAFAEL, THOR, HELENA, monday)), (BOOK, (BEATRIZ, PIPOCA, CAIO, monday))]
    async with phone_call(MARINA, *full) as (session, judge, db):
        result = await session.run(
            user_input=f"Quero marcar a Luna na {spoken(monday)} às 10h, com qualquer veterinário."
        )
        assert await bookings(db, LUNA) == []
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(judge, intent="Makes clear that 10h is not available and offers at least one other time.")
        )


async def an_emergency_goes_to_reception():
    async with phone_call() as (session, _, _db):
        # No phone line in a text eval: the transfer itself is mocked, the decision is not.
        mock_tools(VetAgent, {"transfer_to_reception": lambda: '{"status": "transferred"}'}, session=session)
        result = await session.run(user_input="Socorro, meu cachorro comeu chocolate e está passando mal!")
        result.expect.contains_function_call(name="transfer_to_reception")


SCENARIOS = [
    books_only_after_the_caller_confirms,
    a_change_of_mind_moves_the_appointment,
    someone_elses_appointment_stays_put,
    dates_are_read_with_the_right_weekday,
    offered_dates_say_the_day_of_the_month,
    a_new_caller_registers_then_books,
    a_taken_time_gets_alternatives,
    an_emergency_goes_to_reception,
]


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="evals call OpenAI")
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.__name__)
def test_eval(scenario, capsys):
    failures = []
    CONVERSATIONS.clear()
    for _ in range(ROUNDS):
        try:
            asyncio.run(scenario())
        except AssertionError as e:
            failures.append(str(e).strip())
    ttfts = [t for ts, _ in CONVERSATIONS for t in ts]
    cost = statistics.mean(c for _, c in CONVERSATIONS)
    with capsys.disabled():
        print(
            f"\n  {scenario.__name__}: {ROUNDS - len(failures)}/{ROUNDS} with {MODEL}"
            f" | LLM ttft p50 {statistics.median(ttfts):.2f}s | US$ {cost:.4f} per conversation"
        )
        for failure in failures:
            print("    " + failure.replace("\n", "\n      "))
    assert not failures
