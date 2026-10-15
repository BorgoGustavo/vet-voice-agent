"""Double booking is prevented by Postgres constraints. These tests try to break that.

They drop and rebuild the vetvoice schema, so they only ever use TEST_DATABASE_URL
(default: a local throwaway database) and never DATABASE_URL.
"""

import os
import statistics
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg
import pytest

from db import reset

DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost/vetvoice_test")
CALLERS = 12  # simultaneous callers in every race
ROUNDS = int(os.environ.get("RACE_ROUNDS", "100"))  # crank it up to hunt for rare interleavings
DEADLOCK_STALL = 0.8  # seconds. Postgres waits deadlock_timeout (1 s) before breaking a deadlock.

BOOK = "select vetvoice.book_appointment(%s, %s, %s, %s)"
RESCHEDULE = "select vetvoice.reschedule_appointment(%s, %s, %s, %s)"
CANCEL = "select vetvoice.cancel_appointment(%s, %s)"
FIND = "select vetvoice.find_slots(%s, %s, %s)"
CONTEXT = "select vetvoice.call_context(%s)"
REGISTER = "select vetvoice.register_pet(%s, %s, %s, %s)"

# What application code usually does: look, then write. Used only as a negative control.
NAIVE_BOOK = """
create function vetvoice.naive_book(p_pet_id bigint, p_vet_id int, p_start timestamp) returns text
language plpgsql set search_path = vetvoice as $$
begin
  if exists (select 1 from appointments
              where vet_id = p_vet_id and status = 'booked' and during && slot_range(p_start)) then
    return 'slot_taken';
  end if;
  insert into appointments (vet_id, pet_id, during) values (p_vet_id, p_pet_id, slot_range(p_start));
  return 'booked';
end $$
"""

VET_A, VET_B = 1, 2
ANA, BRUNO = 1, 2  # clients
ANA_PETS = list(range(1, CALLERS + 1))
THOR = CALLERS + 1  # Bruno's pet


def connect(autocommit=True):
    # UTC like a Supabase session, so a missing time zone conversion fails here too.
    return psycopg.connect(DSN, autocommit=autocommit, prepare_threshold=None, options="-c timezone=UTC")


def call(conn, sql, *args):
    return conn.execute(sql, args).fetchone()[0]


def slot(hour=10, minute=0, days=0):
    """A bookable clinic-local time: next Monday (never today) plus `days`."""
    now = datetime.now(ZoneInfo("America/Sao_Paulo")).replace(tzinfo=None)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    monday = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
    return monday + timedelta(days=days, hours=hour, minutes=minute)


def rebuild():
    reset(DSN, seed=False)
    with connect() as c:
        c.execute("insert into vetvoice.vets (name) values ('Vet A'), ('Vet B')")
        c.execute(
            "insert into vetvoice.clients (name, phone_e164) values"
            " ('Ana', '+5511900000001'), ('Bruno', '+5511900000002')"
        )
        c.execute(
            "insert into vetvoice.pets (client_id, name, species)"
            " select 1, 'pet ' || n, 'dog' from generate_series(1, %s) n",
            (CALLERS,),
        )
        c.execute("insert into vetvoice.pets (client_id, name, species) values (2, 'Thor', 'dog')")


@pytest.fixture(scope="session", autouse=True)
def schema():
    rebuild()


@pytest.fixture()
def db():
    with connect() as c:
        c.execute("truncate vetvoice.appointments restart identity")
        yield c


@pytest.fixture(scope="session")
def callers():
    """One warm connection per caller, so a race is not staggered by connection setup."""
    conns = [connect() for _ in range(CALLERS)]
    for c in conns:
        c.execute("select 1")
    yield conns
    for c in conns:
        c.close()


def race(conns, calls):
    """Fire one statement per caller at the same instant.

    `calls` is a list of (sql, args). Returns (results, seconds).
    """
    n = len(calls)
    barrier = threading.Barrier(n)
    results, seconds = [None] * n, [0.0] * n

    def run(i):
        barrier.wait()
        started = time.perf_counter()
        try:
            results[i] = call(conns[i], calls[i][0], *calls[i][1])
        except psycopg.Error as e:  # surface it in the assertion instead of dying in a thread
            results[i] = {"status": f"error: {e}"}
        seconds[i] = time.perf_counter() - started

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, seconds


def booked(db):
    return db.execute(
        "select id, pet_id, vet_id, to_char(lower(during) at time zone 'America/Sao_Paulo', 'HH24:MI')"
        " from vetvoice.appointments where status = 'booked' order by id"
    ).fetchall()


def report(capsys, label, seconds):
    ms = sorted(s * 1000 for s in seconds)
    stalled = sum(s >= DEADLOCK_STALL for s in seconds)
    with capsys.disabled():
        print(
            f"\n  {label}: n={len(ms)}  p50 {statistics.median(ms):.1f} ms"
            f"  p95 {ms[int(len(ms) * 0.95) - 1]:.1f} ms  max {ms[-1]:.1f} ms  stalled {stalled}"
        )


# --- two callers, one slot -------------------------------------------------------------


@pytest.mark.parametrize("finish, expected", [("commit", "slot_taken"), ("rollback", "booked")])
def test_second_caller_waits_for_the_first_to_finish(db, finish, expected):
    with connect(autocommit=False) as a, connect() as b:  # a keeps its transaction open
        assert call(a, BOOK, ANA, 1, VET_A, slot())["status"] == "booked"
        result = {}
        waiting = threading.Thread(target=lambda: result.update(call(b, BOOK, BRUNO, THOR, VET_A, slot())))
        waiting.start()
        waiting.join(0.5)
        assert waiting.is_alive(), "the second caller must block on the first one's uncommitted booking"
        getattr(a, finish)()
        waiting.join(5)
        assert result["status"] == expected


def test_twelve_callers_one_slot_exactly_one_wins(db, callers, capsys):
    losers = []
    for _ in range(ROUNDS):
        results, seconds = race(callers, [(BOOK, (ANA, pet, VET_A, slot())) for pet in ANA_PETS])
        assert Counter(r["status"] for r in results) == {"booked": 1, "slot_taken": CALLERS - 1}
        assert len(booked(db)) == 1
        losers += [s for r, s in zip(results, seconds) if r["status"] == "slot_taken"]
        db.execute("delete from vetvoice.appointments")
    report(capsys, "book, time the losers waited", losers)
    assert max(losers) < DEADLOCK_STALL


def test_without_the_constraint_the_same_race_double_books(db, callers, capsys):
    """Negative controls: if either stops double booking, the races above prove nothing."""
    db.execute("alter table vetvoice.appointments drop constraint vet_not_double_booked")
    try:
        # book_appointment has no hidden check of its own. The constraint was doing all the work.
        results, _ = race(callers, [(BOOK, (ANA, pet, VET_A, slot())) for pet in ANA_PETS])
        assert sum(r["status"] == "booked" for r in results) > 1
        db.execute("delete from vetvoice.appointments")

        # Look-then-write loses the same race, so the callers really do overlap in time.
        db.execute(NAIVE_BOOK)
        double_booked = 0
        for _ in range(ROUNDS):
            results, _ = race(
                callers, [("select vetvoice.naive_book(%s, %s, %s)", (pet, VET_A, slot())) for pet in ANA_PETS]
            )
            double_booked += results.count("booked") > 1
            db.execute("delete from vetvoice.appointments")
        with capsys.disabled():
            print(f"\n  look-then-write, no constraint: double booked in {double_booked} of {ROUNDS} rounds")
        assert double_booked > 0
    finally:
        rebuild()


def test_twelve_reschedules_into_one_slot_exactly_one_wins(db, callers, capsys):
    losers = []
    for _ in range(ROUNDS // 5):
        ids = [
            call(db, BOOK, ANA, pet, VET_A, slot(hour=8 + i // 2, minute=30 * (i % 2)))["appointment"]["id"]
            for i, pet in enumerate(ANA_PETS)
        ]
        results, seconds = race(callers, [(RESCHEDULE, (ANA, id_, VET_A, slot(hour=15))) for id_ in ids])
        assert Counter(r["status"] for r in results) == {"rescheduled": 1, "slot_taken": CALLERS - 1}
        assert len(booked(db)) == CALLERS  # every loser still has the slot it started with
        losers += [s for r, s in zip(results, seconds) if r["status"] == "slot_taken"]
        db.execute("delete from vetvoice.appointments")
    report(capsys, "reschedule, time the losers waited", losers)
    assert max(losers) < DEADLOCK_STALL


def test_bookings_and_reschedules_racing_for_one_slot_exactly_one_wins(db, callers, capsys):
    """Half the callers book the slot while the other half move an existing appointment into it."""
    half = CALLERS // 2
    losers = []
    for _ in range(ROUNDS // 5):
        ids = [
            call(db, BOOK, ANA, pet, VET_A, slot(hour=8 + i // 2, minute=30 * (i % 2)))["appointment"]["id"]
            for i, pet in enumerate(ANA_PETS[:half])
        ]
        calls = [(RESCHEDULE, (ANA, id_, VET_A, slot(hour=15))) for id_ in ids]
        calls += [(BOOK, (ANA, pet, VET_A, slot(hour=15))) for pet in ANA_PETS[half:]]
        results, seconds = race(callers, calls)
        statuses = Counter(r["status"] for r in results)
        assert statuses["booked"] + statuses["rescheduled"] == 1 and statuses["slot_taken"] == CALLERS - 1
        assert [row[3] for row in booked(db)].count("15:00") == 1
        losers += [s for r, s in zip(results, seconds) if r["status"] == "slot_taken"]
        db.execute("delete from vetvoice.appointments")
    report(capsys, "book vs reschedule, time the losers waited", losers)
    assert max(losers) < DEADLOCK_STALL


# --- the same request twice --------------------------------------------------------------


def test_repeating_a_request_returns_the_same_appointment(db):
    first = call(db, BOOK, ANA, 1, VET_A, slot())
    again = call(db, BOOK, ANA, 1, VET_A, slot())
    assert first["status"] == again["status"] == "booked"
    assert first["appointment"]["id"] == again["appointment"]["id"]
    assert len(booked(db)) == 1


def test_the_same_request_twelve_times_at_once_books_once(db, callers):
    results, _ = race(callers, [(BOOK, (ANA, 1, VET_A, slot()))] * CALLERS)
    assert {r["status"] for r in results} == {"booked"}
    assert len({r["appointment"]["id"] for r in results}) == 1
    assert len(booked(db)) == 1


def test_same_pet_same_time_other_vet_is_not_a_second_booking(db):
    first = call(db, BOOK, ANA, 1, VET_A, slot())["appointment"]
    other_vet = call(db, BOOK, ANA, 1, VET_B, slot())
    assert other_vet["status"] == "pet_already_booked"
    assert other_vet["appointment"]["id"] == first["id"]
    moved = call(db, RESCHEDULE, ANA, first["id"], VET_B, slot())
    assert (moved["status"], moved["appointment"]["vet_id"]) == ("rescheduled", VET_B)
    assert len(booked(db)) == 1


# --- reschedule and cancel ---------------------------------------------------------------


def test_rescheduling_into_a_taken_slot_keeps_the_old_one(db):
    mine = call(db, BOOK, ANA, 1, VET_A, slot(hour=10))["appointment"]["id"]
    call(db, BOOK, BRUNO, THOR, VET_A, slot(hour=11))
    refused = call(db, RESCHEDULE, ANA, mine, VET_A, slot(hour=11))
    assert refused["status"] == "slot_taken" and refused["alternatives"]
    assert (mine, 1, VET_A, "10:00") in booked(db)

    for _ in range(2):  # moving to a free slot, then the same request again
        moved = call(db, RESCHEDULE, ANA, mine, VET_A, slot(hour=14))
        assert (moved["status"], moved["appointment"]["id"]) == ("rescheduled", mine)
    assert (mine, 1, VET_A, "14:00") in booked(db)


def test_cancelling_frees_the_slot_and_can_be_repeated(db):
    mine = call(db, BOOK, ANA, 1, VET_A, slot())["appointment"]["id"]
    free = "select count(*) from vetvoice.available_slots(%s::date, %s::date + 1)"
    assert db.execute(free, (slot(), slot())).fetchone()[0] == 39  # 20 slots x 2 vets, minus this one

    for _ in range(2):
        assert call(db, CANCEL, ANA, mine)["status"] == "cancelled"
    assert db.execute(free, (slot(), slot())).fetchone()[0] == 40

    rebooked = call(db, BOOK, BRUNO, THOR, VET_A, slot())
    assert rebooked["status"] == "booked" and rebooked["appointment"]["id"] != mine


def test_a_caller_cannot_touch_someone_elses_appointment(db):
    anas = call(db, BOOK, ANA, 1, VET_A, slot())["appointment"]["id"]
    assert call(db, CANCEL, BRUNO, anas)["status"] == "not_found"
    assert call(db, RESCHEDULE, BRUNO, anas, VET_A, slot(hour=11))["status"] == "not_found"
    assert call(db, BOOK, BRUNO, 1, VET_A, slot(hour=11))["status"] == "invalid"  # pet 1 is Ana's
    assert booked(db) == [(anas, 1, VET_A, "10:00")]


def test_past_appointments_can_be_neither_cancelled_nor_moved(db):
    past = db.execute(
        "insert into vetvoice.appointments (vet_id, pet_id, during) values (%s, 1, vetvoice.slot_range(%s)) returning id",
        (VET_A, slot(days=-28)),
    ).fetchone()[0]
    assert call(db, CANCEL, ANA, past)["status"] == "not_found"
    assert call(db, RESCHEDULE, ANA, past, VET_A, slot())["status"] == "not_found"
    assert booked(db) == [(past, 1, VET_A, "10:00")]


# --- finding a slot ----------------------------------------------------------------------


def test_find_slots_offers_the_first_free_times_one_vet_each(db):
    call(db, BOOK, ANA, 1, VET_A, slot(hour=8))
    call(db, BOOK, ANA, 2, VET_B, slot(hour=8))  # 08:00 is full
    call(db, BOOK, ANA, 3, VET_A, slot(hour=8, minute=30))
    found = call(db, FIND, slot(hour=0), slot(hour=12), None)
    assert [s["starts_at"][11:] for s in found] == ["08:30", "09:00", "09:30", "10:00", "10:30", "11:00"]
    assert [s["vet_id"] for s in found] == [VET_B] + [VET_A] * 5
    with_vet_a = call(db, FIND, slot(hour=8), slot(hour=10), VET_A)
    assert [s["starts_at"][11:] for s in with_vet_a] == ["09:00", "09:30"]


def test_find_slots_clamps_an_absurd_window(db):
    started = time.perf_counter()
    found = call(db, FIND, datetime(2000, 1, 1), datetime(2100, 1, 1), None)
    assert time.perf_counter() - started < 0.1  # unclamped it takes ~650 ms
    now = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y-%m-%dT%H:%M")
    assert len(found) == 6 and found[0]["starts_at"] > now


def test_a_full_day_still_gets_alternatives(db):
    saturday = slot(hour=8, days=5)
    for i in range(8):  # Saturday is 08:00 to 12:00
        call(db, BOOK, ANA, 1, VET_A, saturday + timedelta(minutes=30 * i))
        call(db, BOOK, ANA, 2, VET_B, saturday + timedelta(minutes=30 * i))
    taken = call(db, BOOK, BRUNO, THOR, VET_A, saturday)
    assert taken["status"] == "slot_taken" and len(taken["alternatives"]) == 4
    assert not any(a["starts_at"].startswith(f"{saturday:%Y-%m-%d}") for a in taken["alternatives"])


# --- who is calling ----------------------------------------------------------------------


def test_the_agent_knows_the_caller_before_saying_hello(db):
    db.execute(
        "insert into vetvoice.appointments (vet_id, pet_id, during) values (%s, %s, vetvoice.slot_range(%s))",
        (VET_A, THOR, slot(days=-28)),
    )
    upcoming = call(db, BOOK, BRUNO, THOR, VET_B, slot(hour=11))["appointment"]
    call(db, CANCEL, BRUNO, call(db, BOOK, BRUNO, THOR, VET_A, slot(hour=14))["appointment"]["id"])

    context = call(db, CONTEXT, "+5511900000002")
    assert context["client"] == {"id": BRUNO, "name": "Bruno"}
    assert context["pets"] == [{"id": THOR, "name": "Thor", "species": "dog"}]
    assert context["appointments"] == [upcoming]  # not the past one, not the cancelled one
    assert [v["id"] for v in context["vets"]] == [VET_A, VET_B]

    stranger = call(db, CONTEXT, "+5511999999999")
    assert stranger["client"] is None and stranger["pets"] == stranger["appointments"] == []


NEW_CALLER = "+5511988887777"


def test_a_new_caller_registers_once(db):
    try:
        first = call(db, REGISTER, NEW_CALLER, " Carla ", "Bolinha", "gato")
        again = call(db, REGISTER, NEW_CALLER, "Carla", "Bolinha", "gato")
        assert first == again and first["status"] == "registered"
        assert first["client"]["name"] == "Carla" and [p["name"] for p in first["pets"]] == ["Bolinha"]
        another = call(db, REGISTER, NEW_CALLER, None, "Nina", "cachorro")  # a known owner needs no name
        assert [p["name"] for p in another["pets"]] == ["Bolinha", "Nina"]
    finally:
        db.execute("delete from vetvoice.clients where phone_e164 = %s", (NEW_CALLER,))


@pytest.mark.parametrize(
    "phone, owner, pet, species, reason",
    [
        pytest.param(None, "Carla", "Bolinha", "gato", "no_caller_id", id="hidden number"),
        pytest.param("anonymous", "Carla", "Bolinha", "gato", "no_caller_id", id="not a phone number"),
        pytest.param(NEW_CALLER, "  ", "Bolinha", "gato", "missing_owner_name", id="new owner without a name"),
        pytest.param(NEW_CALLER, "Carla", "", "gato", "missing_pet_name_or_species", id="pet without a name"),
        pytest.param(NEW_CALLER, "Carla", "Bolinha", None, "missing_pet_name_or_species", id="pet without species"),
    ],
)
def test_registration_refuses_what_it_cannot_store(db, phone, owner, pet, species, reason):
    assert call(db, REGISTER, phone, owner, pet, species) == {"status": "invalid", "reason": reason}
    assert call(db, CONTEXT, NEW_CALLER)["client"] is None


# --- untrusted input ---------------------------------------------------------------------


def test_ten_in_the_morning_local_is_one_pm_utc(db):
    call(db, BOOK, ANA, 1, VET_A, slot(hour=10))
    utc = db.execute("select lower(during) at time zone 'UTC' from vetvoice.appointments").fetchone()[0]
    assert (utc.hour, utc.minute) == (13, 0)


@pytest.mark.parametrize(
    "start",
    [
        pytest.param(slot(hour=22), id="after closing"),
        pytest.param(slot(hour=10, minute=15), id="off the 30 minute grid"),
        pytest.param(slot(hour=12, days=5), id="Saturday after noon"),
        pytest.param(slot(hour=10, days=6), id="Sunday"),
        pytest.param(slot(hour=10, days=-28), id="in the past"),
        pytest.param(slot(hour=10, days=70), id="beyond the 60 day horizon"),
    ],
)
def test_unbookable_times_are_refused(db, start):
    assert call(db, BOOK, ANA, 1, VET_A, start)["status"] == "invalid"
    assert call(db, RESCHEDULE, ANA, 1, VET_A, start)["status"] == "invalid"
    assert booked(db) == []


@pytest.mark.parametrize(
    "during",
    [
        pytest.param("[2030-01-07 10:15-03, 2030-01-07 10:45-03)", id="off the grid"),
        pytest.param("[2030-01-07 10:00-03, 2030-01-07 11:00-03)", id="an hour long"),
        pytest.param("[2030-01-07 10:00-03, 2030-01-07 10:30-03]", id="closed upper bound"),
        pytest.param("[2030-01-07 10:00-03,)", id="unbounded"),
        pytest.param("empty", id="empty range"),  # exclusion constraints ignore empty ranges
    ],
)
def test_the_table_itself_rejects_malformed_slots(db, during):
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute("insert into vetvoice.appointments (vet_id, pet_id, during) values (1, 1, %s::tstzrange)", (during,))
