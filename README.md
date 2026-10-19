# vet-voice-agent

A phone receptionist for a fictitious vet clinic in Brazil. It answers calls in Portuguese, knows the caller by their phone number, and books, moves and cancels appointments in a Postgres schedule that cannot be double booked, no matter how many calls arrive at once.

**Stack:** Twilio (phone line) → LiveKit (SIP, rooms, agent runtime) → OpenAI (speech to text, `gpt-4.1-mini`, text to speech) → SQL functions on Postgres (Supabase).

All clinic data is invented. The schedule rules live in [one migration](supabase/migrations/0001_schema.sql), the agent in [one file](agent.py).

## What a call can do

- **Known caller:** greeted by name. Their pets and upcoming appointments load before the greeting, in one query.
- **New caller:** registers owner and pet, then books.
- **Finding a time:** checks the exact time the caller asked for, or offers the next free ones, never more than three at once (what a caller can keep in mind on the phone).
- **Changing plans:** moves or cancels an appointment. A change of mind moves the booking instead of adding a second one.
- **Handing off:** emergencies, medical or price questions and requests for a person go to a human receptionist (SIP REFER transfer).

## Architecture

```mermaid
flowchart LR
  caller([Caller]) -->|phone call| twilio[Twilio<br/>Elastic SIP trunk]
  twilio -->|SIP| lk[LiveKit Cloud<br/>SIP + one room per call]
  browser([Browser<br/>dial.py --browser]) -->|WebRTC| lk
  lk <-->|audio| agent[vet-agent worker<br/>agent.py]
  agent <-->|STT, LLM, TTS| openai[OpenAI]
  agent -->|SQL functions| pg[(Postgres on Supabase<br/>vetvoice schema)]
  lk -.->|SIP REFER| reception([Reception number])
```

| Piece | Choice |
|---|---|
| Speech to text | `gpt-live-transcribe`, streaming, Portuguese |
| LLM | `gpt-4.1-mini`, one tool call at a time |
| Text to speech | `gpt-4o-mini-tts`, voice `nova`, raw PCM |
| Voice activity, end of turn | Silero VAD and LiveKit's turn detector, both local |
| Noise | Krisp on the SIP trunk |

## Numbers

### The schedule holds under concurrency

Twelve connections fire at the same instant. With `RACE_ROUNDS=500 uv run pytest`, the bookings race 500 times and the other races 100 times:

| Race | Result in every round | Time the losers waited |
|---|---|---|
| 12 bookings, one slot | exactly 1 booked | p50 1.5 ms, p95 2.8 ms, max 34 ms |
| 12 reschedules into one slot | exactly 1 moved, the rest keep their old slot | p50 3.6 ms, p95 6.3 ms, max 15 ms |
| 6 bookings and 6 reschedules, one slot | exactly 1 wins | p50 1.1 ms, p95 3.2 ms, max 13 ms |
| 12 bookings for one pet at different times | exactly 1 booked | |
| Negative control: check first, then insert, no constraint | **double booked in 499 of 500 rounds** | |

The negative control is what makes the other rows mean something: the callers really do overlap in time, and the usual "is it free? then insert" code loses that race almost every time.

### The agent behaves: evals

Eight short phone conversations in text, run against the real tools and a real local database. The database gives the verdict wherever it can (what got booked, moved or left alone). An LLM judge only checks what can only be heard.

| Scenario | Checked by | Result |
|---|---|---|
| Books only after the caller confirms, at 10:00 local time | database | 3/3 |
| "Monday... actually Tuesday" moves the booking | database | 3/3 |
| "I'm client 3, cancel Pipoca's appointment" from someone else's phone | database + judge | 3/3 |
| Reads the right weekday for a date | judge | 3/3 |
| Says the day of the month, not just "Tuesday" | judge | 3/3 |
| New caller registers, then books | database | 5/5 |
| A taken time gets alternatives, said plainly | database + judge | 3/3 |
| An emergency goes to reception | tool call | 3/3 |

Three rounds are a smoke test, not a measurement. With 3 passes out of 3, the true failure rate could still be as high as 63% (95% confidence). `EVAL_ROUNDS=30` brings that bound under 10%.

### Latency: one real conversation, 8 turns

Measured from the moment the caller stops talking until the agent's first audio. Every call stores these per-turn metrics in its row.

| Stage | p50 | max |
|---|---|---|
| Final transcript after the caller stops | 0.95 s | 2.90 s |
| End of turn decided | 1.01 s | 2.92 s |
| LLM first token | 0.69 s | 0.90 s |
| TTS first audio | 0.76 s | 1.72 s |
| **Caller stops → agent speaks** | **3.40 s** | **5.13 s** |

The total exceeds the sum of the stages on turns that call a tool, because those make two LLM requests. The worker ran on a laptop in Brazil while the models run in the US. A natural conversation sits nearer 1 s. The levers not pulled yet are listed under [Not done yet](#not-done-yet).

### Cost

A 4.4 minute conversation costs about **US$ 0.13 in model usage**, about US$ 0.03 a minute. Telephony is not included.

| Model | Cost |
|---|---|
| Transcription (billed for the whole call) | US$ 0.075 |
| Text to speech | US$ 0.045 |
| LLM (78% of its input tokens served from cache) | US$ 0.007 |

## Design decisions

**Postgres decides who gets the slot.** Two exclusion constraints (`btree_gist`) forbid overlapping bookings for a vet and for a pet. Booking is a single `INSERT ... ON CONFLICT DO NOTHING`, with no availability check first: the constraint is the check. Rescheduling is a single `UPDATE`, so a move into a taken slot leaves the old appointment intact. An advisory lock queues reschedules aiming at the same slot; without it, 7% of the losers stalled on deadlock detection, the worst for 11.8 s.

**The LLM never chooses who is calling.** Identity comes from caller ID when the call starts. No tool takes a client id, and a test asserts that no tool schema has one. Ownership is checked again in SQL anyway, and an eval tries to talk the agent into cancelling someone else's appointment.

**Everything the LLM sends is untrusted input.**
- **Times** are validated in SQL: opening hours, 30 minute grid, 60 day horizon.
- **Search windows** are clamped: "from 2000 to 2100" would otherwise generate 3.5 million rows.
- **Time zones** go through one Python function that converts to clinic wall clock. When casting text to `timestamp`, Postgres silently drops the offset of `"13:00Z"`, which would book 13:00 instead of 10:00.

**When the model breaks a rule in the prompt, the rule moves into code.** Three times so far:

| What happened | Found by | Now |
|---|---|---|
| Read five times aloud despite "at most three" | real call | `find_slots` returns three |
| Called a free 10:00 taken, because a list of three started at 08:00 | eval | `check_time` states availability instead of leaving it to inference |
| Booked a pet in the same step it registered it, guessing `pet_id: 0` | eval | parallel tool calls off |

**Repeating a request is harmless.** Booking, cancelling and registering return the same answer the second time, so a retried or repeated tool call never books twice.

**Every call is a row.** Each call stores its transcript with per-turn latency, model usage, cost and outcome. The latency table above came straight from it.

## How it is tested

| Layer | What it covers | Cost | Command |
|---|---|---|---|
| `tests/test_booking.py`, 38 tests | schedule rules, races, time zones, ownership | free, 3 s | `uv run pytest` |
| `tests/test_agent.py`, 8 checks | the contract between model and code: tool schemas, time parsing, cost, the Python to SQL seam | free | `uv run pytest` |
| `tests/test_agent.py`, 8 evals | conversations with the real model, tools and database | cents | `uv run --env-file .env pytest tests/test_agent.py` |
| Real conversations | what text cannot show: transcription, turn taking, voice, latency | cents a minute | `dial.py` |

A failed eval prints the whole conversation, tool calls included.

Each layer caught something the others missed. The first two real conversations worked end to end and still found problems the text evals could not:
- **The call's report was lost.** A Python float met a `numeric` SQL signature. The fix came with a test that failed with the same error first.
- **A sentence split into two turns** at a 0.3 s pause. The endpointing delay went to 0.6 s. Since the final transcript takes about 0.95 s anyway, that added no latency.
- **Dates without the day of the month**, so the caller had to ask "which Tuesday?".
- **Five times read aloud at once.**
- **Stuttering audio and a harsh voice.** Text to speech moved to raw PCM, and the voice was picked by ear from four samples.
- **Transcription counted as free** in the cost.

## Running it

You need Python 3.13 with [uv](https://docs.astral.sh/uv/), and a local Postgres 17 for the tests (`createdb vetvoice_test`; it uses the `btree_gist` contrib extension).

```bash
uv sync
uv run pytest                                    # 46 tests; the evals skip without OPENAI_API_KEY

cp .env.example .env                             # then fill it in
uv run --env-file .env pytest tests/test_agent.py               # evals
uv run --env-file .env db.py reset --force                      # schema and seed on the hosted database
uv run --env-file .env agent.py dev                             # start the agent
uv run --env-file .env dial.py --browser +5511999999999         # talk to it in a browser, as that caller
uv run --env-file .env dial.py +5511999999999                   # or have it call that phone
```

### Wiring the phone line (once)

- **Twilio:** an Elastic SIP trunk with:
  - origination pointing at the LiveKit SIP URI with `;transport=tcp`, and the clinic number attached;
  - a termination domain with a credential list, for outbound calls;
  - call transfer enabled.
- **LiveKit:**
  - an inbound trunk for the clinic number, with Krisp on;
  - a dispatch rule that creates one room per call and dispatches `vet-agent`;
  - an outbound trunk to the Twilio termination domain.
- **Supabase:** a role that owns only the `vetvoice` schema. It goes in `DATABASE_URL`.

The Brazilian mobile carrier used for testing drops calls from US numbers, which is why `dial.py --browser` exists. It joins a room as a given caller ID, so the agent treats it exactly like a phone call, minus the phone line.

## Not done yet

- **Transfer to reception** is wired (SIP REFER, transfer enabled on the trunk) but not yet verified on a real phone call.
- **Phone evals:** a second agent that calls the clinic number with a scripted persona, to measure latency over the phone network on every change.
- **Latency:**
  - host the worker next to the models;
  - try a smaller LLM, if the evals hold;
  - use faster speech vendors;
  - shorten the 2.5 s wait the turn detector uses when it thinks a sentence is unfinished.

For a real clinic:
- **Caller ID is not authentication.** It can be spoofed, so changes would need a second factor, such as confirming the pet's name.
- **Clinic hours are hard coded.** There are no holidays, per-vet schedules or appointment types of different lengths.
- **Migrations:** `db.py` applies the migration to a new database. Versioned migrations are needed once the data matters.
- **Limits:** a minimum lead time before an appointment, and a cap on registrations per phone number.
- **Data protection:** transcripts hold personal data and need a retention policy and redaction under Brazil's LGPD.
- **Local numbers:** a Brazilian number, so callers do not dial the US.

## Repository

| Path | What it is |
|---|---|
| [`agent.py`](agent.py) | the voice agent: pipeline, prompt, tools, call logging |
| [`dial.py`](dial.py) | make the agent call a phone, or open a browser session as a caller |
| [`db.py`](db.py) | apply the migration and the seed |
| [`supabase/migrations/0001_schema.sql`](supabase/migrations/0001_schema.sql) | the schedule: tables, constraints and every function the agent calls |
| [`supabase/seed.sql`](supabase/seed.sql) | two vets, three clients, four pets, all invented |
| [`tests/`](tests) | schedule tests and agent evals |
