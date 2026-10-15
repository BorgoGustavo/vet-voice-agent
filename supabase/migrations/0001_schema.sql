-- Vet clinic scheduling.
--
-- The rule "one vet, one pet, one slot" lives in the two exclusion constraints on
-- appointments, not in application code. Whatever a caller, the voice agent or a
-- second server does, Postgres decides who gets the slot.

begin;

create extension if not exists btree_gist;  -- "=" on integer ids inside a gist index, next to "&&" on ranges

create schema vetvoice;
set local search_path = vetvoice;

create table vets (
  id   int generated always as identity primary key,
  name text not null
);

create table clients (
  id         bigint generated always as identity primary key,
  name       text not null,
  phone_e164 text not null unique,  -- callers are identified by the number they call from
  created_at timestamptz not null default now()
);

create table pets (
  id        bigint generated always as identity primary key,
  client_id bigint not null references clients on delete cascade,
  name      text not null,
  species   text not null,
  unique (client_id, name)
);

create table calls (
  id              text primary key,  -- LiveKit room name, one per phone call
  twilio_call_sid text,
  from_number     text,
  client_id       bigint references clients on delete set null,
  scenario        text,              -- set by the scripted test calls
  started_at      timestamptz not null default now(),
  ended_at        timestamptz,
  outcome         text,
  transcript      jsonb,
  usage           jsonb,
  cost_usd        numeric(10, 5)
);

create table appointments (
  id         bigint generated always as identity primary key,
  vet_id     int       not null references vets,
  pet_id     bigint    not null references pets on delete cascade,
  during     tstzrange not null,
  status     text      not null default 'booked' check (status in ('booked', 'cancelled')),
  call_id    text references calls on delete set null,
  created_at timestamptz not null default now(),

  -- Two callers, one slot: only one insert survives. Cancelling frees the slot
  -- because cancelled rows fall outside the predicate.
  constraint vet_not_double_booked
    exclude using gist (vet_id with =, during with &&) where (status = 'booked'),

  -- A pet cannot be in two places at once. This is also what makes a repeated
  -- request harmless: the second insert finds the first one and stops.
  constraint pet_not_double_booked
    exclude using gist (pet_id with =, during with &&) where (status = 'booked'),

  -- Exclusion constraints skip NULL and empty ranges, so pin the shape down:
  -- half-open, exactly 30 minutes, starting on the half hour.
  constraint thirty_minute_slot check (coalesce(
    lower_inc(during) and not upper_inc(during)
    and upper(during) - lower(during) = interval '30 minutes'
    and extract(epoch from lower(during)) % 1800 = 0, false))
);

-- Every function below takes local wall-clock time (America/Sao_Paulo) and converts
-- explicitly. Supabase sessions run in UTC: a bare cast would turn 10:00 into 07:00.

create function slot_range(p_start timestamp) returns tstzrange
language sql stable as $$
  select tstzrange(p_start at time zone 'America/Sao_Paulo',
                   (p_start + interval '30 minutes') at time zone 'America/Sao_Paulo', '[)')
$$;

-- ponytail: clinic hours are hard-coded (Mon to Fri 08:00-18:00, Sat 08:00-12:00, at most
-- 60 days ahead). Move them to a table when holidays or a second clinic show up.
create function is_bookable(p_start timestamp) returns boolean
language sql stable as $$
  select p_start > now() at time zone 'America/Sao_Paulo'
     and p_start < (now() at time zone 'America/Sao_Paulo') + interval '60 days'
     and extract(epoch from p_start) % 1800 = 0
     and case extract(isodow from p_start)
           when 7 then false
           when 6 then p_start::time >= '08:00' and p_start::time < '12:00'
           else        p_start::time >= '08:00' and p_start::time < '18:00'
         end
$$;

-- Free slots starting in [p_from, p_to). The series is clamped to the bookable window
-- because the bounds come from an LLM: "2000 to 2100" must not generate 3.5M rows.
create function available_slots(p_from timestamp, p_to timestamp)
returns table (vet_id int, vet_name text, starts_at timestamp)
language sql stable set search_path = vetvoice as $$
  select v.id, v.name, s.t
    from vets v
   cross join generate_series(date_trunc('hour', greatest(p_from, now() at time zone 'America/Sao_Paulo')),
                              least(p_to, (now() at time zone 'America/Sao_Paulo') + interval '60 days'),
                              interval '30 minutes') as s(t)
   where s.t >= p_from and s.t < p_to
     and is_bookable(s.t)
     and not exists (select 1 from appointments a
                      where a.vet_id = v.id and a.status = 'booked' and a.during && slot_range(s.t))
   order by s.t, v.id
$$;

-- The four free slots closest to the one that was just lost, so the agent can
-- offer an alternative without another round trip. Two days either side, so a
-- full day still gets an answer.
create function nearby_slots(p_start timestamp) returns jsonb
language sql stable set search_path = vetvoice as $$
  select coalesce(jsonb_agg(to_jsonb(s)), '[]'::jsonb)
    from (select f.vet_id, f.vet_name as vet, to_char(f.starts_at, 'YYYY-MM-DD"T"HH24:MI') as starts_at
            from available_slots(p_start - interval '2 days', p_start + interval '2 days') f
           order by abs(extract(epoch from f.starts_at - p_start)), f.vet_id
           limit 4) s
$$;

-- What the agent searches with: the first six free times in a window. One vet per
-- time unless the caller asked for one, so the list stays short enough to read aloud.
create function find_slots(p_from timestamp, p_to timestamp, p_vet_id int default null) returns jsonb
language sql stable set search_path = vetvoice as $$
  select coalesce(jsonb_agg(to_jsonb(s) order by s.starts_at), '[]'::jsonb)
    from (select distinct on (f.starts_at) f.vet_id, f.vet_name as vet,
                 to_char(f.starts_at, 'YYYY-MM-DD"T"HH24:MI') as starts_at
            from available_slots(p_from, p_to) f
           where p_vet_id is null or f.vet_id = p_vet_id
           order by f.starts_at, f.vet_id
           limit 6) s
$$;

create function appointment_info(a appointments) returns jsonb
language sql stable set search_path = vetvoice as $$
  select jsonb_build_object(
    'id', a.id,
    'pet', (select name from pets where id = a.pet_id),
    'vet_id', a.vet_id,
    'vet', (select name from vets where id = a.vet_id),
    'starts_at', to_char(lower(a.during) at time zone 'America/Sao_Paulo', 'YYYY-MM-DD"T"HH24:MI'))
$$;

create function appointment_json(p_status text, a appointments) returns jsonb
language sql stable set search_path = vetvoice as $$
  select jsonb_build_object('status', p_status, 'appointment', appointment_info(a))
$$;

create function book_appointment(
  p_client_id bigint, p_pet_id bigint, p_vet_id int, p_start timestamp, p_call_id text default null)
returns jsonb language plpgsql set search_path = vetvoice as $$
declare
  v_during tstzrange := slot_range(p_start);
  v_appt   appointments;
begin
  -- The arguments come from an LLM: untrusted input.
  if is_bookable(p_start) is not true then
    return jsonb_build_object('status', 'invalid', 'reason', 'not_bookable');
  end if;
  if not exists (select 1 from pets where id = p_pet_id and client_id = p_client_id)
     or not exists (select 1 from vets where id = p_vet_id) then
    return jsonb_build_object('status', 'invalid', 'reason', 'unknown_pet_or_vet');
  end if;

  -- No "is it free?" query first: the constraints are the check. With no conflict
  -- target both exclusion constraints arbitrate, and a concurrent insert for the
  -- same slot makes this one wait for the other transaction instead of failing.
  insert into appointments (vet_id, pet_id, during, call_id)
  values (p_vet_id, p_pet_id, v_during, p_call_id)
  on conflict do nothing
  returning * into v_appt;
  if v_appt.id is not null then
    return appointment_json('booked', v_appt);
  end if;

  -- Something already holds the slot. Reading it back only decides what to tell
  -- the caller, never who gets the slot.
  select * into v_appt from appointments
   where status = 'booked' and pet_id = p_pet_id and during && v_during;
  if v_appt.id is null then
    return jsonb_build_object('status', 'slot_taken', 'alternatives', nearby_slots(p_start));
  end if;
  if v_appt.vet_id = p_vet_id and v_appt.during = v_during then
    return appointment_json('booked', v_appt);  -- the same request again gets the same answer
  end if;
  return appointment_json('pet_already_booked', v_appt);
end $$;

create function reschedule_appointment(
  p_client_id bigint, p_appointment_id bigint, p_vet_id int, p_start timestamp)
returns jsonb language plpgsql set search_path = vetvoice as $$
declare
  v_appt appointments;
begin
  if is_bookable(p_start) is not true then
    return jsonb_build_object('status', 'invalid', 'reason', 'not_bookable');
  end if;
  if not exists (select 1 from vets where id = p_vet_id) then
    return jsonb_build_object('status', 'invalid', 'reason', 'unknown_pet_or_vet');
  end if;
  -- An UPDATE cannot use ON CONFLICT, so two of them aiming at one slot wait on each
  -- other until Postgres breaks the deadlock, a second at a time. Measured with twelve
  -- at once: 7% of the losers stalled, the worst for 11.8 s. This lock queues the
  -- contenders for one slot. It decides nothing, the constraint still does.
  perform pg_advisory_xact_lock(p_vet_id, (extract(epoch from p_start) / 1800)::int);
  begin
    -- One UPDATE moves the appointment. If the new slot is taken the constraint
    -- rejects the row and the old slot is untouched: the caller never ends up with nothing.
    update appointments a
       set during = slot_range(p_start), vet_id = p_vet_id
     where a.id = p_appointment_id and a.status = 'booked' and lower(a.during) > now()
       and exists (select 1 from pets p where p.id = a.pet_id and p.client_id = p_client_id)
    returning a.* into v_appt;
  exception when exclusion_violation or deadlock_detected then
    return jsonb_build_object('status', 'slot_taken', 'alternatives', nearby_slots(p_start));
  end;
  if v_appt.id is null then
    return jsonb_build_object('status', 'not_found');
  end if;
  return appointment_json('rescheduled', v_appt);
end $$;

create function cancel_appointment(p_client_id bigint, p_appointment_id bigint)
returns jsonb language plpgsql set search_path = vetvoice as $$
declare
  v_appt appointments;
begin
  -- Cancelling an already cancelled appointment is the same answer again, not an error.
  -- One that already happened is history: to the caller it does not exist.
  update appointments a
     set status = 'cancelled'
   where a.id = p_appointment_id and lower(a.during) > now()
     and exists (select 1 from pets p where p.id = a.pet_id and p.client_id = p_client_id)
  returning a.* into v_appt;
  if v_appt.id is null then
    return jsonb_build_object('status', 'not_found');
  end if;
  return appointment_json('cancelled', v_appt);
end $$;

-- Everything the agent needs before it says hello, in one round trip: who is calling
-- (by caller ID), their pets, their upcoming appointments and the vets. An unknown
-- number gets "client": null.
create function call_context(p_phone text) returns jsonb
language sql stable set search_path = vetvoice as $$
  select jsonb_build_object(
    'client', (select jsonb_build_object('id', c.id, 'name', c.name) from clients c where c.phone_e164 = p_phone),
    'pets', coalesce((select jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name, 'species', p.species) order by p.id)
                        from pets p join clients c on c.id = p.client_id
                       where c.phone_e164 = p_phone), '[]'::jsonb),
    'appointments', coalesce((select jsonb_agg(appointment_info(a) order by a.during)
                                from appointments a
                                join pets p on p.id = a.pet_id
                                join clients c on c.id = p.client_id
                               where c.phone_e164 = p_phone and a.status = 'booked' and lower(a.during) > now()), '[]'::jsonb),
    'vets', (select jsonb_agg(jsonb_build_object('id', v.id, 'name', v.name) order by v.id) from vets v))
$$;

-- Adds a pet, and its owner too when the number is new. Repeating it changes nothing.
-- The phone comes from caller ID (a hidden number fails here); the names come from an LLM.
create function register_pet(p_phone text, p_owner_name text, p_pet_name text, p_species text)
returns jsonb language plpgsql set search_path = vetvoice as $$
declare
  v_client_id bigint;
begin
  if (p_phone ~ '^\+[1-9][0-9]{7,14}$') is not true then
    return jsonb_build_object('status', 'invalid', 'reason', 'no_caller_id');
  end if;
  if (length(trim(p_pet_name)) between 1 and 60 and length(trim(p_species)) between 1 and 60) is not true then
    return jsonb_build_object('status', 'invalid', 'reason', 'missing_pet_name_or_species');
  end if;

  select id into v_client_id from clients where phone_e164 = p_phone;
  if v_client_id is null then
    if (length(trim(p_owner_name)) between 1 and 80) is not true then
      return jsonb_build_object('status', 'invalid', 'reason', 'missing_owner_name');
    end if;
    insert into clients (name, phone_e164) values (trim(p_owner_name), p_phone)
    on conflict (phone_e164) do nothing;  -- the same number registering twice at once
    select id into v_client_id from clients where phone_e164 = p_phone;
  end if;

  insert into pets (client_id, name, species) values (v_client_id, trim(p_pet_name), trim(p_species))
  on conflict (client_id, name) do nothing;
  return jsonb_build_object('status', 'registered') || call_context(p_phone);
end $$;

-- Supabase serves tables over HTTP with a public key. Nothing here is meant for that:
-- row level security on with zero policies, so only the owning role (this app) gets in.
alter table vets         enable row level security;
alter table clients      enable row level security;
alter table pets         enable row level security;
alter table calls        enable row level security;
alter table appointments enable row level security;

commit;
