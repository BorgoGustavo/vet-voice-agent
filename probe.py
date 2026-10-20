"""How fast the models answer right now, so latency can be seen by time of day.

    uv run --env-file .env probe.py >> latency.jsonl

An hourly GitHub Actions schedule runs it and appends the line to the latency-data branch.
Variants are interleaved, so both sides of each comparison see the same moment:
- LLM time to first token, standard vs Fast mode (service_tier=priority), with the agent's
  real prompt and tools and a request that makes it call a tool, as on a real turn;
- TTS time to first audio, gpt-4o-mini-tts (the agent's voice) vs tts-1.
A failed request is recorded, not raised: an outage is data too.
"""

import json
import os
import time
from datetime import UTC, datetime

from livekit.agents import llm
from openai import OpenAI

from agent import LLM_MODEL, TTS_INSTRUCTIONS, TTS_VOICE, VetAgent, instructions

SAMPLES = 5
CONTEXT = {
    "client": {"id": 1, "name": "Marina Lopes"},
    "pets": [{"id": 1, "name": "Luna", "species": "cachorro"}],
    "appointments": [],
    "vets": [{"id": 1, "name": "Dra. Helena Prado"}, {"id": 2, "name": "Dr. Caio Mendes"}],
}
CALLER = "Quero marcar a Luna na quinta às 10h."
SPOKEN = "Quinta-feira, dia 22, às 10h está livre. Confirmo?"  # short: only the first byte matters
# No retries: they would hide the latency being measured. A stuck request becomes an error.
client = OpenAI(timeout=30, max_retries=0)


def llm_first_token(tier, prompt, tools):
    """Seconds to the first token or tool call, and the tier OpenAI says it served."""
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        service_tier=tier,
        stream=True,
        tools=tools,
        parallel_tool_calls=False,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": CALLER}],
    )
    for chunk in stream:
        if chunk.choices and (chunk.choices[0].delta.content or chunk.choices[0].delta.tool_calls):
            elapsed = time.perf_counter() - start
            stream.close()  # the rest of the reply is not needed
            return elapsed, chunk.service_tier
    raise RuntimeError("empty reply")


def tts_first_audio(model):
    extra = {"instructions": TTS_INSTRUCTIONS} if model == "gpt-4o-mini-tts" else {}  # tts-1 takes none
    start = time.perf_counter()
    with client.audio.speech.with_streaming_response.create(
        model=model, voice=TTS_VOICE, input=SPOKEN, response_format="pcm", **extra
    ) as response:
        for _ in response.iter_bytes(1024):
            return time.perf_counter() - start


def main():
    prompt = instructions(CONTEXT)
    tools = [llm.utils.build_legacy_openai_schema(t) for t in VetAgent(CONTEXT).tools if not isinstance(t, llm.Toolset)]
    variants = {
        "llm_standard": lambda: llm_first_token("default", prompt, tools),
        "llm_fast": lambda: llm_first_token("priority", prompt, tools),
        "tts_gpt-4o-mini-tts": lambda: (tts_first_audio("gpt-4o-mini-tts"), None),
        "tts_tts-1": lambda: (tts_first_audio("tts-1"), None),
    }
    seconds = {name: [] for name in variants}
    served, errors = {}, {}
    for _ in range(SAMPLES):
        for name, measure in variants.items():
            try:
                elapsed, tier = measure()
                seconds[name].append(round(elapsed, 3))
                if tier:
                    served.setdefault(name, set()).add(tier)
            except Exception as e:  # noqa: BLE001
                errors.setdefault(name, []).append(type(e).__name__)
    print(
        json.dumps(
            {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "where": os.environ.get("PROBE_WHERE", "local"),
                "llm_model": LLM_MODEL,
                "seconds": seconds,
                "served_tier": {k: sorted(v) for k, v in served.items()},
                "errors": errors,
            }
        )
    )


if __name__ == "__main__":
    main()
