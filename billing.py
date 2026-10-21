"""What a call costs in model usage, from the usage report of the session."""

# USD, from developers.openai.com/api/docs/pricing on 2026-10-19. Per 1M tokens, or per
# minute of audio for the live transcription model. Telephony minutes are not included.
PRICES = {
    "gpt-4.1-mini": {"input": 0.40, "cached": 0.10, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "cached": 0.50, "output": 8.00},
    "gpt-4o-mini-tts": {"input": 0.60, "output": 12.00},
    "tts-1": {"character": 15.00},  # per 1M characters
    "gpt-live-transcribe": {"minute": 0.017},
}


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
