# latency-data

Measurements only, no code. Every hour for a week, a GitHub Actions schedule on `main` runs
[`probe.py`](https://github.com/BorgoGustavo/vet-voice-agent/blob/main/probe.py) and appends one
JSON line to `latency.jsonl`: how many seconds the models took, at that moment, to start
answering (LLM first token, standard vs Fast mode; TTS first audio, `gpt-4o-mini-tts` vs `tts-1`).
