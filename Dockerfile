# The image LiveKit Cloud builds and runs in us-east, next to OpenAI and the database.
# Trimmed from `lk agent dockerfile`: every dependency ships as a wheel and the VAD and
# turn-detector weights come inside livekit-local-inference, so nothing compiles and
# nothing downloads.
FROM ghcr.io/astral-sh/uv:0.11.14-python3.13-trixie-slim
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 RECORDINGS_DIR=/tmp/recordings
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY *.py ./
USER 10001
CMD [".venv/bin/python", "agent.py", "start"]
