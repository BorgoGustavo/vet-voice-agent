"""The agent's voice: OpenAI text to speech, hedged against a stalled request."""

import asyncio
import logging

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, tts
from livekit.agents.utils import shortuuid
from livekit.plugins import openai

logger = logging.getLogger("vet-agent")

TTS_VOICE = "nova"  # picked by ear among coral, shimmer, sage and nova; coral sounded harsh
TTS_INSTRUCTIONS = (
    "Fale em português do Brasil com voz calma, suave e gentil, como uma recepcionista simpática. "
    "Ritmo tranquilo, sem ênfases fortes."
)
HEDGE_AFTER = 1.5  # seconds without audio from the voice before the backup is asked too


class HedgedTTS(tts.TTS):
    """Text to speech with a hedged request ("The Tail at Scale", Dean and Barroso).

    The hourly probe caught gpt-4o-mini-tts taking 10 s, 54 s or past a 30 s timeout to start
    4 replies out of 10, and the plugin waits up to 30 s per attempt: the caller hears silence
    and hangs up. Here, if the primary has not started speaking after `hedge_after` seconds
    (or fails), the backup is asked as well and whichever speaks first is played. The extra
    request is paid only on the slow ones.
    """

    def __init__(self, primary: tts.TTS, backup: tts.TTS, hedge_after: float) -> None:
        assert (primary.sample_rate, primary.num_channels) == (backup.sample_rate, backup.num_channels)
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=primary.sample_rate,
            num_channels=primary.num_channels,
        )
        self.primary, self.backup, self.hedge_after = primary, backup, hedge_after
        for t in (primary, backup):  # usage and cost come from the request that ran
            t.on("metrics_collected", lambda m: self.emit("metrics_collected", m))

    @property
    def model(self) -> str:
        return f"hedged:{self.primary.model}"  # unpriced on purpose: the inner metrics carry the cost

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        return _HedgedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        await asyncio.gather(self.primary.aclose(), self.backup.aclose())


class _HedgedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        hedged: HedgedTTS = self._tts
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=hedged.sample_rate,
            num_channels=hedged.num_channels,
            mime_type="audio/pcm",
        )
        once = APIConnectOptions(max_retry=0, timeout=self._conn_options.timeout)  # this stream retries as a whole
        streams = [hedged.primary.synthesize(self._input_text, conn_options=once)]
        firsts = [asyncio.ensure_future(anext(streams[0]))]
        try:
            await asyncio.wait(firsts, timeout=hedged.hedge_after)
            if not firsts[0].done() or firsts[0].exception():
                logger.info("tts hedge: primary silent after %.1f s, asking the backup", hedged.hedge_after)
                streams.append(hedged.backup.synthesize(self._input_text, conn_options=once))
                firsts.append(asyncio.ensure_future(anext(streams[1])))
            async for first in asyncio.as_completed(firsts):
                if first.exception() is None:
                    break
            else:
                raise firsts[-1].exception()
            winner = streams[firsts.index(first)]
            if len(streams) > 1:
                logger.info("tts hedge: %s spoke first", "backup" if winner is streams[1] else "primary")
            output_emitter.push(first.result().frame.data.tobytes())
            async for audio in winner:
                output_emitter.push(audio.frame.data.tobytes())
            output_emitter.flush()
        finally:
            for f in firsts:
                f.cancel()
            await asyncio.gather(*(s.aclose() for s in streams), return_exceptions=True)


def make_tts() -> HedgedTTS:
    # pcm: raw audio, nothing to decode, the format OpenAI recommends for the fastest start.
    # mp3 left gaps ("slow audio generation") that the caller heard as stutter. tts-1 is the
    # backup: slower to start (~2 s) but it never stalled in the probe. Same voice, no tone
    # instructions, so a rescued sentence can sound slightly different.
    return HedgedTTS(
        primary=openai.TTS(voice=TTS_VOICE, instructions=TTS_INSTRUCTIONS, response_format="pcm"),
        backup=openai.TTS(model="tts-1", voice=TTS_VOICE, response_format="pcm"),
        hedge_after=HEDGE_AFTER,
    )
