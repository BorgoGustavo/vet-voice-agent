# /// script
# requires-python = ">=3.13"
# dependencies = ["openai", "pillow", "psycopg[binary]"]
# ///
"""Turn a recorded call into a captioned demo video.

    uv run --env-file .env demo.py call-web-1234abcd

Needs ffmpeg. Reads recordings/<call>.ogg, both voices as agent.py saved them, and the
call's transcript from the database. Each line becomes a caption in Portuguese with an
English translation, and every reply shows how long the agent took to start speaking.
Writes recordings/<call>.mp4.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import psycopg
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

W, H, CAPTION_Y = 1280, 720, 450
BG, CALLER, AGENT, TEXT, MUTED = "#0f1115", "#4fc3f7", "#ffb74d", "#f5f5f5", "#9aa0a6"
FONT, BOLD = "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def lines_of(items, t0):
    """(start, end, role, text, seconds to reply) for every spoken line, relative to the recording."""
    out = []
    for i in items:
        m = i.get("metrics") or {}
        if i.get("type") == "message" and i["role"] in ("user", "assistant") and "started_speaking_at" in m:
            text = " ".join(c for c in i["content"] if isinstance(c, str)).strip()
            if text:
                start, end = m["started_speaking_at"] - t0, m["stopped_speaking_at"] - t0
                out.append((start, end, i["role"], text, m.get("e2e_latency")))
    return out


def translate(texts):
    reply = OpenAI().chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Translate each Brazilian Portuguese line of a phone call to natural English. "
                'Answer as JSON {"en": [...]}, one item per line, same order.',
            },
            {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
        ],
    )
    en = json.loads(reply.choices[0].message.content)["en"]
    return en if len(en) == len(texts) else [""] * len(texts)


def background(path, subtitle):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((50, 30), "vet-voice-agent", font=font(34, True), fill=TEXT)
    d.text((50, 75), subtitle, font=font(20), fill=MUTED)
    d.text((50, 118), "Caller", font=font(20, True), fill=CALLER)
    d.text((50, 278), "Agent", font=font(20, True), fill=AGENT)
    img.save(path)


def caption(path, role, text, english, latency):
    img = Image.new("RGBA", (W, H - CAPTION_Y), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color, label = (AGENT, "Agent") if role == "assistant" else (CALLER, "Caller")
    if latency is not None:
        label += f"   replied in {latency:.1f} s"
    d.text((50, 0), label, font=font(22, True), fill=color)
    y = 38
    for line in textwrap.wrap(text, 62)[:3]:
        d.text((50, y), line, font=font(30), fill=TEXT)
        y += 38
    for line in textwrap.wrap(english, 80)[:2]:
        d.text((50, y + 8), line, font=font(22), fill=MUTED)
        y += 30
    img.save(path)


def main(call_id):
    recordings = Path(os.environ.get("RECORDINGS_DIR", "recordings"))
    audio = recordings / f"{call_id}.ogg"
    t0 = json.loads((recordings / f"{call_id}.json").read_text())["started_at"]
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=2, load_balance_hosts="random") as db:
        transcript, cost = db.execute(
            "select transcript, cost_usd from vetvoice.calls where id = %s", (call_id,)
        ).fetchone()
    lines = lines_of(transcript["items"], t0)
    english = translate([text for _, _, _, text, _ in lines])
    # The recording's length, which is what the viewer sees. The call row also counts the
    # wrap-up after the caller hangs up.
    probe = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio)]
    seconds = round(float(subprocess.run(probe, capture_output=True, text=True, check=True).stdout))
    subtitle = f"A real call, recorded by the agent.  {seconds // 60}:{seconds % 60:02d}  |  US$ {cost:.2f} in models"

    with tempfile.TemporaryDirectory() as tmp:
        background(f"{tmp}/bg.png", subtitle)
        inputs, chain, last = ["-loop", "1", "-framerate", "30", "-i", f"{tmp}/bg.png", "-i", str(audio)], [], "v1"
        for n, (start, end, role, text, latency) in enumerate(lines):
            caption(f"{tmp}/c{n}.png", role, text, english[n], latency)
            inputs += ["-i", f"{tmp}/c{n}.png"]
            until = lines[n + 1][0] if n + 1 < len(lines) else end + 2
            chain.append(f"[{last}][{n + 2}:v]overlay=0:{CAPTION_Y}:enable='between(t,{start:.2f},{until:.2f})'[c{n}]")
            last = f"c{n}"
        graph = ";".join(
            [
                "[1:a]asplit[aout][awave]",
                "[awave]channelsplit=channel_layout=stereo[left][right]",  # left: caller, right: agent
                f"[left]showwaves=s=1180x110:mode=cline:draw=full:scale=sqrt:rate=30:colors={CALLER}[wl]",
                f"[right]showwaves=s=1180x110:mode=cline:draw=full:scale=sqrt:rate=30:colors={AGENT}[wr]",
                "[0:v][wl]overlay=50:150:shortest=1[v0]",
                "[v0][wr]overlay=50:310[v1]",
                *chain,
            ]
        )
        out = recordings / f"{call_id}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                *inputs,
                "-filter_complex",
                graph,
                "-map",
                f"[{last}]",
                "-map",
                "[aout]",
                "-ac",
                "1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(out),
            ],
            check=True,
        )
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
