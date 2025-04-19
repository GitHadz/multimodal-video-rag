#!/usr/bin/env python
"""
Download → transcribe → segment → sample key‑frames.
Outputs go to data/.
"""
import argparse, json, os, subprocess
from pathlib import Path

import ffmpeg
from faster_whisper import WhisperModel
from tqdm import tqdm

# ----------------------- CLI -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True,
                    help="YouTube URL or local file path")
parser.add_argument("--fps", type=float, default=1.0,
                    help="Key‑frame sampling rate (frames/sec)")
args = parser.parse_args()

DATA   = Path("data"); RAW = DATA / "raw"; FRM = DATA / "frames"
for p in (RAW, FRM): p.mkdir(parents=True, exist_ok=True)

# ------------------- download ----------------------
if args.video.startswith("http"):
    print("▶ Downloading video …")
    subprocess.run(
        ["yt-dlp", "-f", "mp4", "-o", RAW / "%(id)s.%(ext)s", args.video],
        check=True,
    )
    video_path = next(RAW.glob("*.mp4"))
else:
    video_path = Path(args.video).resolve()
print("✓ Video ready at", video_path)

# ------------------- whisper -----------------------
print("🎙  Transcribing with faster‑whisper base …")
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, _ = model.transcribe(str(video_path), beam_size=5)

chunks, buf, start_ts = [], [], None
for seg in tqdm(segments):
    if start_ts is None:
        start_ts = seg.start
    buf.append(seg.text.strip())
    if seg.end - start_ts >= 10:       # 10‑s chunks
        chunks.append(dict(
            id=len(chunks), start=start_ts, end=seg.end,
            text=" ".join(buf)
        ))
        buf, start_ts = [], None

(DATA/"chunks.jsonl").write_text(
    "\n".join(json.dumps(c) for c in chunks) + "\n")
print(f"✓ {len(chunks)} transcript chunks saved")

# ------------------- frame sampling ---------------
print("🖼  Sampling key‑frames …")
(ffmpeg
 .input(str(video_path))
 .filter("fps", fps=args.fps)
 .output(str(FRM/"%06d.jpg"), **{"q:v": 2})
 .overwrite_output()
 .run(quiet=True))
print("✓ Frames saved to", FRM)

# ------------------- save meta --------------------
meta = {"fps": args.fps}
(Path("data")/"meta.json").write_text(json.dumps(meta))
print("✓ meta.json written")
