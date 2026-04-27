"""
video_editor.py
===============
PersonaVid — Automated video editing pipeline.

Takes a raw or Runway-generated video and produces a platform-ready
vertical (9:16) clip with:
  - Dynamic 9:16 crop (computed from probed dimensions, never hardcoded)
  - Color grade (warm tone, boosted contrast — matches Carter footage aesthetic)
  - Text overlay (hook line at top, product CTA at bottom)
  - Watermark crop (bottom-right 15% strip removed)
  - Platform encode presets (TikTok / Instagram / Pinterest)

Rules from the PersonaVid brain:
  - Always write drawtext content to a temp file (textfile=) — never inline
  - Always probe input dimensions before any crop/scale
  - Always separate audio stream — never let filter chain drop it
  - Use -filter_complex when audio and video are both touched
  - Log ffmpeg stderr to timestamped log file on every error

Usage:
    from video_editor import edit_video
    out = edit_video("input.mp4", hook_text="he claimed it as his",
                     cta_text="Moon Lamp — link in bio", platform="tiktok")
"""

import json
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import ffmpeg

ROOT = Path(__file__).parent
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

PLATFORM_PRESETS = {
    "tiktok":    {"codec": "libx264", "bitrate": "1M",   "ext": "mp4"},
    "instagram": {"codec": "libx264", "bitrate": "2.5M", "ext": "mp4"},
    "pinterest": {"codec": "libx264", "bitrate": "1.5M", "ext": "mp4"},
    "preview":   {"codec": "libx264", "bitrate": "2M",   "ext": "mp4"},
}


def probe_video(input_path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", input_path],
        capture_output=True, text=True, timeout=30,
    )
    data = json.loads(result.stdout)
    video_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    duration = float(data["format"].get("duration", 0))
    return {
        "width":     int(video_stream["width"]),
        "height":    int(video_stream["height"]),
        "duration":  duration,
        "has_audio": has_audio,
    }


def _write_text_file(text: str) -> Path:
    p = Path(tempfile.mktemp(suffix=".txt", prefix="cg_caption_"))
    p.write_text(text, encoding="utf-8")
    return p


def _log_error(stderr: bytes, label: str) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = LOGS / f"ffmpeg_error_{stamp}_{label}.log"
    log.write_bytes(stderr)
    print(f"   FFmpeg error logged → {log.name}")


def edit_video(
    input_path: str,
    hook_text: str = "",
    cta_text: str = "",
    platform: str = "tiktok",
    output_path: str | None = None,
) -> str:
    """
    Edit a video for platform publishing.
    Returns path to the output file.
    """
    preset = PLATFORM_PRESETS.get(platform, PLATFORM_PRESETS["tiktok"])
    info = probe_video(input_path)
    w, h = info["width"], info["height"]

    # ── Compute 9:16 crop (center, dynamic) ────────────────────────────────────
    target_w = int(h * 9 / 16)
    if target_w > w:
        # Source is already narrower than 9:16 — pad instead of crop
        crop_filter = f"scale={w}:-2,pad={target_w}:{h}:(ow-iw)/2:0:black"
    else:
        x_offset = int((w - target_w) / 2)
        crop_filter = f"crop={target_w}:{h}:{x_offset}:0,scale=1080:1920"

    # ── Color grade (warm, cozy — matches Carter footage) ──────────────────────
    grade_filter = "eq=brightness=0.03:contrast=1.08:saturation=1.15,curves=r='0/0 0.5/0.55 1/1'"

    # ── Watermark strip — crop bottom 4% to remove Runway watermark ───────────
    wm_filter = "crop=iw:ih*0.96:0:0"

    # ── Combine video filters ───────────────────────────────────────────────────
    vf_chain = f"{crop_filter},{wm_filter},{grade_filter}"

    # ── Text overlays via temp files ────────────────────────────────────────────
    tmp_files = []
    drawtext_filters = []

    if hook_text:
        hook_file = _write_text_file(hook_text)
        tmp_files.append(hook_file)
        drawtext_filters.append(
            f"drawtext=textfile='{hook_file}'"
            ":fontcolor=white:fontsize=54:x=(w-text_w)/2:y=80"
            ":shadowcolor=black:shadowx=2:shadowy=2"
            ":font='DejaVu Sans Bold'"
        )

    if cta_text:
        cta_file = _write_text_file(cta_text)
        tmp_files.append(cta_file)
        drawtext_filters.append(
            f"drawtext=textfile='{cta_file}'"
            ":fontcolor=white:fontsize=38:x=(w-text_w)/2:y=h-120"
            ":shadowcolor=black:shadowx=2:shadowy=2"
            ":font='DejaVu Sans'"
        )

    if drawtext_filters:
        vf_chain = vf_chain + "," + ",".join(drawtext_filters)

    # ── Output path ─────────────────────────────────────────────────────────────
    if output_path is None:
        uid = uuid.uuid4().hex[:8]
        out_dir = ROOT / "outputs"
        out_dir.mkdir(exist_ok=True)
        output_path = str(out_dir / f"{platform}_{uid}.{preset['ext']}")

    # ── Build ffmpeg command ─────────────────────────────────────────────────────
    print(f"   Editing → {Path(output_path).name} [{platform}]")
    print(f"   Source: {w}x{h}, {info['duration']:.1f}s, audio: {info['has_audio']}")

    try:
        inp = ffmpeg.input(input_path)
        video = inp.video.filter_multi_output("null") if not info["has_audio"] else inp.video

        if info["has_audio"]:
            out = ffmpeg.output(
                inp.video, inp.audio, output_path,
                vf=vf_chain,
                vcodec=preset["codec"],
                video_bitrate=preset["bitrate"],
                profile__v="baseline", level="3.1",
                pix_fmt="yuv420p",
                acodec="aac", audio_bitrate="128k",
                movflags="+faststart",
            )
        else:
            out = ffmpeg.output(
                inp.video, output_path,
                vf=vf_chain,
                vcodec=preset["codec"],
                video_bitrate=preset["bitrate"],
                profile__v="baseline", level="3.1",
                pix_fmt="yuv420p",
                movflags="+faststart",
                an=None,
            )

        ffmpeg.run(out, overwrite_output=True, quiet=True)

    except ffmpeg.Error as e:
        _log_error(e.stderr, platform)
        _run_subprocess(input_path, output_path, vf_chain, preset, info)
    finally:
        for f in tmp_files:
            try:
                f.unlink()
            except Exception:
                pass

    print(f"   ✅ Saved → {output_path}")
    return output_path


def _run_subprocess(input_path: str, output_path: str,
                     vf_chain: str, preset: dict, info: dict) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf_chain,
        "-vcodec", preset["codec"],
        "-b:v", preset["bitrate"],
        "-profile:v", "baseline", "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-t", "60",
    ]
    if info["has_audio"]:
        cmd += ["-acodec", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        _log_error(result.stderr, "subprocess_fallback")
        raise RuntimeError(f"FFmpeg failed: {result.stderr.decode()[-500:]}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 video_editor.py <input_video> [hook_text] [cta_text] [platform]")
        sys.exit(1)
    inp = sys.argv[1]
    hook = sys.argv[2] if len(sys.argv) > 2 else "he claimed it as his"
    cta  = sys.argv[3] if len(sys.argv) > 3 else "Moon Lamp — link in bio"
    plat = sys.argv[4] if len(sys.argv) > 4 else "tiktok"
    out  = edit_video(inp, hook_text=hook, cta_text=cta, platform=plat)
    print(f"\nDone: {out}")
