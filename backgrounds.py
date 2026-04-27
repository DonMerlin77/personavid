"""
backgrounds.py
==============
PersonaVid — Background scene library.

User drops screen recordings (Hawaii beach, fireplace, pyramids, etc.)
into references/backgrounds/. This module extracts keyframes, uses Claude
Vision to describe the scene, and injects that description into video prompts
so the AI generates content that feels like it was shot in that location.

No compositing — the scene description shapes the AI generation prompt itself.

Usage:
    from backgrounds import get_scene_context, list_backgrounds

    # Get scene context for a brief (auto-selects best matching background)
    ctx = get_scene_context("Carter near a warm cozy product")
    # ctx = {"description": "crackling stone fireplace, warm orange glow...", "source": "fireplace.mp4"}

    # Or pick a specific background
    ctx = get_scene_context(brief, background_name="hawaii_beach.mp4")

    # List what's available
    list_backgrounds()

Add a lesson:
    python3 brain.py teach "hawaii background scores well for summer product drops"
"""

import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
import anthropic

load_dotenv(Path(__file__).parent / ".env")

ROOT = Path(__file__).parent
BACKGROUNDS_DIR = ROOT / "references" / "backgrounds"
CACHE_PATH = ROOT / "data" / "background_cache.json"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)


# ── Cache ──────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


# ── Frame extraction ───────────────────────────────────────────────────────────

def _get_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 10.0


def extract_keyframes(video_path: str, n: int = 4) -> list[bytes]:
    """Extract n evenly-spaced frames from a video as JPEG bytes."""
    duration = _get_duration(video_path)
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            t = duration * (i + 0.5) / n
            out = f"{tmp}/frame_{i:02d}.jpg"
            subprocess.run(
                ["ffmpeg", "-ss", str(t), "-i", video_path,
                 "-vframes", "1", "-q:v", "2", out, "-y"],
                capture_output=True,
            )
            p = Path(out)
            if p.exists() and p.stat().st_size > 0:
                frames.append(p.read_bytes())
    return frames


# ── Scene description ──────────────────────────────────────────────────────────

def describe_scene(video_path: str, force: bool = False) -> dict:
    """
    Describe the scene in a background video using Claude Vision.
    Result is cached by filename — only re-runs if force=True or file changes.

    Returns:
        {
          "name": "fireplace.mp4",
          "setting": "crackling stone fireplace, warm orange glow...",
          "lighting": "warm, orange, flickering firelight",
          "mood": "cozy, intimate, calm",
          "prompt_context": "<ready-to-inject string for video prompt>",
          "selection_tags": ["warm", "indoor", "cozy", "fire"]
        }
    """
    cache = _load_cache()
    name = Path(video_path).name
    mtime = str(Path(video_path).stat().st_mtime)

    if not force and name in cache and cache[name].get("mtime") == mtime:
        print(f"  [Backgrounds] Using cached description for {name}")
        return cache[name]

    print(f"  [Backgrounds] Describing scene: {name}...")
    frames = extract_keyframes(video_path, n=4)
    if not frames:
        return {"name": name, "error": "no_frames", "prompt_context": ""}

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    content = []
    for frame_bytes in frames:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(frame_bytes).decode(),
            },
        })
    content.append({
        "type": "text",
        "text": """Describe this video's scene for use as a background setting in a pet/lifestyle video.
Focus on: location, lighting quality, color palette, mood, and key visual elements.

Return ONLY valid JSON:
{
  "setting": "<specific location description, 1-2 sentences, rich visual detail>",
  "lighting": "<lighting type and quality: warm/cool, natural/artificial, intensity>",
  "mood": "<3 mood words: e.g. cozy, serene, dramatic>",
  "prompt_context": "<ready-to-use phrase for a video generation prompt, under 40 words, describing the environment a cat could inhabit in this setting>",
  "selection_tags": ["<tag1>", "<tag2>", "<tag3>"]
}""",
    })

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        result = json.loads(raw[start:end])

    result["name"] = name
    result["mtime"] = mtime
    print(f"  [Backgrounds] Scene: {result.get('setting', '')[:80]}...")

    cache[name] = result
    _save_cache(cache)
    return result


# ── Background selection ───────────────────────────────────────────────────────

def list_backgrounds() -> list[dict]:
    """Return all available background videos with their cached descriptions."""
    videos = [
        p for p in BACKGROUNDS_DIR.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    cache = _load_cache()
    results = []
    for v in sorted(videos):
        entry = cache.get(v.name, {"name": v.name, "setting": "not yet described"})
        entry["path"] = str(v)
        results.append(entry)
    return results


def select_background(brief: str, backgrounds: list[dict] | None = None) -> dict | None:
    """
    Use Claude to pick the best background for a given brief.
    Returns the background dict with prompt_context, or None if no backgrounds available.
    """
    if backgrounds is None:
        backgrounds = list_backgrounds()

    # Auto-describe any undescribed backgrounds
    described = []
    for bg in backgrounds:
        if "prompt_context" not in bg or not bg["prompt_context"]:
            bg = describe_scene(bg["path"])
        described.append(bg)

    if not described:
        return None

    if len(described) == 1:
        return described[0]

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    options = "\n".join(
        f"{i+1}. {bg['name']}: {bg.get('setting', '')} [tags: {bg.get('selection_tags', [])}]"
        for i, bg in enumerate(described)
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": (
                f"Pick the best background for this video brief: '{brief}'\n\n"
                f"Options:\n{options}\n\n"
                f"Reply with ONLY the number of the best match."
            ),
        }],
    )
    try:
        idx = int(response.content[0].text.strip()) - 1
        idx = max(0, min(idx, len(described) - 1))
        chosen = described[idx]
        print(f"  [Backgrounds] Selected: {chosen['name']} — {chosen.get('mood', '')}")
        return chosen
    except (ValueError, IndexError):
        return described[0]


def get_scene_context(brief: str, background_name: str | None = None) -> dict | None:
    """
    Main entry point. Returns scene context dict ready to inject into prompt.

    If background_name given, uses that specific file.
    Otherwise auto-selects best match for the brief.
    Returns None if references/backgrounds/ is empty.
    """
    if background_name:
        path = BACKGROUNDS_DIR / background_name
        if path.exists():
            return describe_scene(str(path))
        print(f"  [Backgrounds] {background_name} not found in references/backgrounds/")
        return None

    backgrounds = list_backgrounds()
    if not backgrounds:
        return None

    return select_background(brief, backgrounds)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "describe":
        path = sys.argv[2] if len(sys.argv) > 2 else None
        if path:
            result = describe_scene(path, force=True)
            print(json.dumps(result, indent=2))
        else:
            for bg in list_backgrounds():
                print(f"  {bg['name']}: {bg.get('setting', 'not described')[:60]}")
    elif len(sys.argv) > 1 and sys.argv[1] == "select":
        brief = " ".join(sys.argv[2:]) or "cat near product, warm cozy mood"
        ctx = get_scene_context(brief)
        print(json.dumps(ctx, indent=2) if ctx else "No backgrounds available")
    else:
        bgs = list_backgrounds()
        if bgs:
            print(f"{len(bgs)} backgrounds available:")
            for bg in bgs:
                print(f"  {bg['name']}: {bg.get('setting', 'not yet described')[:70]}")
        else:
            print(f"No backgrounds yet. Drop videos into: {BACKGROUNDS_DIR}")
