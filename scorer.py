"""
scorer.py
=========
PersonaVid — Video quality scorer using Claude Vision.

Extracts evenly-spaced frames from a generated video and scores them
against the Continuum Goods brand rubric. Used for rejection sampling
and prompt bank feedback.
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

SCORE_THRESHOLD = 65  # minimum overall score to pass rejection sampling


def extract_frames(video_path: str, n_frames: int = 6) -> list[bytes]:
    """Extract n evenly-spaced frames from video as JPEG bytes."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        duration = 5.0  # fallback if probe fails

    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n_frames):
            t = duration * (i + 0.5) / n_frames
            frame_path = f"{tmp}/frame_{i:02d}.jpg"
            subprocess.run(
                ["ffmpeg", "-ss", str(t), "-i", video_path,
                 "-vframes", "1", "-q:v", "2", frame_path, "-y"],
                capture_output=True,
            )
            p = Path(frame_path)
            if p.exists() and p.stat().st_size > 0:
                frames.append(p.read_bytes())
    return frames


def score_video(video_path: str, prompt: str, user_id: str = "continuum_goods") -> dict:
    """
    Score a generated video against the brand rubric using Claude Vision.

    Dimensions (0-100 each):
      cat_recognized    — is this specifically Carter/Cash vs generic cat
      location_appropriate — cozy home / brand-fitting setting
      lighting_quality  — warm natural vs harsh/artificial
      motion_quality    — calm handheld feel vs jarring/unnatural
      brand_alignment   — overall continuum_goods aesthetic

    overall = weighted composite (cat_recognized weighted most heavily)

    Returns dict with all scores + passed (bool) + notes.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print(f"  [Scorer] Extracting frames from {Path(video_path).name}...")
    frames = extract_frames(video_path)

    if not frames:
        print("  [Scorer] No frames extracted — scoring failed")
        return {"error": "no_frames", "overall": 0, "passed": False}

    print(f"  [Scorer] Scoring {len(frames)} frames with Claude Vision...")

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
        "text": f"""Score these AI-generated video frames for Continuum Goods brand quality.

Brand mascots: Carter (large orange tabby cat) or Cash (black-and-white tuxedo cat).
Target aesthetic: warm natural lighting, cozy home settings, calm handheld-feel motion, boho/modern decor.
Platform: TikTok/Instagram Reels (9:16 vertical).

Original generation prompt: "{prompt}"

Score each dimension 0-100. Return ONLY valid JSON — no markdown, no explanation:
{{
  "cat_recognized": <0-100, 100=clearly Carter or Cash specifically, 50=generic cat, 0=no cat>,
  "location_appropriate": <0-100, 100=cozy home/brand-fitting, 0=wrong setting>,
  "lighting_quality": <0-100, 100=warm natural, 0=harsh/flat/artificial>,
  "motion_quality": <0-100, 100=calm handheld, 0=jarring/unnatural/static>,
  "brand_alignment": <0-100, overall continuum_goods vibe>,
  "overall": <weighted: cat_recognized*0.35 + location*0.2 + lighting*0.15 + motion*0.15 + brand*0.15>,
  "notes": "<one sentence: biggest strength OR weakness to guide prompt mutation>"
}}""",
    })

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        scores = json.loads(raw[start:end])

    scores["video_path"] = video_path
    scores["prompt"] = prompt
    scores["passed"] = scores.get("overall", 0) >= SCORE_THRESHOLD

    status = "PASS" if scores["passed"] else "FAIL"
    print(f"  [Scorer] {status} — overall={scores.get('overall', 0):.0f} "
          f"cat={scores.get('cat_recognized', 0):.0f} "
          f"location={scores.get('location_appropriate', 0):.0f} "
          f"| {scores.get('notes', '')}")

    return scores


def mutate_prompt(prompt: str, scores: dict) -> str:
    """
    Use scorer feedback + brain rules to rewrite a failing prompt.
    Called during rejection sampling when score < threshold.
    """
    from brain import video_brain
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    weaknesses = []
    if scores.get("cat_recognized", 100) < 50:
        weaknesses.append("cat not recognizable as Carter (large orange tabby) or Cash (tuxedo) — be explicit about breed, size, coloring")
    if scores.get("location_appropriate", 100) < 50:
        weaknesses.append("wrong setting — must be cozy indoor home: living room, shelf, window sill, blanket")
    if scores.get("lighting_quality", 100) < 50:
        weaknesses.append("lighting wrong — specify warm natural window light or soft lamp glow, no overhead/studio light")
    if scores.get("motion_quality", 100) < 50:
        weaknesses.append("motion unnatural — add 'handheld documentary feel, gentle natural camera sway'")
    if scores.get("brand_alignment", 100) < 50:
        weaknesses.append("off-brand — needs warm, lived-in, cozy cat-mom aesthetic")

    feedback = scores.get("notes", "")
    weakness_str = "; ".join(weaknesses) if weaknesses else feedback

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=video_brain(),
        messages=[{
            "role": "user",
            "content": (
                f"Rewrite this failing video prompt to fix these issues: {weakness_str}\n\n"
                f"Original: {prompt}\n\n"
                f"Output ONLY the new prompt. Under 120 words. Plain text."
            ),
        }],
    )
    mutated = response.content[0].text.strip()
    print(f"  [Scorer] Mutated prompt: {mutated[:80]}...")
    return mutated


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 scorer.py <video_path> <prompt>")
        sys.exit(1)
    result = score_video(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))
