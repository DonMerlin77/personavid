"""
analyzer.py
===========
Video Style Analyzer for PersonaVid.

Ingests uploaded videos, extracts frames at key scene boundaries,
and uses Claude Vision to build a style fingerprint — the "learning"
layer that makes generated content feel like YOUR footage.

What it extracts:
  - Color palette (dominant colors, warmth, saturation)
  - Shot composition (wide/medium/close-up ratio, rule-of-thirds usage)
  - Lighting style (natural/studio/golden hour/mixed)
  - Subject matter (what appears in the frame most)
  - Pacing (cuts-per-minute, clip duration patterns)
  - Motion style (handheld/tripod/pan/static)
  - Mood/aesthetic keywords

The output is a StyleProfile JSON saved to profiles/{user_id}.json
This profile is the input to the generator.

Run:  python3 analyzer.py --video path/to/video.mp4 --user my_brand
"""

import os
import sys
import json
import base64
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ROOT      = Path(__file__).parent
PROFILES  = ROOT / "profiles"
UPLOADS   = ROOT / "uploads"
PROFILES.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# How many frames to sample per video for style analysis
FRAMES_PER_VIDEO = 8

# ── Frame Extraction ────────────────────────────────────────────────────────────

def extract_key_frames(video_path: str, n_frames: int = FRAMES_PER_VIDEO) -> list:
    """
    Extract n evenly-spaced frames from a video.
    Returns list of (timestamp_sec, frame_as_jpg_bytes).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    duration_sec = total_frames / fps

    # Sample evenly, skip first and last 5% to avoid intros/outros
    start  = int(total_frames * 0.05)
    end    = int(total_frames * 0.95)
    step   = max(1, (end - start) // n_frames)
    sample = list(range(start, end, step))[:n_frames]

    frames = []
    for frame_idx in sample:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        timestamp = frame_idx / fps
        # Resize to max 640px wide to keep Claude API calls lean
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frames.append((timestamp, buf.tobytes()))

    cap.release()
    return frames, duration_sec


def get_video_metadata(video_path: str) -> dict:
    """Extract basic video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    meta = {
        "width":    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":      round(cap.get(cv2.CAP_PROP_FPS), 2),
        "frames":   int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    meta["duration_sec"] = round(meta["frames"] / max(meta["fps"], 1), 1)
    cap.release()
    return meta


# ── Claude Vision Analysis ──────────────────────────────────────────────────────

def analyze_frames_with_claude(frames: list, video_filename: str) -> dict:
    """
    Send sampled frames to Claude Vision.
    Returns a structured style analysis dict.
    """
    # Build the content blocks: interleave frames with timestamps
    content = [
        {
            "type": "text",
            "text": f"""You are a professional video director and cinematographer analyzing footage for a video style fingerprinting system.

I'm showing you {len(frames)} frames sampled from a video called "{video_filename}".
Analyze the visual style and return a JSON object with exactly these fields:

{{
  "color_palette": ["list of 3-5 dominant hex colors or color names"],
  "color_temperature": "warm|cool|neutral",
  "saturation": "high|medium|low|desaturated",
  "lighting_style": "natural|studio|golden_hour|artificial|mixed",
  "lighting_notes": "one sentence description of the lighting",
  "shot_types": ["wide", "medium", "close-up"],  // list only what appears
  "dominant_shot": "wide|medium|close-up|macro",
  "composition_style": "centered|rule_of_thirds|dynamic|minimal",
  "motion_style": "static|handheld|smooth_pan|timelapse|mixed",
  "subjects": ["list of main subjects: person, product, food, landscape, etc."],
  "setting": "indoor|outdoor|studio|mixed",
  "aesthetic_keywords": ["5-8 single words: cinematic, cozy, professional, raw, etc."],
  "mood": "one word: energetic|calm|professional|playful|dramatic|aspirational",
  "editing_pace": "fast|medium|slow",
  "brand_feel": "one sentence: what kind of brand would use this footage style",
  "ad_suitability": ["product", "lifestyle", "tutorial", "testimonial"]  // which ad types this style fits
}}

Return ONLY the JSON object, no other text."""
        }
    ]

    for i, (timestamp, frame_bytes) in enumerate(frames):
        b64 = base64.standard_b64encode(frame_bytes).decode("utf-8")
        content.append({
            "type": "text",
            "text": f"Frame {i+1} (at {timestamp:.1f}s):"
        })
        content.append({
            "type": "image",
            "source": {
                "type":       "base64",
                "media_type": "image/jpeg",
                "data":       b64,
            }
        })

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": content}]
    )

    text = response.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ── Style Profile ───────────────────────────────────────────────────────────────

def load_profile(user_id: str) -> dict:
    """Load an existing style profile, or return empty."""
    path = PROFILES / f"{user_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "user_id":   user_id,
        "created_at": datetime.now().isoformat(),
        "videos_analyzed": 0,
        "style_fingerprint": {},
        "video_history": [],
    }


def merge_analysis_into_profile(profile: dict, analysis: dict,
                                  video_path: str, meta: dict) -> dict:
    """
    Merge a new video's analysis into the cumulative style fingerprint.
    Uses a simple weighted average approach — newer videos weighted slightly higher.
    """
    profile["videos_analyzed"] += 1
    profile["updated_at"] = datetime.now().isoformat()

    # Log this video
    profile["video_history"].append({
        "filename":    Path(video_path).name,
        "analyzed_at": datetime.now().isoformat(),
        "duration_sec": meta.get("duration_sec", 0),
        "analysis":    analysis,
    })

    # Build cumulative fingerprint
    fp = profile.get("style_fingerprint", {})
    n  = profile["videos_analyzed"]

    # Merge list fields by frequency count
    for list_field in ["color_palette", "subjects", "aesthetic_keywords",
                        "ad_suitability", "shot_types"]:
        if list_field in analysis:
            counts = fp.get(f"{list_field}_counts", {})
            for item in analysis[list_field]:
                counts[item] = counts.get(item, 0) + 1
            fp[f"{list_field}_counts"] = counts
            # Top items by frequency
            sorted_items = sorted(counts.items(), key=lambda x: -x[1])
            fp[list_field] = [k for k, _ in sorted_items[:8]]

    # Merge scalar fields by plurality vote
    for scalar_field in ["color_temperature", "saturation", "lighting_style",
                          "dominant_shot", "composition_style", "motion_style",
                          "setting", "mood", "editing_pace"]:
        if scalar_field in analysis:
            votes = fp.get(f"{scalar_field}_votes", {})
            val   = analysis[scalar_field]
            votes[val] = votes.get(val, 0) + 1
            fp[f"{scalar_field}_votes"] = votes
            # Winner
            fp[scalar_field] = max(votes.items(), key=lambda x: x[1])[0]

    # Text fields — keep most recent
    for text_field in ["lighting_notes", "brand_feel"]:
        if text_field in analysis:
            fp[text_field] = analysis[text_field]

    profile["style_fingerprint"] = fp
    return profile


def save_profile(profile: dict, user_id: str):
    """Save style profile to disk."""
    path = PROFILES / f"{user_id}.json"
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    return path


# ── Photo Analysis ──────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp"}

def analyze_photo(photo_path: str, user_id: str,
                   label: str = "subject",
                   verbose: bool = True) -> dict:
    """
    Analyze a single photo (product shot, character reference, etc.)
    and store it in the profile's reference library.

    label: what this photo is — "carter", "product", "setting", etc.
    These become reference descriptions injected into every future prompt.
    """
    from PIL import Image as PILImage
    import io

    path = Path(photo_path)
    if not path.exists():
        raise FileNotFoundError(f"Photo not found: {photo_path}")

    if verbose:
        print(f"\n📸  Analyzing photo: {path.name} [{label}]")
        print(f"    User: {user_id}")

    # Load and resize for API
    img = PILImage.open(str(path)).convert("RGB")
    w, h = img.size
    if w > 800:
        scale = 800 / w
        img = img.resize((800, int(h * scale)), PILImage.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    # Ask Claude to describe this reference specifically
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Describe this reference image in precise visual detail for use in AI video generation prompts.
First identify what type of subject this is (cat, dog, person, product, room, etc.) — be explicit.
Focus on: exact colors, fur/texture/material patterns, distinctive features, size, shape, lighting on the subject.
Be specific enough that an AI video model could recreate this exact subject.
Write 2-3 sentences maximum. Start with the subject type, e.g. "A cat..." or "A product..." or "A room...". No preamble."""
                },
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}
                }
            ]
        }]
    )

    description = response.content[0].text.strip()

    if verbose:
        print(f"    Description: {description}")

    # Store in profile references
    profile = load_profile(user_id)
    refs = profile.get("references", {})
    refs[label] = {
        "filename":    path.name,
        "description": description,
        "added_at":    datetime.now().isoformat(),
    }
    profile["references"] = refs
    save_profile(profile, user_id)

    if verbose:
        print(f"    ✅ Saved reference '{label}' to profile")

    return profile


def get_references(user_id: str) -> dict:
    """Return all photo references for a user."""
    profile = load_profile(user_id)
    return profile.get("references", {})


# ── Main Entry Point ────────────────────────────────────────────────────────────

def analyze_video(video_path: str, user_id: str,
                   verbose: bool = True) -> dict:
    """
    Full pipeline: extract frames → Claude Vision → merge into profile → save.
    Accepts both video files and image files.
    Returns the updated profile.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Route images to photo analyzer
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return analyze_photo(video_path, user_id, label=path.stem, verbose=verbose)

    if verbose:
        print(f"\n🎬  Analyzing: {path.name}")
        print(f"    User: {user_id}")

    # ── Extract frames ──────────────────────────────────────────────────────────
    if verbose:
        print(f"    Extracting {FRAMES_PER_VIDEO} key frames...")
    frames, duration = extract_key_frames(str(path), FRAMES_PER_VIDEO)
    meta = get_video_metadata(str(path))
    if verbose:
        print(f"    Duration: {duration:.1f}s | "
              f"Resolution: {meta.get('width')}x{meta.get('height')} | "
              f"FPS: {meta.get('fps')}")
        print(f"    Sending {len(frames)} frames to Claude Vision...")

    # ── Claude analysis ─────────────────────────────────────────────────────────
    analysis = analyze_frames_with_claude(frames, path.name)
    if verbose:
        print(f"    Analysis complete:")
        print(f"      Mood:       {analysis.get('mood', '?')}")
        print(f"      Motion:     {analysis.get('motion_style', '?')}")
        print(f"      Lighting:   {analysis.get('lighting_style', '?')}")
        print(f"      Aesthetic:  {', '.join(analysis.get('aesthetic_keywords', [])[:4])}")

    # ── Merge into profile ──────────────────────────────────────────────────────
    profile = load_profile(user_id)
    profile = merge_analysis_into_profile(profile, analysis, str(path), meta)
    save_path = save_profile(profile, user_id)

    if verbose:
        print(f"\n    ✅ Profile updated → {save_path}")
        print(f"    Total videos analyzed: {profile['videos_analyzed']}")

    return profile


def get_style_summary(user_id: str) -> str:
    """
    Return a human-readable style summary for a user.
    Used by the generator to build prompts.
    """
    profile = load_profile(user_id)
    fp = profile.get("style_fingerprint", {})

    if not fp:
        return "No style profile yet. Upload some videos first."

    lines = [
        f"Style profile for '{user_id}' ({profile['videos_analyzed']} videos analyzed):",
        f"  Mood:          {fp.get('mood', 'unknown')}",
        f"  Motion:        {fp.get('motion_style', 'unknown')}",
        f"  Lighting:      {fp.get('lighting_style', 'unknown')} — {fp.get('lighting_notes', '')}",
        f"  Color feel:    {fp.get('color_temperature', 'unknown')}, {fp.get('saturation', 'unknown')} saturation",
        f"  Dominant shot: {fp.get('dominant_shot', 'unknown')}",
        f"  Aesthetic:     {', '.join(fp.get('aesthetic_keywords', [])[:6])}",
        f"  Subjects:      {', '.join(fp.get('subjects', [])[:5])}",
        f"  Brand feel:    {fp.get('brand_feel', 'unknown')}",
        f"  Best for ads:  {', '.join(fp.get('ad_suitability', []))}",
    ]
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PersonaVid — Video Style Analyzer")
    parser.add_argument("--video",   required=False, help="Path to video file to analyze")
    parser.add_argument("--user",    default="default", help="User/brand ID for the profile")
    parser.add_argument("--summary", action="store_true", help="Print style summary for a user")
    args = parser.parse_args()

    if args.summary:
        print(get_style_summary(args.user))
    elif args.video:
        profile = analyze_video(args.video, args.user, verbose=True)
        print()
        print(get_style_summary(args.user))
    else:
        parser.print_help()
