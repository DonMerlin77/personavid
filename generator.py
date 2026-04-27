"""
generator.py
============
PersonaVid — AI Video Generator.

Takes a style profile (built by analyzer.py) + a generation brief,
and creates a video generation prompt that will produce output
matching YOUR visual style.

Generation backends supported:
  - Kling AI (via API — requires KLING_API_KEY)
  - Runway ML (via API — requires RUNWAY_API_KEY)
  - Prompt-only mode (outputs the crafted prompt for manual use)

The key innovation: the prompt isn't generic.
It's constructed directly from your style fingerprint — your colors,
your lighting, your motion, your subjects — so the output feels
like footage from your library, not stock video.

Run:
  python3 generator.py --user my_brand --brief "30-second kitchen gadget ad"
  python3 generator.py --user my_brand --brief "product showcase" --backend prompt
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import anthropic
import runwayml

import base64
import io

ROOT       = Path(__file__).parent
OUTPUTS    = ROOT / "outputs"
REFS_DIR   = ROOT / "references"
OUTPUTS.mkdir(exist_ok=True)
REFS_DIR.mkdir(exist_ok=True)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

KLING_API_KEY  = os.getenv("KLING_API_KEY", "")
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY", "")


# ── Prompt Builder ──────────────────────────────────────────────────────────────

def build_generation_prompt(style_fp: dict, brief: str,
                              duration_sec: int = 10) -> str:
    """
    Construct a video generation prompt from a style fingerprint + creative brief.
    This is the core "personalization" — every element is derived from your footage.
    """
    # Pull style elements
    mood        = style_fp.get("mood", "professional")
    motion      = style_fp.get("motion_style", "smooth")
    lighting    = style_fp.get("lighting_style", "natural")
    light_note  = style_fp.get("lighting_notes", "")
    color_temp  = style_fp.get("color_temperature", "warm")
    saturation  = style_fp.get("saturation", "medium")
    shot        = style_fp.get("dominant_shot", "medium")
    composition = style_fp.get("composition_style", "rule_of_thirds")
    aesthetic   = style_fp.get("aesthetic_keywords", [])
    subjects    = style_fp.get("subjects", [])
    brand_feel  = style_fp.get("brand_feel", "")
    colors      = style_fp.get("color_palette", [])

    # Map our terms to video generation terminology
    motion_map = {
        "static":     "locked-off tripod shot, no camera movement",
        "handheld":   "subtle handheld movement, organic camera shake",
        "smooth_pan": "smooth cinematic pan, fluid motion",
        "timelapse":  "timelapse, dynamic time passage",
        "mixed":      "varied camera movement, dynamic angles",
    }
    camera_motion = motion_map.get(motion, "smooth camera movement")

    lighting_map = {
        "natural":     "natural window light, soft shadows",
        "studio":      "professional studio lighting, controlled environment",
        "golden_hour": "golden hour sunlight, warm directional light",
        "artificial":  "artificial lighting, LED/lamp lit",
        "mixed":       "mixed natural and artificial lighting",
    }
    lighting_desc = lighting_map.get(lighting, "well-lit scene")
    if light_note:
        lighting_desc += f", {light_note}"

    shot_map = {
        "wide":     "wide establishing shot",
        "medium":   "medium shot",
        "close-up": "close-up detail shot",
        "macro":    "extreme close-up macro shot",
    }
    shot_desc = shot_map.get(shot, "medium shot")

    color_desc = f"{color_temp} color temperature"
    if colors:
        color_desc += f", palette of {', '.join(colors[:3])}"
    if saturation != "medium":
        color_desc += f", {saturation} saturation"

    aesthetic_str = ", ".join(aesthetic[:5]) if aesthetic else "cinematic"

    # Build the full prompt
    prompt = f"""{brief.strip()}.

Visual style: {aesthetic_str}.
Camera: {shot_desc}, {camera_motion}.
Lighting: {lighting_desc}.
Color: {color_desc}.
Mood: {mood}.
Composition: {composition} framing.
Duration: {duration_sec} seconds.
Quality: 4K, sharp focus, professional production value.
No text overlays, no watermarks."""

    if brand_feel:
        prompt += f"\nBrand aesthetic: {brand_feel}"

    return prompt


def refine_prompt_with_claude(raw_prompt: str, brief: str,
                                ad_type: str = "product",
                                max_chars: int = 950) -> str:
    """
    Use Claude to polish the generation prompt into the best possible
    video generation instruction for Kling/Runway.
    Enforces max_chars limit (default 950 — safe under Runway's 1000 cap).
    """
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""You are an expert at writing video generation prompts for AI tools like Kling AI and Runway.

Take this draft prompt and rewrite it to be maximally effective for AI video generation.
STRICT LIMIT: {max_chars} characters including spaces. Count carefully — this is a hard limit.
No markdown, no bold, no bullet points — plain text only.
Priority order if you must cut: keep subject description, lighting, camera motion, mood. Cut technical specs last.
Output ONLY the final prompt, nothing else.

Ad type: {ad_type}
Creative brief: {brief}

Draft prompt:
{raw_prompt}"""
        }]
    )
    result = response.content[0].text.strip()
    # Hard truncate as safety net
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0]
    return result


# ── Source Image Auto-Selection ────────────────────────────────────────────────

def _load_image_b64(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for a given image path."""
    from PIL import Image as PILImage
    img = PILImage.open(str(path)).convert("RGB")
    w, h = img.size
    if max(w, h) > 1024:
        scale = 1024 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"


def auto_select_source_image(final_prompt: str, user_id: str,
                              verbose: bool = True) -> str | None:
    """
    Use Claude Vision to score every image in references/ against the
    generated video prompt and return the path of the best match.

    Scoring criteria:
      - Subject match: does the image contain the right cat/product/setting?
      - Mood match: does the image tone fit the prompt?
      - Composition: would this make a good starting frame for the video?

    Returns absolute file path of winner, or None if no images available
    or no image scores above 40.
    """
    candidates = sorted(REFS_DIR.glob("*"))
    candidates = [
        p for p in candidates
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]

    if not candidates:
        if verbose:
            print("  [auto_select] No reference images found in references/ — skipping")
        return None

    if verbose:
        print(f"  [auto_select] Scoring {len(candidates)} images against prompt...")

    # Build vision message with all candidates
    content = [
        {
            "type": "text",
            "text": (
                "You are selecting the best source image to use as the first frame "
                "for an AI video generation task.\n\n"
                f"VIDEO PROMPT:\n{final_prompt}\n\n"
                f"Below are {len(candidates)} candidate images, labeled image_0 through "
                f"image_{len(candidates)-1}.\n\n"
                "For each image, score it 0–100 on how well it would serve as a starting "
                "frame for this video:\n"
                "  90-100: Perfect match — right subject, right setting, right mood\n"
                "  70-89:  Good match — right subject but different setting or mood\n"
                "  40-69:  Partial match — related but missing key elements\n"
                "  0-39:   Wrong subject or setting entirely\n\n"
                "Return ONLY a JSON object like:\n"
                '{"scores": {"image_0": 85, "image_1": 42, ...}, '
                '"winner": "image_0", "reason": "one sentence"}'
            ),
        }
    ]

    label_to_path = {}
    for i, path in enumerate(candidates):
        label = f"image_{i}"
        label_to_path[label] = path
        try:
            b64, mime = _load_image_b64(path)
            content.append({
                "type": "text",
                "text": f"{label} ({path.name}):",
            })
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })
        except Exception as e:
            if verbose:
                print(f"  [auto_select] Could not load {path.name}: {e}")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()
    try:
        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        result = json.loads(raw)
        scores = result.get("scores", {})
        winner = result.get("winner", "")
        reason = result.get("reason", "")

        if verbose:
            print(f"  [auto_select] Scores:")
            for label, score in sorted(scores.items(), key=lambda x: -x[1]):
                path = label_to_path.get(label)
                name = path.name if path else label
                print(f"    {score:3d}  {name}")
            if winner and winner in label_to_path:
                print(f"  [auto_select] Winner: {label_to_path[winner].name} — {reason}")

        if winner and winner in label_to_path:
            best_score = scores.get(winner, 0)
            if best_score >= 40:
                return str(label_to_path[winner])
            else:
                if verbose:
                    print(f"  [auto_select] Best score {best_score} < 40 — using text-only")
                return None

    except Exception as e:
        if verbose:
            print(f"  [auto_select] Could not parse scores: {e}\nRaw: {raw}")

    return None


# ── Generation Backends ─────────────────────────────────────────────────────────

def generate_with_kling(prompt: str, duration: int = 10,
                         aspect_ratio: str = "16:9") -> dict:
    """
    Submit a video generation job to Kling AI.
    Returns job info dict. Poll status separately.
    Docs: https://klingai.com/api-reference
    """
    if not KLING_API_KEY:
        raise ValueError("KLING_API_KEY not set in .env")

    resp = requests.post(
        "https://api.klingai.com/v1/videos/text2video",
        headers={
            "Authorization": f"Bearer {KLING_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "model":        "kling-v1",
            "prompt":       prompt,
            "duration":     duration,
            "aspect_ratio": aspect_ratio,
            "mode":         "std",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def generate_with_runway(prompt: str, duration: int = 10,
                          image_path: str | None = None) -> dict:
    """
    Submit a text-to-video (or image-to-video) job via Runway Gen-4.5 SDK.
    Polls until completion. Returns result dict with output URL.
    """
    if not RUNWAY_API_KEY:
        raise ValueError("RUNWAY_API_KEY not set in .env")

    runway = runwayml.RunwayML(api_key=RUNWAY_API_KEY)

    # Clamp duration to Runway's supported values (5 or 10)
    duration = 10 if duration >= 8 else 5

    if image_path:
        import base64, mimetypes
        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        data_uri = f"data:{mime};base64,{b64}"
        task = runway.image_to_video.create(
            model="gen4_turbo",
            prompt_image=data_uri,
            prompt_text=prompt,
            duration=duration,
            ratio="720:1280",
        )
    else:
        task = runway.text_to_video.create(
            model="gen4_turbo",
            prompt_text=prompt,
            duration=duration,
            ratio="720:1280",
        )

    task_id = task.id
    print(f"  Runway task submitted: {task_id}")

    task = runway.tasks.retrieve(task_id)
    while task.status not in ("SUCCEEDED", "FAILED"):
        print(f"  Status: {task.status}...")
        time.sleep(10)
        task = runway.tasks.retrieve(task_id)

    if task.status == "FAILED":
        raise RuntimeError(f"Runway task failed: {getattr(task, 'failure', 'unknown')}")

    output_url = task.output[0] if task.output else ""
    return {"id": task_id, "status": task.status, "output_url": output_url}


# ── Main Generation Pipeline ────────────────────────────────────────────────────

def generate_video(user_id: str, brief: str,
                    backend: str = "prompt",
                    duration: int = 10,
                    ad_type: str = "product",
                    verbose: bool = True) -> dict:
    """
    Full generation pipeline:
      1. Load user's style fingerprint
      2. Build personalized prompt
      3. Refine with Claude
      4. Submit to generation backend (or print prompt-only)

    Returns result dict with prompt, backend response, and output path.
    """
    from analyzer import load_profile, get_style_summary, get_references

    if verbose:
        print("\n🎬  PersonaVid Generator")
        print(f"    User:     {user_id}")
        print(f"    Brief:    {brief}")
        print(f"    Backend:  {backend}")
        print(f"    Duration: {duration}s")
        print()

    # ── Load style profile ──────────────────────────────────────────────────────
    profile = load_profile(user_id)
    fp = profile.get("style_fingerprint", {})

    if not fp and backend != "generic":
        print("⚠️  No style profile found for this user.")
        print("   Run: python3 analyzer.py --video your_video.mp4 --user", user_id)
        print("   Continuing with generic style...")

    videos_seen = profile.get("videos_analyzed", 0)
    if verbose and videos_seen > 0:
        print(f"  Using style fingerprint from {videos_seen} analyzed video(s)")
        print(f"  Mood: {fp.get('mood')} | Motion: {fp.get('motion_style')} | "
              f"Lighting: {fp.get('lighting_style')}")

    # ── Load photo references ───────────────────────────────────────────────────
    refs = get_references(user_id)
    if refs and verbose:
        print(f"  References: {', '.join(refs.keys())}")
    print()

    # ── Build prompt ────────────────────────────────────────────────────────────
    if verbose:
        print("  🧠 Building personalized prompt...")

    # Inject reference descriptions into the brief
    enriched_brief = brief
    if refs:
        ref_block = "\n".join(
            f"- {label} (use exactly as described): {r['description']}"
            for label, r in refs.items()
        )
        enriched_brief = (
            f"{brief}\n\n"
            f"IMPORTANT — use these exact visual references. "
            f"Do not change their species, type, or appearance:\n{ref_block}"
        )

    raw_prompt = build_generation_prompt(fp, enriched_brief, duration_sec=duration)
    final_prompt = refine_prompt_with_claude(raw_prompt, enriched_brief, ad_type)

    if verbose:
        print()
        print("  ── GENERATED PROMPT ──────────────────────────────────")
        print(f"  {final_prompt}")
        print("  ──────────────────────────────────────────────────────")
        print()

    result = {
        "user_id":      user_id,
        "brief":        brief,
        "ad_type":      ad_type,
        "duration":     duration,
        "backend":      backend,
        "raw_prompt":   raw_prompt,
        "final_prompt": final_prompt,
        "generated_at": datetime.now().isoformat(),
        "job":          None,
    }

    # ── Auto-select source image ────────────────────────────────────────────────
    source_image = result.get("source_image")
    if backend in ("runway", "kling") and not source_image:
        if verbose:
            print("  No source image provided — running auto-selection...")
        source_image = auto_select_source_image(final_prompt, user_id, verbose=verbose)
        result["source_image"] = source_image

    # ── Submit to backend ───────────────────────────────────────────────────────
    if backend == "prompt":
        if verbose:
            print("  📋 Prompt-only mode — copy the prompt above into Kling AI or Runway")
            print("  💡 To generate automatically: --backend kling  or  --backend runway")
        selected = auto_select_source_image(final_prompt, user_id, verbose=verbose)
        result["source_image"] = selected
        if selected and verbose:
            print(f"  Best source image: {Path(selected).name}")

    elif backend == "kling":
        if verbose:
            print("  🚀 Submitting to Kling AI...")
        job = generate_with_kling(final_prompt, duration=duration)
        result["job"] = job
        if verbose:
            print(f"  ✅ Job submitted: {job}")

    elif backend == "modal":
        if verbose:
            print("  Submitting to Modal GPU (Wan-2.1)...")
        from modal_video import generate_video_modal
        out_path = str(OUTPUTS / f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_raw.mp4")
        local_path = generate_video_modal(
            prompt=final_prompt,
            image_path=source_image,
            output_path=out_path,
            num_frames=81,
            num_inference_steps=20,
        )
        result["local_video"] = local_path
        result["output_url"] = f"file://{local_path}"
        if verbose:
            print(f"  Video saved: {local_path}")

    elif backend == "runway":
        if verbose:
            print("  Submitting to Runway Gen-4.5...")
        job = generate_with_runway(final_prompt, duration=duration, image_path=source_image)
        result["job"] = job
        result["output_url"] = job.get("output_url", "")
        if result["output_url"] and verbose:
            print(f"  Video ready: {result['output_url']}")

    # ── Save result log ─────────────────────────────────────────────────────────
    log_path = OUTPUTS / f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"\n  📁 Result saved: {log_path}")

    return result


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PersonaVid — Video Generator")
    parser.add_argument("--user",     default="default", help="User/brand ID")
    parser.add_argument("--brief",    required=True,     help="What to generate, e.g. 'kitchen gadget ad'")
    parser.add_argument("--backend",  default="prompt",
                        choices=["prompt", "kling", "runway"],
                        help="Generation backend (default: prompt-only mode)")
    parser.add_argument("--duration", type=int, default=10, help="Video duration in seconds")
    parser.add_argument("--ad-type",  default="product",
                        choices=["product", "lifestyle", "tutorial", "testimonial"],
                        help="Type of ad content")
    args = parser.parse_args()

    generate_video(
        user_id  = args.user,
        brief    = args.brief,
        backend  = args.backend,
        duration = args.duration,
        ad_type  = args.ad_type,
    )
