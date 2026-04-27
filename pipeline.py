"""
pipeline.py
===========
PersonaVid — End-to-end video pipeline orchestrator.

Takes a creative brief + source image/video → generates AI video via Runway
→ auto-edits (crop, grade, text overlays) → queues for approval → publishes.

Stages (each persisted to job_state.json before next stage runs):
  1. PROMPT     — build personalized prompt from style fingerprint
  2. GENERATE   — submit to Runway Gen-4.5, poll for result
  3. DOWNLOAD   — save Runway output to disk
  4. EDIT       — ffmpeg crop/grade/overlay → platform-ready file
  5. APPROVE    — 15-min approval window (post_now / reject / let_post)
  6. PUBLISH    — send to TikTok / Instagram / Pinterest

Usage:
    from pipeline import run_pipeline
    job = run_pipeline(
        user_id="continuum_goods",
        brief="Cash the cat sitting near the glowing moon lamp",
        platform="tiktok",
        hook_text="he claimed it as his",
        cta_text="Moon Lamp — link in bio",
        source_image="/path/to/frame.jpg",   # optional
    )
"""

import json
import os
import time
import uuid
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ROOT   = Path(__file__).parent
STATES = ROOT / "outputs" / "states"
STATES.mkdir(parents=True, exist_ok=True)


def _state_path(job_id: str) -> Path:
    return STATES / f"{job_id}.json"


def _save_state(state: dict) -> None:
    _state_path(state["job_id"]).write_text(json.dumps(state, indent=2))


def _load_state(job_id: str) -> dict | None:
    p = _state_path(job_id)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _generate_caption(brief: str, platform: str, user_id: str) -> str:
    """Generate a Carter-voice caption using Claude Haiku."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    platform_rules = {
        "tiktok":    "TikTok caption: hook in first line, 1 CTA, 3-5 hashtags, under 150 chars total",
        "instagram": "Instagram caption: Carter voice, identity hook (cat mom/gift lover), 1 CTA, 5-8 hashtags",
        "pinterest": "Pinterest description: keyword-rich, 2-3 sentences, no hashtags, link in bio CTA",
    }

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"Write a social media caption for Continuum Goods.\n"
                f"Brand mascot: Carter the opinionated orange tabby cat.\n"
                f"Platform rules: {platform_rules.get(platform, 'short, punchy, one CTA')}\n"
                f"Creative brief: {brief}\n"
                f"Output ONLY the caption text, nothing else."
            ),
        }],
    )
    return response.content[0].text.strip()


def run_pipeline(
    user_id: str,
    brief: str,
    platform: str = "tiktok",
    hook_text: str = "",
    cta_text: str = "",
    source_image: str | None = None,
    resume_job_id: str | None = None,
) -> dict:
    """
    Run the full pipeline. Returns job state dict.
    If resume_job_id given, picks up from last completed stage.
    """
    job_id = resume_job_id or f"{platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    state = _load_state(job_id) or {
        "job_id":       job_id,
        "user_id":      user_id,
        "brief":        brief,
        "platform":     platform,
        "hook_text":    hook_text,
        "cta_text":     cta_text,
        "source_image": source_image,
        "stage":        "PROMPT",
        "created_at":   datetime.now().isoformat(),
        "runway_url":   None,
        "local_video":  None,
        "edited_video": None,
        "caption":      None,
        "approval_id":  None,
        "error":        None,
    }

    print(f"\n PersonaVid Pipeline — {job_id}")
    print(f"   Platform: {platform} | User: {user_id}")

    try:
        # ── Stage 1: PROMPT ──────────────────────────────────────────────────────
        if state["stage"] == "PROMPT":
            print("\n[1/5] Building prompt...")
            from generator import generate_video
            result = generate_video(
                user_id=user_id,
                brief=brief,
                backend="prompt",
                duration=10,
                ad_type="lifestyle",
                verbose=False,
            )
            state["final_prompt"] = result["final_prompt"]
            state["stage"] = "GENERATE"
            _save_state(state)

        # ── Stage 2: GENERATE ────────────────────────────────────────────────────
        if state["stage"] == "GENERATE":
            print("\n[2/5] Submitting to Runway Gen-4.5...")
            from generator import generate_with_runway, auto_select_source_image
            source_image = state.get("source_image")
            if not source_image:
                print("  Auto-selecting best source image...")
                source_image = auto_select_source_image(
                    state["final_prompt"], state["user_id"], verbose=True
                )
                state["source_image"] = source_image
            job = generate_with_runway(
                state["final_prompt"],
                duration=10,
                image_path=source_image,
            )
            state["runway_url"] = job["output_url"]
            state["stage"] = "DOWNLOAD"
            _save_state(state)

        # ── Stage 3: DOWNLOAD ────────────────────────────────────────────────────
        if state["stage"] == "DOWNLOAD":
            print("\n[3/5] Downloading video...")
            out_dir = ROOT / "outputs"
            local_path = str(out_dir / f"{job_id}_raw.mp4")
            r = requests.get(state["runway_url"], timeout=120, stream=True)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            state["local_video"] = local_path
            state["stage"] = "EDIT"
            _save_state(state)

        # ── Stage 4: EDIT ────────────────────────────────────────────────────────
        if state["stage"] == "EDIT":
            print("\n[4/5] Editing video...")
            from video_editor import edit_video
            edited = edit_video(
                input_path=state["local_video"],
                hook_text=state["hook_text"],
                cta_text=state["cta_text"],
                platform=state["platform"],
            )
            caption = _generate_caption(brief, platform, user_id)
            state["edited_video"] = edited
            state["caption"] = caption
            state["stage"] = "APPROVE"
            _save_state(state)

        # ── Stage 5: APPROVE ─────────────────────────────────────────────────────
        if state["stage"] == "APPROVE":
            print("\n[5/5] Queuing for approval...")
            from approval import add_to_queue
            job = add_to_queue(
                job_id=job_id,
                video_path=state["edited_video"],
                platform=platform,
                caption=state["caption"],
                brief=brief,
            )
            state["approval_id"] = job_id
            state["stage"] = "PENDING_APPROVAL"
            _save_state(state)
            print(f"\n   Ready for review — open the Approval tab in the app.")
            print(f"   Job ID: {job_id}")

    except Exception as e:
        state["error"] = str(e)
        state["stage"] = f"ERROR_{state['stage']}"
        _save_state(state)
        print(f"\n   Pipeline error at {state['stage']}: {e}")

    return state
