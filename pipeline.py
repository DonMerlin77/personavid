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
    """Generate a Carter-voice caption using Claude Haiku + personavid brain."""
    import anthropic
    from brain import video_brain
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    platform_rules = {
        "tiktok":    "TikTok caption: hook in first line, 1 CTA, 3-5 hashtags, under 150 chars total",
        "instagram": "Instagram caption: Carter voice, identity hook (cat mom/gift lover), 1 CTA, 5-8 hashtags",
        "pinterest": "Pinterest description: keyword-rich, 2-3 sentences, no hashtags, link in bio CTA",
    }

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=video_brain(include_trends=True),
        messages=[{
            "role": "user",
            "content": (
                f"Write a social media caption for Continuum Goods.\n"
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

        # ── Stage 2: GENERATE (with rejection sampling) ─────────────────────────
        if state["stage"] == "GENERATE":
            print("\n[2/5] Submitting to Modal GPU (Wan-2.1)...")
            from generator import auto_select_source_image
            from modal_video import generate_video_modal
            from scorer import score_video, mutate_prompt, SCORE_THRESHOLD
            from prompt_bank import add_prompt

            source_image = state.get("source_image")
            if not source_image:
                print("  Auto-selecting best source image...")
                source_image = auto_select_source_image(
                    state["final_prompt"], state["user_id"], verbose=True
                )
                state["source_image"] = source_image

            active_prompt = state["final_prompt"]
            best_video = None
            best_score = None
            max_attempts = state.get("max_generate_attempts", 3)

            for attempt in range(1, max_attempts + 1):
                seed = 42 + (attempt - 1) * 13  # deterministic but varied
                print(f"\n  [Generate] Attempt {attempt}/{max_attempts} (seed={seed})...")
                raw_path = str(ROOT / "outputs" / f"{state['job_id']}_raw_attempt{attempt}.mp4")

                generate_video_modal(
                    prompt=active_prompt,
                    image_path=source_image,
                    output_path=raw_path,
                    num_inference_steps=20,
                )

                scores = score_video(raw_path, active_prompt, state["user_id"])

                # Track best candidate (bootstrap — keep highest scorer)
                if best_score is None or scores.get("overall", 0) > best_score.get("overall", 0):
                    best_video = raw_path
                    best_score = scores

                add_prompt(active_prompt, scores, brief=brief, user_id=user_id)

                if scores.get("passed"):
                    print(f"  [Generate] Passed on attempt {attempt} (overall={scores['overall']:.0f})")
                    break

                if attempt < max_attempts:
                    print(f"  [Generate] Score {scores.get('overall', 0):.0f} < {SCORE_THRESHOLD} — mutating prompt...")
                    active_prompt = mutate_prompt(active_prompt, scores)
                else:
                    print(f"  [Generate] Max attempts reached — using best candidate (overall={best_score.get('overall', 0):.0f})")

            state["local_video"] = best_video
            state["generate_scores"] = best_score
            state["final_prompt_used"] = active_prompt
            state["stage"] = "EDIT"
            _save_state(state)

        # ── Stage 3: DOWNLOAD (skipped — Modal returns file directly) ────────────
        if state["stage"] == "DOWNLOAD":
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
            state["stage"] = "SCORE"
            _save_state(state)

        # ── Stage 4b: SCORE edited video ─────────────────────────────────────────
        if state["stage"] == "SCORE":
            from scorer import score_video
            scores = state.get("generate_scores") or score_video(
                state["edited_video"], state.get("final_prompt_used", state["final_prompt"]), user_id
            )
            state["final_scores"] = scores
            passed = scores.get("passed", True)  # if no scores, let it through
            print(f"\n  Final score: overall={scores.get('overall', 'N/A')} | passed={passed}")
            state["stage"] = "APPROVE"
            _save_state(state)

        # ── Stage 5: APPROVE ─────────────────────────────────────────────────────
        if state["stage"] == "APPROVE":
            print("\n[5/5] Queuing for approval...")
            from approval import add_to_queue
            scores = state.get("final_scores", {})
            score_summary = (
                f"Score: {scores.get('overall', '?'):.0f}/100 — "
                f"cat={scores.get('cat_recognized', '?'):.0f} "
                f"location={scores.get('location_appropriate', '?'):.0f} "
                f"brand={scores.get('brand_alignment', '?'):.0f}"
            ) if scores else ""

            add_to_queue(
                job_id=job_id,
                video_path=state["edited_video"],
                platform=platform,
                caption=state["caption"] + (f"\n\n[{score_summary}]" if score_summary else ""),
                brief=brief,
            )
            state["approval_id"] = job_id
            state["stage"] = "PENDING_APPROVAL"
            _save_state(state)
            print(f"\n   Ready for review — open the Approval tab in the app.")
            print(f"   Job ID: {job_id}")
            if score_summary:
                print(f"   {score_summary}")

    except Exception as e:
        state["error"] = str(e)
        state["stage"] = f"ERROR_{state['stage']}"
        _save_state(state)
        print(f"\n   Pipeline error at {state['stage']}: {e}")

    return state
