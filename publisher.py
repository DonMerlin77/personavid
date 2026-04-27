"""
publisher.py
============
PersonaVid — Platform publishers.

Called by approval.py when a job is approved. Three publishers:
  - publish_tiktok    → Saves to outputs/ready_for_metricool/ (TikTok API blocked for internal use)
  - publish_instagram → Reel via Instagram Graph API (rupload.facebook.com)
  - publish_pinterest → Pin via Pinterest API v5

All credentials read from .env.
Required .env keys:
  INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_USER_ID
  PINTEREST_ACCESS_TOKEN, PINTEREST_BOARD_ID

TikTok note: Direct Post API rejected — "personal or company internal use."
Workaround: Metricool (continuum.goods connected). Pipeline exports video + caption
to outputs/ready_for_metricool/ — upload from there to Metricool for scheduling.
"""

import os
import shutil
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
INSTAGRAM_USER_ID      = os.getenv("INSTAGRAM_USER_ID", "")

PINTEREST_ACCESS_TOKEN = os.getenv("PINTEREST_ACCESS_TOKEN", "")
PINTEREST_BOARD_ID     = os.getenv("PINTEREST_BOARD_ID", "")

METRICOOL_DIR = Path(__file__).parent / "outputs" / "ready_for_metricool"
METRICOOL_DIR.mkdir(parents=True, exist_ok=True)


# ── TikTok (Metricool export) ───────────────────────────────────────────────────

def publish_tiktok(video_path: str, caption: str) -> str:
    """
    TikTok Direct Post API is blocked for internal/single-account use.
    Exports video + caption to outputs/ready_for_metricool/ for manual
    upload via Metricool (continuum.goods account connected).
    """
    src = Path(video_path)
    if not src.exists():
        return f"Failed (TikTok export): video not found at {video_path}"

    stem      = src.stem
    dest_mp4  = METRICOOL_DIR / src.name
    dest_txt  = METRICOOL_DIR / f"{stem}_caption.txt"

    shutil.copy2(src, dest_mp4)
    dest_txt.write_text(caption, encoding="utf-8")

    return (
        f"Ready for Metricool: {dest_mp4.name}\n"
        f"Caption saved: {dest_txt.name}\n"
        f"Upload at: app.metricool.com → TikTok → New Post"
    )


# ── Instagram ──────────────────────────────────────────────────────────────────

def publish_instagram(video_path: str, caption: str) -> str:
    if not INSTAGRAM_ACCESS_TOKEN or not INSTAGRAM_USER_ID:
        return "Failed (Instagram): INSTAGRAM_ACCESS_TOKEN or INSTAGRAM_USER_ID not set in .env"

    file_size = Path(video_path).stat().st_size

    # Step 1 — Create upload session via rupload
    session_resp = requests.post(
        f"https://rupload.facebook.com/video-upload/v18.0/{INSTAGRAM_USER_ID}/videos",
        headers={
            "Authorization":    f"OAuth {INSTAGRAM_ACCESS_TOKEN}",
            "offset":           "0",
            "file_size":        str(file_size),
        },
        timeout=30,
    )
    if session_resp.status_code != 200:
        # Fallback: use Graph API container + publish flow (for hosted videos)
        return _publish_instagram_container(video_path, caption)

    video_id = session_resp.json().get("video_id")
    if not video_id:
        return _publish_instagram_container(video_path, caption)

    # Step 2 — Upload bytes
    with open(video_path, "rb") as f:
        video_data = f.read()

    upload_resp = requests.post(
        f"https://rupload.facebook.com/video-upload/v18.0/{INSTAGRAM_USER_ID}/videos/{video_id}",
        headers={
            "Authorization": f"OAuth {INSTAGRAM_ACCESS_TOKEN}",
            "offset":        "0",
            "file_size":     str(file_size),
            "Content-Type":  "video/mp4",
        },
        data=video_data,
        timeout=180,
    )
    if upload_resp.status_code != 200:
        return f"Failed (Instagram upload {upload_resp.status_code}): {upload_resp.text[:200]}"

    # Step 3 — Create container and publish
    container_resp = requests.post(
        f"https://graph.facebook.com/v18.0/{INSTAGRAM_USER_ID}/media",
        params={
            "media_type":     "REELS",
            "video_id":       video_id,
            "caption":        caption,
            "share_to_feed":  "true",
            "access_token":   INSTAGRAM_ACCESS_TOKEN,
        },
        timeout=30,
    )
    if container_resp.status_code != 200:
        return f"Failed (Instagram container {container_resp.status_code}): {container_resp.text[:200]}"

    creation_id = container_resp.json().get("id")
    if not creation_id:
        return f"Failed (Instagram): no creation_id — {container_resp.json()}"

    # Step 4 — Wait for container to be ready then publish
    for _ in range(12):
        time.sleep(5)
        status_resp = requests.get(
            f"https://graph.facebook.com/v18.0/{creation_id}",
            params={"fields": "status_code", "access_token": INSTAGRAM_ACCESS_TOKEN},
            timeout=15,
        )
        if status_resp.status_code == 200:
            status = status_resp.json().get("status_code", "")
            if status == "FINISHED":
                break
            if status == "ERROR":
                return f"Failed (Instagram): container processing error"

    publish_resp = requests.post(
        f"https://graph.facebook.com/v18.0/{INSTAGRAM_USER_ID}/media_publish",
        params={"creation_id": creation_id, "access_token": INSTAGRAM_ACCESS_TOKEN},
        timeout=30,
    )
    if publish_resp.status_code != 200:
        return f"Failed (Instagram publish {publish_resp.status_code}): {publish_resp.text[:200]}"

    media_id = publish_resp.json().get("id", "unknown")
    return f"Published (Instagram): {media_id}"


def _publish_instagram_container(video_path: str, caption: str) -> str:
    """Fallback: Graph API container flow using a publicly accessible video URL."""
    return (
        "Failed (Instagram): video must be hosted at a public URL for container flow. "
        "Use rupload or host the video file publicly first."
    )


# ── Pinterest ──────────────────────────────────────────────────────────────────

def publish_pinterest(video_path: str, caption: str) -> str:
    if not PINTEREST_ACCESS_TOKEN or not PINTEREST_BOARD_ID:
        return "Failed (Pinterest): PINTEREST_ACCESS_TOKEN or PINTEREST_BOARD_ID not set in .env"

    # Step 1 — Register upload
    reg_resp = requests.post(
        "https://api.pinterest.com/v5/media",
        headers={
            "Authorization": f"Bearer {PINTEREST_ACCESS_TOKEN}",
            "Content-Type":  "application/json",
        },
        json={"media_type": "video"},
        timeout=30,
    )
    if reg_resp.status_code not in (200, 201):
        return f"Failed (Pinterest register {reg_resp.status_code}): {reg_resp.text[:200]}"

    reg_data   = reg_resp.json()
    media_id   = reg_data.get("media_id")
    upload_url = reg_data.get("upload_url")
    upload_params = reg_data.get("upload_parameters", {})

    if not media_id or not upload_url:
        return f"Failed (Pinterest): no media_id/upload_url — {reg_data}"

    # Step 2 — Upload to S3 pre-signed URL
    with open(video_path, "rb") as f:
        files = {k: (None, v) for k, v in upload_params.items()}
        files["file"] = ("video.mp4", f, "video/mp4")
        upload_resp = requests.post(upload_url, files=files, timeout=180)

    if upload_resp.status_code not in (200, 201, 204):
        return f"Failed (Pinterest upload {upload_resp.status_code}): {upload_resp.text[:200]}"

    # Step 3 — Create Pin
    pin_resp = requests.post(
        "https://api.pinterest.com/v5/pins",
        headers={
            "Authorization": f"Bearer {PINTEREST_ACCESS_TOKEN}",
            "Content-Type":  "application/json",
        },
        json={
            "board_id":   PINTEREST_BOARD_ID,
            "title":      caption[:100],
            "description": caption,
            "media_source": {
                "source_type": "video_id",
                "cover_image_url": "",
                "media_id": media_id,
            },
        },
        timeout=30,
    )
    if pin_resp.status_code not in (200, 201):
        return f"Failed (Pinterest pin {pin_resp.status_code}): {pin_resp.text[:200]}"

    pin_id = pin_resp.json().get("id", "unknown")
    return f"Published (Pinterest): pin {pin_id}"
