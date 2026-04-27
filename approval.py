"""
approval.py
===========
PersonaVid — 15-minute approval window.

When a video is ready, this module:
1. Writes a pending job to approval_queue.json
2. Starts a 15-minute countdown
3. Exposes three actions via CLI or Gradio:
     - post_now   → publish immediately
     - reject     → discard the video
     - let_post   → approve, auto-publishes when window closes
4. If no action taken within 15 minutes, auto-publishes approved jobs

Usage (CLI):
    python3 approval.py list                      # show pending jobs
    python3 approval.py post_now <job_id>         # publish immediately
    python3 approval.py reject <job_id>           # discard
    python3 approval.py let_post <job_id>         # approve for auto-post
    python3 approval.py process                   # check + publish expired approvals
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT  = Path(__file__).parent
QUEUE = ROOT / "approval_queue.json"

WINDOW_MINUTES = 15


def _load_queue() -> list:
    if not QUEUE.exists():
        return []
    return json.loads(QUEUE.read_text())


def _save_queue(jobs: list) -> None:
    QUEUE.write_text(json.dumps(jobs, indent=2))


def add_to_queue(
    job_id: str,
    video_path: str,
    platform: str,
    caption: str,
    brief: str,
    preview_url: str = "",
) -> dict:
    jobs = _load_queue()
    expires_at = (datetime.now() + timedelta(minutes=WINDOW_MINUTES)).isoformat()
    job = {
        "job_id":      job_id,
        "video_path":  video_path,
        "platform":    platform,
        "caption":     caption,
        "brief":       brief,
        "preview_url": preview_url,
        "status":      "pending",
        "created_at":  datetime.now().isoformat(),
        "expires_at":  expires_at,
        "action":      None,
    }
    jobs.append(job)
    _save_queue(jobs)
    print(f"   Queued for approval: {job_id} (expires {expires_at[:16]})")
    return job


def _find_job(job_id: str) -> tuple[int, dict | None]:
    jobs = _load_queue()
    for i, j in enumerate(jobs):
        if j["job_id"] == job_id:
            return i, j
    return -1, None


def post_now(job_id: str) -> str:
    jobs = _load_queue()
    idx, job = _find_job(job_id)
    if not job:
        return f"Job {job_id} not found."
    if job["status"] != "pending":
        return f"Job {job_id} already {job['status']}."
    jobs[idx]["status"] = "approved"
    jobs[idx]["action"] = "post_now"
    _save_queue(jobs)
    result = _publish(job)
    jobs = _load_queue()
    jobs[idx]["status"] = "published" if "published" in result else "failed"
    _save_queue(jobs)
    return result


def reject(job_id: str) -> str:
    jobs = _load_queue()
    idx, job = _find_job(job_id)
    if not job:
        return f"Job {job_id} not found."
    jobs[idx]["status"] = "rejected"
    jobs[idx]["action"] = "reject"
    _save_queue(jobs)
    try:
        Path(job["video_path"]).unlink(missing_ok=True)
    except Exception:
        pass
    return f"Rejected and deleted: {job_id}"


def let_post(job_id: str) -> str:
    jobs = _load_queue()
    idx, job = _find_job(job_id)
    if not job:
        return f"Job {job_id} not found."
    if job["status"] != "pending":
        return f"Job {job_id} already {job['status']}."
    jobs[idx]["status"] = "approved"
    jobs[idx]["action"] = "let_post"
    _save_queue(jobs)
    return f"Approved — will auto-post when window closes at {job['expires_at'][:16]}"


def process_queue() -> list[str]:
    """Check queue and publish any approved + expired jobs. Call on a schedule."""
    jobs = _load_queue()
    results = []
    now = datetime.now()

    for i, job in enumerate(jobs):
        if job["status"] == "approved" and job["action"] == "let_post":
            expires = datetime.fromisoformat(job["expires_at"])
            if now >= expires:
                result = _publish(job)
                jobs[i]["status"] = "published" if "published" in result else "failed"
                results.append(result)

        elif job["status"] == "pending":
            expires = datetime.fromisoformat(job["expires_at"])
            if now >= expires:
                jobs[i]["status"] = "expired"
                results.append(f"Expired (no action taken): {job['job_id']}")

    _save_queue(jobs)
    return results


def _publish(job: dict) -> str:
    """Route to the correct publisher based on platform."""
    platform = job["platform"]
    video_path = job["video_path"]
    caption = job["caption"]

    if not Path(video_path).exists():
        return f"Failed: video file missing — {video_path}"

    try:
        if platform == "tiktok":
            from publisher import publish_tiktok
            return publish_tiktok(video_path, caption)
        elif platform == "instagram":
            from publisher import publish_instagram
            return publish_instagram(video_path, caption)
        elif platform == "pinterest":
            from publisher import publish_pinterest
            return publish_pinterest(video_path, caption)
        else:
            return f"Unknown platform: {platform}"
    except Exception as e:
        return f"Publish failed ({platform}): {e}"


def list_queue() -> list[dict]:
    jobs = _load_queue()
    now = datetime.now()
    result = []
    for j in jobs:
        expires = datetime.fromisoformat(j["expires_at"])
        remaining = max(0, int((expires - now).total_seconds() // 60))
        result.append({
            **j,
            "minutes_remaining": remaining if j["status"] == "pending" else 0,
        })
    return result


def format_queue_display() -> str:
    jobs = list_queue()
    if not jobs:
        return "No jobs in approval queue."
    lines = []
    for j in jobs:
        status_line = f"[{j['status'].upper()}]"
        if j["status"] == "pending":
            status_line += f" {j['minutes_remaining']}m remaining"
        lines.append(
            f"{j['job_id']} | {j['platform']} | {status_line}\n"
            f"  Brief: {j['brief'][:60]}\n"
            f"  Caption: {j['caption'][:60]}"
        )
    return "\n\n".join(lines)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args or args[0] == "list":
        print(format_queue_display())
    elif args[0] == "post_now" and len(args) > 1:
        print(post_now(args[1]))
    elif args[0] == "reject" and len(args) > 1:
        print(reject(args[1]))
    elif args[0] == "let_post" and len(args) > 1:
        print(let_post(args[1]))
    elif args[0] == "process":
        results = process_queue()
        for r in results:
            print(r)
        if not results:
            print("Nothing to process.")
    else:
        print(__doc__)
