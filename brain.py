"""
brain.py
========
PersonaVid — Brain loader.

Loads personavid_brain.md + optionally injects live trend signals
from the store's trend_results.json.

Usage in any PersonaVid Claude call:
    from brain import video_brain
    client.messages.create(
        model="claude-haiku-4-5-20251001",
        system=video_brain(),
        messages=[{"role": "user", "content": per_call_prompt}],
    )
"""

import json
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).parent
BRAIN_PATH = ROOT / "intelligence" / "personavid_brain.md"
TREND_LOG = Path.home() / "shopify_store" / "logs" / "trend_results.json"
LEARNED_PATH = ROOT / "intelligence" / "personavid_brain.learned.md"


def _load_trends(max_keywords: int = 8) -> str:
    """Pull top trending product keywords from the store's trend log."""
    if not TREND_LOG.exists():
        return ""
    try:
        data = json.loads(TREND_LOG.read_text())
        seen = set()
        keywords = []
        for source in ["google_trends", "youtube_trends", "instagram_trends", "amazon_trends"]:
            for item in data.get(source, [])[:10]:
                kw = item.get("keyword", "").strip()
                score = item.get("score", 0)
                if kw and kw not in seen and score >= 20:
                    seen.add(kw)
                    keywords.append(kw)
                if len(keywords) >= max_keywords:
                    break
            if len(keywords) >= max_keywords:
                break
        if not keywords:
            return ""
        ts = data.get("timestamp", "unknown")[:10]
        return (
            f"\n\n## LIVE TREND SIGNALS (as of {ts})\n"
            f"Currently trending products relevant to our store:\n"
            + "\n".join(f"- {kw}" for kw in keywords)
            + "\nUse these as props or scene context when they fit the brief naturally."
        )
    except Exception:
        return ""


@lru_cache(maxsize=1)
def _core_brain() -> str:
    return BRAIN_PATH.read_text() if BRAIN_PATH.exists() else ""


def _learned_brain() -> str:
    return LEARNED_PATH.read_text() if LEARNED_PATH.exists() else ""


def video_brain(include_trends: bool = True) -> str:
    """
    Return the full system prompt for PersonaVid Claude calls.
    Concatenates: core brain + learned lessons + live trend signals.
    """
    parts = [_core_brain()]
    learned = _learned_brain()
    if learned:
        parts.append(f"\n\n## LEARNED LESSONS\n{learned}")
    if include_trends:
        trends = _load_trends()
        if trends:
            parts.append(trends)
    return "".join(parts)


def teach(lesson: str) -> None:
    """
    Append a user-fed lesson to the learned brain layer.
    Takes effect on the next video_brain() call (cache is not invalidated — restart or call directly).
    """
    LEARNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEARNED_PATH, "a") as f:
        f.write(f"- {lesson}\n")
    print(f"  [Brain] Lesson added: {lesson}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "teach":
        teach(" ".join(sys.argv[2:]))
    else:
        brain = video_brain()
        print(f"Brain loaded: {len(brain)} chars")
        print(brain[:300] + "...")
