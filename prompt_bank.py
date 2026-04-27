"""
prompt_bank.py
==============
PersonaVid — Weighted prompt store with epsilon-greedy sampling.

Proven prompts are stored with their average scores. Sampling uses
exploitation (pick by score weight) vs exploration (random pick) to
balance using what works vs discovering new things that work.

Bootstrap: seed the bank with top prompts from past sessions.
The bank grows over time as the pipeline runs and scores accumulate.
"""

import json
import random
from datetime import datetime
from pathlib import Path

BANK_PATH = Path(__file__).parent / "data" / "prompt_bank.json"
MIN_SCORE_TO_STORE = 55    # don't bother storing low-quality prompts
MIN_SCORE_TO_SAMPLE = 60   # only sample from prompts that actually worked
EXPLORE_RATIO = 0.2        # 20% random exploration, 80% exploitation


def _load() -> list:
    if BANK_PATH.exists():
        return json.loads(BANK_PATH.read_text())
    return []


def _save(bank: list) -> None:
    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    BANK_PATH.write_text(json.dumps(bank, indent=2))


def add_prompt(prompt: str, scores: dict, brief: str = "", user_id: str = "continuum_goods") -> None:
    """
    Store a scored prompt. Updates running average if prompt already exists.
    Skips storage if overall score < MIN_SCORE_TO_STORE.
    """
    overall = scores.get("overall", 0)
    if overall < MIN_SCORE_TO_STORE:
        return

    bank = _load()
    existing = next((p for p in bank if p["prompt"] == prompt and p["user_id"] == user_id), None)

    if existing:
        n = existing.get("n_uses", 1)
        existing["avg_score"] = round((existing["avg_score"] * n + overall) / (n + 1), 1)
        existing["last_score"] = overall
        existing["n_uses"] = n + 1
        existing["last_used"] = datetime.now().isoformat()
        existing["notes"] = scores.get("notes", existing.get("notes", ""))
    else:
        bank.append({
            "prompt": prompt,
            "brief": brief,
            "user_id": user_id,
            "avg_score": round(overall, 1),
            "last_score": round(overall, 1),
            "best_dimensions": {
                "cat_recognized": scores.get("cat_recognized"),
                "location_appropriate": scores.get("location_appropriate"),
                "lighting_quality": scores.get("lighting_quality"),
                "motion_quality": scores.get("motion_quality"),
                "brand_alignment": scores.get("brand_alignment"),
            },
            "n_uses": 1,
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "notes": scores.get("notes", ""),
        })

    _save(bank)
    print(f"  [PromptBank] Stored prompt (score={overall:.0f}, n={existing['n_uses'] if existing else 1})")


def sample_prompt(user_id: str = "continuum_goods") -> dict | None:
    """
    Epsilon-greedy sampling:
      - 80% of the time: pick weighted by avg_score (exploitation)
      - 20% of the time: pick uniformly at random (exploration)

    Returns a prompt dict or None if bank is empty.
    """
    bank = [p for p in _load() if p["user_id"] == user_id and p["avg_score"] >= MIN_SCORE_TO_SAMPLE]
    if not bank:
        return None

    if random.random() < EXPLORE_RATIO:
        chosen = random.choice(bank)
        print(f"  [PromptBank] Exploring — random pick (score={chosen['avg_score']})")
        return chosen

    # Softmax-style weighted sampling by score
    weights = [p["avg_score"] ** 2 for p in bank]  # square to sharpen preference
    total = sum(weights)
    r = random.random() * total
    cumulative = 0.0
    for prompt, weight in zip(bank, weights):
        cumulative += weight
        if r <= cumulative:
            print(f"  [PromptBank] Exploiting — score={prompt['avg_score']} prompt")
            return prompt
    return bank[-1]


def top_prompts(user_id: str = "continuum_goods", n: int = 5) -> list:
    """Return top n prompts by avg_score for a given user."""
    bank = [p for p in _load() if p["user_id"] == user_id]
    return sorted(bank, key=lambda p: p["avg_score"], reverse=True)[:n]


def bank_stats(user_id: str = "continuum_goods") -> dict:
    """Summary stats for the prompt bank."""
    bank = [p for p in _load() if p["user_id"] == user_id]
    if not bank:
        return {"total": 0}
    scores = [p["avg_score"] for p in bank]
    return {
        "total": len(bank),
        "avg_score": round(sum(scores) / len(scores), 1),
        "best_score": max(scores),
        "worst_score": min(scores),
        "total_uses": sum(p.get("n_uses", 1) for p in bank),
    }


def seed_bank(prompts: list[dict]) -> None:
    """
    Bootstrap the bank with known-good prompts from prior sessions.
    Each item: {"prompt": str, "score": float, "brief": str, "user_id": str}
    """
    for p in prompts:
        add_prompt(
            prompt=p["prompt"],
            scores={"overall": p.get("score", 70)},
            brief=p.get("brief", ""),
            user_id=p.get("user_id", "continuum_goods"),
        )
    print(f"  [PromptBank] Seeded {len(prompts)} prompts")


if __name__ == "__main__":
    stats = bank_stats()
    print(f"Bank stats: {stats}")
    tops = top_prompts()
    for i, p in enumerate(tops, 1):
        print(f"  {i}. [{p['avg_score']}] {p['prompt'][:80]}...")
