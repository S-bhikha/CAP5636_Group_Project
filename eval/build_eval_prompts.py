"""Build the frozen eval prompt pack from Lane A's held-out fact cards (Lane C).

The eval pack must be generated from the fact cards with the SAME renderer M2
was supervised with (`scripts/lab_gpt/prompts.render_model_input`), per the
data/SCHEMA.md contract. Hand-written bare sentences put M2 off-distribution:
it is trained on "Topic / Facts / Instruction" and would be tested on a single
sentence.

Two conditions are built from the SAME cards under the SAME prompt ids, so the
two runs are paired per topic:

  card    topic + numbered facts + frozen instruction. This is the format M2
          was trained on -- the PRIMARY eval set for all systems.
  nocard  the card's first fact only, no topic/facts/instruction scaffold.
          Reproduces the pre-fix pack, and is the README's prompt ablation
          ("B0 / M2 with vs without fact card in context").

Usage:
    # primary set (what B0/B1/M2 are scored on)
    python eval/build_eval_prompts.py --condition card \
        --out eval/prompts/eval_prompts.jsonl

    # ablation set (no fact card in context)
    python eval/build_eval_prompts.py --condition nocard \
        --out eval/prompts/eval_prompts_nocard.jsonl

The frozen prompt_id -> card_id map lives in eval/prompts/frozen_eval_ids.txt
and must not be reordered once scoring starts. `--freeze-from <bare pack>`
regenerates that map from a legacy pack (provenance only; already done).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.lab_gpt.prompts import load_templates, render_model_input

DEFAULT_CARDS = REPO_ROOT / "data" / "fact_cards" / "eval.jsonl"
DEFAULT_TEMPLATES = REPO_ROOT / "data" / "prompts" / "templates.json"
DEFAULT_IDS = REPO_ROOT / "eval" / "prompts" / "frozen_eval_ids.txt"


def load_cards(path: Path) -> Dict[str, Dict[str, Any]]:
    cards: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            if card.get("review_status") != "approved" or card.get("split") != "eval":
                continue
            cards[card["id"]] = card
    return cards


def load_frozen_ids(path: Path) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            prompt_id, card_id = line.split()
            pairs.append((prompt_id, card_id))
    return pairs


def freeze_from_legacy_pack(legacy: Path, cards: Dict[str, Dict[str, Any]], out: Path) -> None:
    """Recover prompt_id -> card_id from a pack whose prompts were bare first facts."""
    by_first_fact = {c["facts"][0]: cid for cid, c in cards.items() if c.get("facts")}
    lines = [
        "# Frozen eval set: prompt_id -> card_id (data/fact_cards/eval.jsonl).",
        "# Order and pairing are FROZEN -- appending is fine, reordering invalidates scores.",
    ]
    with legacy.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            card_id = by_first_fact.get(item["prompt"])
            if card_id is None:
                raise SystemExit(f"{item['id']}: prompt does not match any approved eval card's first fact")
            lines.append(f"{item['id']} {card_id}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines) - 2} frozen pairs -> {out}")


def build_prompt(card: Dict[str, Any], templates: Dict[str, Any], condition: str) -> str:
    if condition == "card":
        return render_model_input(card, templates)
    return card["facts"][0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", choices=["card", "nocard"], default="card")
    ap.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    ap.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    ap.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--freeze-from", type=Path, help="Rebuild --ids from a legacy bare-sentence pack, then exit")
    args = ap.parse_args()

    cards = load_cards(args.cards)
    if not cards:
        raise SystemExit(f"No approved split=eval cards in {args.cards}")

    if args.freeze_from:
        freeze_from_legacy_pack(args.freeze_from, cards, args.ids)
        return

    if args.out is None:
        raise SystemExit("--out is required")

    templates = load_templates(args.templates)
    pairs = load_frozen_ids(args.ids)

    rows = []
    for prompt_id, card_id in pairs:
        card = cards.get(card_id)
        if card is None:
            raise SystemExit(f"{prompt_id}: frozen card_id {card_id} is not an approved split=eval card")
        rows.append({
            "id": prompt_id,
            "card_id": card_id,
            "condition": args.condition,
            "topic": card["topic"],
            "facts": card["facts"],
            "prompt": build_prompt(card, templates, args.condition),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    chars = sorted(len(r["prompt"]) for r in rows)
    print(f"Wrote {len(rows)} '{args.condition}' prompts -> {args.out}")
    print(f"  prompt chars: median {chars[len(chars) // 2]}, max {chars[-1]} "
          f"(~{chars[-1] / 3.9:.0f} tokens at ~3.9 chars/token)")
    print("  Check generate_samples.py's context guard against the model's block_size.")


if __name__ == "__main__":
    main()
