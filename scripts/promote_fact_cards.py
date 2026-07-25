#!/usr/bin/env python3
"""
Promote approved fact-card drafts (JSON array) into train/eval JSONL.

Workflow
--------
  1. Extract drafts:
       python scripts/extract_fact_card_drafts.py ...
       → data/fact_cards/drafts/wiki_candidates.json   (JSON array; edit this)

  2. Human review in the JSON file:
       - edit / paraphrase facts
       - set "split": "train" | "eval"
       - set "review_status": "approved"   (or "rejected")

  3. Promote:
       python scripts/promote_fact_cards.py
       → appends approved cards to:
            data/fact_cards/train.jsonl
            data/fact_cards/eval.jsonl

Why JSON for drafts and JSONL for approved?
-------------------------------------------
  - Drafts: one pretty-printed array is easy to open, search, and edit in an IDE.
  - Approved train/eval: JSONL is the runtime format for B/C loaders (streamable,
    one record per line, matches other corpora in this repo).

Examples
--------
  # Dry-run: show what would be written, change nothing
  python scripts/promote_fact_cards.py --dry-run

  # Promote from the default drafts file
  python scripts/promote_fact_cards.py

  # Custom drafts path
  python scripts/promote_fact_cards.py --drafts data/fact_cards/drafts/wiki_candidates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRAFTS = REPO_ROOT / "data" / "fact_cards" / "drafts" / "wiki_candidates_batch4.json"
TRAIN_JSONL = REPO_ROOT / "data" / "fact_cards" / "train.jsonl"
EVAL_JSONL = REPO_ROOT / "data" / "fact_cards" / "eval.jsonl"

REQUIRED_FIELDS = ("id", "split", "topic", "facts", "review_status")


def load_jsonl_ids(path: Path) -> Set[str]:
    """Return the set of card ids already present in a JSONL file."""
    ids: Set[str] = set()
    if not path.is_file():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"[error] bad JSONL in {path}:{line_no}: {e}") from e
            cid = obj.get("id")
            if cid:
                ids.add(cid)
    return ids


def load_drafts(path: Path) -> List[Dict[str, Any]]:
    """
    Load draft cards from a pretty-printed JSON array.

    Accepts only a top-level list (the format written by extract_fact_card_drafts.py).
    """
    if not path.is_file():
        raise SystemExit(f"[error] drafts file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(
            f"[error] drafts must be a JSON array of card objects, got {type(data).__name__}"
        )
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise SystemExit(f"[error] drafts[{i}] is not an object")
    return data


def validate_card(card: Dict[str, Any], *, index: int) -> List[str]:
    """
    Lightweight checks before writing to the approved JSONL files.

    Returns a list of error strings (empty ⇒ ok).
    """
    errors: List[str] = []
    label = card.get("id") or f"drafts[{index}]"

    for field in REQUIRED_FIELDS:
        if field not in card:
            errors.append(f"{label}: missing required field '{field}'")

    if card.get("review_status") != "approved":
        # Caller should only pass approved cards; belt-and-suspenders.
        errors.append(f"{label}: review_status is not 'approved'")

    split = card.get("split")
    if split not in ("train", "eval"):
        errors.append(f"{label}: split must be 'train' or 'eval' (got {split!r})")

    facts = card.get("facts")
    if not isinstance(facts, list) or not all(isinstance(x, str) and x.strip() for x in facts):
        errors.append(f"{label}: facts must be a non-empty list of strings")
    elif not (4 <= len(facts) <= 7):
        errors.append(f"{label}: facts must have 4–7 items (got {len(facts)})")

    if not card.get("id"):
        errors.append(f"{label}: empty id")
    if not (card.get("topic") or "").strip():
        errors.append(f"{label}: empty topic")

    return errors


def card_to_jsonl_record(card: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip review-queue-only noise if we ever add any; for now pass through.

    Always write a compact single-line JSON object (JSONL row).
    """
    # Ensure schema_id is present for downstream loaders that care.
    out = dict(card)
    out.setdefault("schema_id", "factcard_v1")
    return out


def append_jsonl(path: Path, cards: Iterable[Dict[str, Any]]) -> int:
    """Append cards as JSONL lines. Creates parent dirs / file as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card_to_jsonl_record(card), ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--drafts",
        type=Path,
        default=DEFAULT_DRAFTS,
        help="Path to drafts JSON array (default: data/fact_cards/drafts/wiki_candidates.json)",
    )
    p.add_argument(
        "--train-out",
        type=Path,
        default=TRAIN_JSONL,
        help="Approved train JSONL path",
    )
    p.add_argument(
        "--eval-out",
        type=Path,
        default=EVAL_JSONL,
        help="Approved eval JSONL path",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print plan only; do not write JSONL",
    )
    p.add_argument(
        "--allow-update",
        action="store_true",
        help=(
            "If an approved draft id already exists in train/eval JSONL, "
            "rewrite that file replacing the old row (default: skip duplicates)"
        ),
    )
    return p.parse_args(argv)


def rewrite_jsonl_with_updates(
    path: Path,
    updates: Dict[str, Dict[str, Any]],
) -> None:
    """
    Rewrite a JSONL file, replacing rows whose id is in `updates`.

    Rows not in updates are kept as-is. New ids should be appended separately.
    """
    if not path.is_file() and not updates:
        return
    kept: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = obj.get("id")
                if cid in updates:
                    kept.append(card_to_jsonl_record(updates[cid]))
                    seen.add(cid)
                else:
                    kept.append(obj)
    # Append updates that were not previously in the file
    for cid, card in updates.items():
        if cid not in seen:
            kept.append(card_to_jsonl_record(card))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    drafts = load_drafts(args.drafts)

    approved = [(i, c) for i, c in enumerate(drafts) if c.get("review_status") == "approved"]
    rejected = sum(1 for c in drafts if c.get("review_status") == "rejected")
    still_draft = sum(1 for c in drafts if c.get("review_status") == "draft")

    print(f"Drafts file   : {args.drafts}")
    print(f"Total cards   : {len(drafts)}")
    print(f"  approved    : {len(approved)}")
    print(f"  draft       : {still_draft}")
    print(f"  rejected    : {rejected}")
    print(f"  other       : {len(drafts) - len(approved) - still_draft - rejected}")
    print("-" * 60)

    if not approved:
        print("Nothing to promote (no review_status=approved).")
        print("Edit the drafts JSON, then re-run.")
        return 0

    # Validate all approved cards first.
    all_errors: List[str] = []
    for i, card in approved:
        all_errors.extend(validate_card(card, index=i))
    if all_errors:
        print("[error] validation failed:")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    existing_train = load_jsonl_ids(args.train_out)
    existing_eval = load_jsonl_ids(args.eval_out)
    existing_all = existing_train | existing_eval

    to_train: List[Dict[str, Any]] = []
    to_eval: List[Dict[str, Any]] = []
    skip_dup: List[str] = []
    update_train: Dict[str, Dict[str, Any]] = {}
    update_eval: Dict[str, Dict[str, Any]] = {}

    for _, card in approved:
        cid = card["id"]
        split = card["split"]
        if cid in existing_all:
            if not args.allow_update:
                skip_dup.append(cid)
                continue
            if split == "train":
                update_train[cid] = card
            else:
                update_eval[cid] = card
            continue
        if split == "train":
            to_train.append(card)
        else:
            to_eval.append(card)

    if skip_dup:
        print(f"Skipping {len(skip_dup)} id(s) already in JSONL (use --allow-update to replace):")
        for cid in skip_dup[:20]:
            print(f"  - {cid}")
        if len(skip_dup) > 20:
            print(f"  ... +{len(skip_dup) - 20} more")

    print(f"Will append   : train={len(to_train)}  eval={len(to_eval)}")
    if args.allow_update:
        print(f"Will update   : train={len(update_train)}  eval={len(update_eval)}")

    if args.dry_run:
        print("[dry-run] no files written")
        for card in to_train + to_eval + list(update_train.values()) + list(update_eval.values()):
            print(f"  {card['split']:5}  {card['id']}  ·  {card['topic']}")
        return 0

    n_train = append_jsonl(args.train_out, to_train) if to_train else 0
    n_eval = append_jsonl(args.eval_out, to_eval) if to_eval else 0

    if update_train:
        rewrite_jsonl_with_updates(args.train_out, update_train)
    if update_eval:
        rewrite_jsonl_with_updates(args.eval_out, update_eval)

    print("-" * 60)
    print(f"Appended train : {n_train}  → {args.train_out.relative_to(REPO_ROOT)}")
    print(f"Appended eval  : {n_eval}  → {args.eval_out.relative_to(REPO_ROOT)}")
    if update_train or update_eval:
        print(f"Updated train  : {len(update_train)}")
        print(f"Updated eval   : {len(update_eval)}")
    print("Done. B/C should load the JSONL files (approved only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
