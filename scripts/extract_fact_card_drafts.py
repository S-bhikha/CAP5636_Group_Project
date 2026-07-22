#!/usr/bin/env python3
"""
Extract draft fact cards from local Simple English Wikipedia JSONL.

Purpose
-------
Speed up Lane A review by turning Wiki articles into *candidate* fact cards.
Humans still edit, paraphrase, assign train/eval, and set review_status=approved.

This is intentionally heuristic (no LLM): repeatable, cheap, and easy to audit.

Pipeline (schema factcard_v1, method lead_sentences_v1)
-------------------------------------------------------
  1. Read article objects from data/raw/wikipedia/*.jsonl
  2. Filter out weak titles / extreme lengths
  3. Take the article "lead" (first paragraph / first ~1200 chars)
  4. Split the lead into sentences
  5. Keep short, declarative sentences that look like checkable claims
  6. Take the first 4–7 survivors (in lead order) as draft `facts`
  7. Write a single JSON *array* to fact_cards/drafts/*.json (easy to edit in an IDE)

Review → approve workflow
-------------------------
  1. Edit the drafts JSON (fix facts, set split, set review_status to "approved")
  2. Promote approved cards into train/eval JSONL:
       python scripts/promote_fact_cards.py

Important
---------
- Drafts are NOT gold labels. Many will be rejected or rewritten.
- We do not score "importance"; order is lead order after filters.
- Reported runs should use only review_status=approved cards in the JSONL
  files (see data/SCHEMA.md).

Examples
--------
  python scripts/extract_fact_card_drafts.py \\
      --input data/raw/wikipedia/20231101_simple_smoke.jsonl \\
      --max-drafts 30

  python scripts/extract_fact_card_drafts.py \\
      --input data/raw/wikipedia/20231101_simple.jsonl \\
      --max-drafts 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# ---------------------------------------------------------------------------
# Paths & version tags
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "wikipedia" / "20231101_simple.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "fact_cards" / "drafts" / "wiki_candidates.json"

# Bump EXTRACT_METHOD when changing how facts are chosen (keeps provenance honest).
EXTRACT_METHOD = "lead_sentences_v1"
SCHEMA_ID = "factcard_v1"

# Titles that rarely yield a clean story-teachable fact card (lists, meta pages, …).
TITLE_DENY_PREFIXES = (
    "list of ",
    "timeline of ",
    "index of ",
    "category:",
    "wikipedia:",
    "template:",
    "file:",
    "portal:",
    "help:",
    "module:",
)

# Lead sentences that are wiki chrome / non-claims, not teaching content.
BAD_STARTS = (
    "this article",
    "this page",
    "references",
    "see also",
    "external links",
    "disambiguation",
    "coordinates",
    "isbn",
)

# Used only to count "words" in a sentence for length filters.
WORD_RE = re.compile(r"[A-Za-z]+")


def _utc_now() -> str:
    """ISO-8601 UTC timestamp for source.extracted_at."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(title: str, max_len: int = 40) -> str:
    """
    Turn a Wiki title into a stable id fragment: "Water cycle" → "water_cycle".

    Used in card ids: fc_<slug>_<nn>. Reviewers can rename later if needed.
    """
    s = title.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "topic"
    return s[:max_len].strip("_")


def title_ok(title: str) -> bool:
    """
    Article-level title gate.

    Drops meta/list pages, pure year pages, and extremely long titles.
    These rarely map cleanly to a short narrative teaching topic.
    """
    if not title or not title.strip():
        return False
    t = title.strip()
    low = t.lower()
    if any(low.startswith(p) for p in TITLE_DENY_PREFIXES):
        return False
    # Bare years ("1999") are weak standalone story topics for our setup.
    if re.fullmatch(r"\d{1,4}", t):
        return False
    if len(t) > 60:
        return False
    return True


def lead_text(text: str, max_chars: int = 1200) -> str:
    """
    Extract a short lead blob to mine for facts.

    Why the lead?
      Simple English articles usually front-load definition + core claims.
      Using the whole page pulls in history sections, lists, and noise.

    Strategy:
      - Prefer text before the first blank line (first paragraph).
      - Collapse whitespace.
      - Cap length so we never fact-mine deep into the article body.
    """
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return ""
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    lead = parts[0].strip()
    lead = re.sub(r"\s+", " ", lead)
    return lead[:max_chars]


def split_sentences(lead: str) -> List[str]:
    """
    Lightweight sentence split on . ? ! followed by whitespace.

    Not a full NLP segmenter — good enough for Simple English leads.
    Normalizes each sentence to end with terminal punctuation so facts
    look consistent as checklist bullets.
    """
    if not lead:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", lead)
    out: List[str] = []
    for c in chunks:
        s = c.strip()
        if not s:
            continue
        if s[-1] not in ".!?":
            s = s + "."
        out.append(s)
    return out


def sentence_ok(s: str, *, min_chars: int, max_chars: int) -> bool:
    """
    Per-sentence filter: keep only strings that can work as gold fact bullets.

    A good draft fact for our project is:
      - short enough to check in a ~120–180 word story
      - long enough to be a real claim (not a fragment)
      - mostly natural language (not table/wiki junk)
      - not meta ("This page…") or pure glossary glosses

    Failures here are the main reason an article produces no card.
    """
    s = s.strip()
    if len(s) < min_chars or len(s) > max_chars:
        return False
    low = s.lower()
    if any(low.startswith(b) for b in BAD_STARTS):
        return False

    words = WORD_RE.findall(s)
    # Word band keeps bullets story-coverable and avoids one-word stubs /
    # multi-clause essays.
    if len(words) < 5 or len(words) > 28:
        return False

    # Glossary-style lines are weak teaching targets for narrative SFT.
    if low.startswith("the word ") or low.startswith("the name "):
        return False

    # High symbol density → coordinates, tables, markup leftovers.
    letters = sum(ch.isalpha() or ch.isspace() for ch in s)
    if letters / max(len(s), 1) < 0.85:
        return False
    return True


def pick_facts(
    sentences: Sequence[str],
    *,
    min_facts: int,
    max_facts: int,
    min_chars: int,
    max_chars: int,
) -> List[str]:
    """
    Select draft facts from ordered lead sentences.

    Selection policy (lead_sentences_v1):
      - Walk sentences in lead order (not shuffled, not ranked by "importance").
      - Keep the first ones that pass sentence_ok.
      - Deduplicate near-exact repeats.
      - Stop at max_facts.
      - If fewer than min_facts survive, return [] (skip article).

    This is deliberately simple so reviewers know what they are looking at:
    "top of the page, cleaned," not a black-box summarizer.
    """
    facts: List[str] = []
    seen: Set[str] = set()
    for s in sentences:
        if not sentence_ok(s, min_chars=min_chars, max_chars=max_chars):
            continue
        key = re.sub(r"\s+", " ", s.lower())
        if key in seen:
            continue
        seen.add(key)
        facts.append(s.strip())
        if len(facts) >= max_facts:
            break
    if len(facts) < min_facts:
        return []
    return facts


def guess_domain(title: str, facts: Sequence[str]) -> str:
    """
    Cheap keyword tag for filtering/sorting in review (not a model feature).

    Default domain set matches data/SCHEMA.md. Wrong tags are fine — humans
    can fix on approve. Returns "other" when nothing matches.
    """
    blob = (" ".join([title, *facts])).lower()
    rules = [
        ("nature", ("animal", "plant", "river", "ocean", "tree", "forest", "bird", "fish", "weather", "rain", "cloud")),
        ("science", ("energy", "atom", "force", "electric", "magnet", "chemical", "planet", "star", "gravity", "science")),
        ("body", ("body", "bone", "blood", "heart", "brain", "muscle", "tooth", "health")),
        ("geography", ("country", "city", "capital", "mountain", "island", "continent", "map", "ocean")),
        ("history", ("king", "queen", "war", "ancient", "century", "empire", "history", "president")),
        ("everyday", ("food", "bread", "school", "family", "house", "clothes", "music", "sport", "game")),
    ]
    for domain, keys in rules:
        if any(k in blob for k in keys):
            return domain
    return "other"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file; skip blank/bad lines with a warning."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] skip bad json line {line_no}: {e}", file=sys.stderr)


def make_card(
    *,
    article: Dict[str, Any],
    facts: List[str],
    hf_config: str,
    seq: int,
) -> Dict[str, Any]:
    """
    Build one draft fact-card object (schema factcard_v1).

    Defaults chosen for the review queue:
      - split=train (reviewer may reassign to eval)
      - review_status=draft (must not train/eval until approved)
      - false_claims=[] (optional; raters fill later if useful)
      - source.* filled for provenance + license attribution trail
    """
    title = (article.get("title") or "").strip()
    topic = title  # Wiki title is already a short topic label
    slug = slugify(title)
    return {
        "id": f"fc_{slug}_{seq:02d}",
        "split": "train",
        "topic": topic,
        "domain": guess_domain(title, facts),
        "facts": facts,
        "false_claims": [],
        "source": {
            "type": "wikipedia_simple",
            "hf_dataset": "wikimedia/wikipedia",
            "hf_config": hf_config,
            "wiki_id": article.get("id"),
            "title": title,
            "url": article.get("url"),
            "extract_method": EXTRACT_METHOD,
            "extracted_at": _utc_now(),
        },
        "review_status": "draft",
        "notes": (
            f"Auto-draft via {EXTRACT_METHOD}. Edit or paraphrase facts for clarity "
            "and story-coverability; set split=eval only when reserving held-out ids; "
            "then set review_status=approved."
        ),
        "word_target": [120, 180],
        "schema_id": SCHEMA_ID,
    }


def extract_from_article(
    article: Dict[str, Any],
    *,
    min_chars_article: int,
    max_chars_article: int,
    min_facts: int,
    max_facts: int,
    min_sent_chars: int,
    max_sent_chars: int,
) -> Optional[List[str]]:
    """
    Full per-article extract: filters → lead → sentences → facts.

    Returns:
      list of fact strings, or None if the article should be skipped.
    """
    title = (article.get("title") or "").strip()
    text = article.get("text") or ""
    if not title_ok(title):
        return None

    # Mid-length articles: enough signal for a few facts, not a wall of text.
    n = len(text)
    if n < min_chars_article or n > max_chars_article:
        return None

    lead = lead_text(text)
    sents = split_sentences(lead)
    facts = pick_facts(
        sents,
        min_facts=min_facts,
        max_facts=max_facts,
        min_chars=min_sent_chars,
        max_chars=max_sent_chars,
    )
    return facts or None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Wiki JSONL path")
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Draft cards path (pretty-printed JSON array for human review)",
    )
    p.add_argument(
        "--hf-config",
        default="20231101.simple",
        help="Recorded in source.hf_config (provenance only)",
    )
    p.add_argument("--max-drafts", type=int, default=50, help="Stop after this many draft cards")
    p.add_argument(
        "--min-article-chars",
        type=int,
        default=500,
        help="Skip articles shorter than this (stubs)",
    )
    p.add_argument(
        "--max-article-chars",
        type=int,
        default=2500,
        help="Skip articles longer than this (harder to mine a tight lead)",
    )
    p.add_argument("--min-facts", type=int, default=4, help="Need at least this many facts or skip")
    p.add_argument("--max-facts", type=int, default=7, help="Cap facts per card (schema: 4–7)")
    p.add_argument("--min-sent-chars", type=int, default=25, help="Min chars per fact sentence")
    p.add_argument("--max-sent-chars", type=int, default=160, help="Max chars per fact sentence")
    p.add_argument(
        "--seed-seq",
        type=int,
        default=1,
        help="Starting nn suffix for fc_<slug>_<nn> ids",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Scan wiki JSONL, write draft fact cards, print a short preview."""
    args = parse_args(argv)
    in_path: Path = args.input
    out_path: Path = args.output

    if not in_path.is_file():
        print(f"[error] input not found: {in_path}", file=sys.stderr)
        print("Run: python scripts/download_data.py   # or --smoke-only", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    drafts: List[Dict[str, Any]] = []
    scanned = 0
    seq = args.seed_seq

    print(f"Input     : {in_path}")
    print(f"Output    : {out_path}")
    print(f"Method    : {EXTRACT_METHOD} ({SCHEMA_ID})")
    print(f"Max drafts: {args.max_drafts}")
    print("-" * 60)

    # Stream articles so full dumps do not need to sit in memory.
    for article in iter_jsonl(in_path):
        scanned += 1
        facts = extract_from_article(
            article,
            min_chars_article=args.min_article_chars,
            max_chars_article=args.max_article_chars,
            min_facts=args.min_facts,
            max_facts=args.max_facts,
            min_sent_chars=args.min_sent_chars,
            max_sent_chars=args.max_sent_chars,
        )
        if not facts:
            continue
        card = make_card(
            article=article,
            facts=facts,
            hf_config=args.hf_config,
            seq=seq,
        )
        seq += 1
        drafts.append(card)
        if len(drafts) >= args.max_drafts:
            break

    # Pretty JSON array: easier to open/edit in an IDE than JSONL while reviewing.
    # Approved cards are later exported to JSONL via promote_fact_cards.py.
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(drafts, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Scanned articles : {scanned:,}")
    print(f"Draft cards      : {len(drafts):,}")
    print(f"Wrote            : {out_path.relative_to(REPO_ROOT)}  (JSON array)")
    if drafts:
        print("-" * 60)
        print("First draft preview:")
        d0 = drafts[0]
        print(f"  id     : {d0['id']}")
        print(f"  topic  : {d0['topic']}")
        print(f"  domain : {d0['domain']}")
        print(f"  url    : {d0['source'].get('url')}")
        for i, fact in enumerate(d0["facts"], 1):
            print(f"  fact {i}: {fact}")
    print("-" * 60)
    print("Next:")
    print("  1. Edit the drafts JSON — fix facts, set split, review_status=approved")
    print("  2. python scripts/promote_fact_cards.py")
    print("See data/SCHEMA.md for the full contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
