"""Aggregate eval/scores.csv into the paper's tables (Lane C).

Rating stays manual; this only adds up what the raters saved, so the numbers in
the paper are reproducible from a command instead of an ad-hoc spreadsheet.

Prints, per system:
  * mean Likert score on each of the four rubric axes (+ n)
  * error-tag counts (the paper's required error analysis)
  * perplexity and story length (supporting automatic metrics)

Usage:
    python eval/summarize_scores.py
    python eval/summarize_scores.py --scores eval/scores_nocard_ablation.csv
    python eval/summarize_scores.py --generations eval/generations/run_20260726.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = REPO_ROOT / "eval" / "scores.csv"

AXES = ["grammar", "factual_correctness", "storytelling_creativity", "coherence"]
ERROR_TAGS = [
    "Omission",
    "Contradiction",
    "Unconstrained invention",
    "Encyclopedia dump",
    "Story domination",
]


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def floats(rows: List[Dict[str, str]], field: str) -> List[float]:
    out = []
    for r in rows:
        raw = (r.get(field) or "").strip()
        if not raw:
            continue
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def mean_or_none(values: List[float]):
    return statistics.mean(values) if values else None


def fmt(value, spec: str = ".2f") -> str:
    return "--" if value is None else format(value, spec)


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    line = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    print(line)
    print(sep)
    for r in rows:
        print("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--generations", type=Path,
                    help="Optional: report how many of this file's prompts still need scoring")
    args = ap.parse_args()

    if not args.scores.exists():
        raise SystemExit(f"No scores file at {args.scores} -- run `streamlit run eval/app.py` first.")

    rows = load_rows(args.scores)
    if not rows:
        raise SystemExit(f"{args.scores} has no rows yet.")

    conditions = {(r.get("condition") or "unlabeled") for r in rows}
    systems = sorted({r["system_id"] for r in rows})
    prompts = {r["prompt_id"] for r in rows}

    print(f"Scores      : {args.scores}")
    print(f"Condition   : {', '.join(sorted(conditions))}"
          + ("   <-- MIXED: split before reporting" if len(conditions) > 1 else ""))
    print(f"Systems     : {', '.join(systems)}")
    print(f"Prompts     : {len(prompts)} scored\n")

    print("## Human ratings (1-5, higher is better)\n")
    table = []
    for system in systems:
        srows = [r for r in rows if r["system_id"] == system]
        cells = [system, str(len(srows))]
        for axis in AXES:
            values = floats(srows, axis)
            cells.append(f"{fmt(mean_or_none(values))}" + (f" (n={len(values)})" if len(values) != len(srows) else ""))
        table.append(cells)
    print_table(["system", "n", *AXES], table)

    print("\n## Supporting automatic metrics\n")
    print("Perplexity is each sample's score under its OWN generating model: a")
    print("fluency/confidence proxy that rewards repetitive degeneration, not a")
    print("quality measure. See eval/rubric.md before citing it.\n")
    table = []
    for system in systems:
        srows = [r for r in rows if r["system_id"] == system]
        ppl = floats(srows, "perplexity")
        toks = floats(srows, "num_tokens")
        table.append([
            system,
            fmt(mean_or_none(ppl), ".1f"),
            fmt(statistics.median(ppl) if ppl else None, ".1f"),
            fmt(mean_or_none(toks), ".0f"),
        ])
    print_table(["system", "mean ppl", "median ppl", "mean story tokens"], table)

    print("\n## Error analysis\n")
    tag_counts: Dict[str, Counter] = defaultdict(Counter)
    tagged = 0
    for r in rows:
        raw = (r.get("error_tags") or "").strip()
        if not raw:
            continue
        tagged += 1
        for tag in raw.split(";"):
            tag = tag.strip()
            if tag:
                tag_counts[r["system_id"]][tag] += 1

    if not tagged:
        print("No error tags recorded on any of "
              f"{len(rows)} rows. The rubric's five tags feed the paper's required")
        print("error-analysis section -- they are optional in the UI, so a rater has to")
        print("apply them deliberately while scoring.")
    else:
        table = []
        for system in systems:
            counts = tag_counts.get(system, Counter())
            table.append([system, *(str(counts.get(tag, 0)) for tag in ERROR_TAGS)])
        print_table(["system", *ERROR_TAGS], table)
        print(f"\n{tagged}/{len(rows)} rated stories carry at least one tag.")

    if args.generations:
        gen_prompts = set()
        with args.generations.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    gen_prompts.add(json.loads(line)["prompt_id"])
        missing = sorted(gen_prompts - prompts)
        print(f"\n## Coverage vs {args.generations.name}\n")
        print(f"{len(gen_prompts) - len(missing)}/{len(gen_prompts)} prompts scored.")
        if missing:
            preview = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
            print(f"Unscored: {preview}")


if __name__ == "__main__":
    main()
