"""Blind human-eval scoring UI (Lane C).

Reads a generations file produced by generate_samples.py (or a snapshot under
eval/snapshots/), shows each prompt's stories side by side under randomized
"Model A/B/..." labels (never the real system id), collects the rubric in
eval/rubric.md, and appends one row per system per prompt to a scores CSV.

Keep card and nocard ratings in SEPARATE score files — same prompt ids are
reused across conditions. Defaults:

  card   generations → eval/scores_card.csv   (or eval/scores.csv)
  nocard generations → eval/scores_nocard.csv

Run with: streamlit run eval/app.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval"
GENERATIONS_DIR = EVAL_DIR / "generations"
SNAPSHOTS_DIR = EVAL_DIR / "snapshots"
LIKERT_AXES = [
    ("grammar", "Grammar"),
    ("factual_correctness", "Factual correctness"),
    ("storytelling_creativity", "Storytelling creativity"),
    ("coherence", "Coherence"),
]
ERROR_TAGS = [
    "Omission",
    "Contradiction",
    "Unconstrained invention",
    "Encyclopedia dump",
    "Story domination",
]
SCORE_FIELDS = [
    "timestamp", "prompt_id", "card_id", "condition", "prompt_text", "shown_label",
    "system_id", "story_text", "grammar", "factual_correctness",
    "storytelling_creativity", "coherence", "error_tags", "perplexity", "num_tokens",
]

st.set_page_config(page_title="LLM Story Eval", layout="wide")


def discover_generations_files() -> List[Path]:
    """Newest-first list from eval/generations/ and eval/snapshots/**."""
    files: List[Path] = []
    if GENERATIONS_DIR.exists():
        files.extend(GENERATIONS_DIR.glob("*.jsonl"))
    if SNAPSHOTS_DIR.exists():
        files.extend(SNAPSHOTS_DIR.glob("*/*.jsonl"))
    # Prefer real runs over the UI dry-run placeholder.
    files = [p for p in files if p.name != "placeholder.jsonl" or len(files) == 1]
    return sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)


def default_scores_path(condition: str) -> Path:
    if condition == "nocard":
        return EVAL_DIR / "scores_nocard.csv"
    if condition == "card":
        # Prefer the explicit card file once the primary sheet has been renamed.
        card = EVAL_DIR / "scores_card.csv"
        legacy = EVAL_DIR / "scores.csv"
        return card if card.exists() else legacy
    return EVAL_DIR / "scores.csv"


def peek_condition(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return str(json.loads(line).get("condition") or "")
    return ""


@st.cache_data
def load_generations(path_str: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    with Path(path_str).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped.setdefault(row["prompt_id"], []).append(row)
    return grouped


def load_scores(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def scored_prompt_ids(
    scores: List[Dict[str, str]],
    generations: Dict[str, List[Dict[str, Any]]],
    condition: str,
) -> Set[str]:
    """A prompt is done only if every system is scored under the SAME condition."""
    by_prompt: Dict[str, Set[str]] = {}
    for row in scores:
        row_cond = (row.get("condition") or "").strip()
        # Legacy sheets (no condition column) only count for nocard ablation.
        if condition and row_cond and row_cond != condition:
            continue
        if condition and not row_cond and condition != "nocard":
            continue
        by_prompt.setdefault(row["prompt_id"], set()).add(row["system_id"])
    done: Set[str] = set()
    for prompt_id, rows in generations.items():
        expected = {r["system_id"] for r in rows}
        if expected and by_prompt.get(prompt_id, set()) >= expected:
            done.add(prompt_id)
    return done


def shuffled_order(prompt_id: str, system_ids: List[str]) -> List[str]:
    seed = int.from_bytes(hashlib.sha256(prompt_id.encode("utf-8")).digest()[:4], "big")
    order = sorted(system_ids)
    random.Random(seed).shuffle(order)
    return order


def stale_score_header(path: Path) -> Optional[List[str]]:
    """Return the on-disk header if it predates the current SCORE_FIELDS schema."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f), [])
    return header if header != SCORE_FIELDS else None


def append_scores(path: Path, rows: List[Dict[str, Any]]) -> None:
    is_new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    st.title("LLM Story Output — Blind Human Evaluation")

    candidates = discover_generations_files()
    preferred = next(
        (p for p in candidates if "nocard" in p.name and "20260726_card60" in str(p)),
        None,
    )
    if preferred is None:
        preferred = next((p for p in candidates if "nocard" in p.name), None)
    if preferred is None and candidates:
        preferred = candidates[0]

    with st.sidebar:
        st.header("Data source")
        labels = [str(p.relative_to(REPO_ROOT)) for p in candidates] if candidates else []
        if labels:
            default_idx = labels.index(str(preferred.relative_to(REPO_ROOT))) if preferred else 0
            chosen = st.selectbox("Generations file", options=labels, index=default_idx)
            path = REPO_ROOT / chosen
        else:
            path_str = st.text_input("Generations file", value="")
            path = Path(path_str) if path_str else Path()

        # Allow paste override for paths outside the discover list.
        path_override = st.text_input(
            "Or paste a path",
            value="",
            help="Optional absolute/relative path; overrides the dropdown when set.",
        )
        if path_override.strip():
            path = Path(path_override.strip())
            if not path.is_absolute():
                path = REPO_ROOT / path

    if not path or not path.exists():
        st.warning(
            "No generations file found. Run `python eval/generate_samples.py ...` "
            "or point at `eval/snapshots/.../generations_*.jsonl`."
        )
        return

    condition = peek_condition(path)
    suggested_scores = default_scores_path(condition)

    with st.sidebar:
        scores_str = st.text_input(
            "Scores CSV (output)",
            value=str(suggested_scores.relative_to(REPO_ROOT)),
            help="Card and nocard must use different files — prompt ids overlap.",
        )
        scores_path = Path(scores_str)
        if not scores_path.is_absolute():
            scores_path = REPO_ROOT / scores_path
        st.caption(f"Condition in file: `{condition or 'unknown'}`")

    stale = stale_score_header(scores_path)
    if stale is not None:
        st.error(
            f"`{scores_path.name}` has an old column layout, so new rows would be misaligned.\n\n"
            f"On disk: `{','.join(stale)}`\n\nExpected: `{','.join(SCORE_FIELDS)}`\n\n"
            "Move the old file aside (e.g. rename to `scores_nocard_ablation.csv`) and reload."
        )
        return

    generations = load_generations(str(path))
    prompt_ids = list(generations.keys())
    scores = load_scores(scores_path)

    # Guard: don't append nocard rows into a card sheet (or vice versa).
    existing_conditions = {((r.get("condition") or "").strip()) for r in scores} - {""}
    if scores and condition and existing_conditions and condition not in existing_conditions:
        st.error(
            f"Scores file `{scores_path.name}` already has condition(s) "
            f"{sorted(existing_conditions)}, but generations are `{condition}`.\n\n"
            f"Use a different scores path (suggested: `{suggested_scores.relative_to(REPO_ROOT)}`)."
        )
        return

    done_ids = scored_prompt_ids(scores, generations, condition)

    if not prompt_ids:
        st.info("Generations file is empty.")
        return

    if "current_idx" not in st.session_state or st.session_state.get("_gen_path") != str(path):
        first_unscored = next((i for i, pid in enumerate(prompt_ids) if pid not in done_ids), 0)
        st.session_state.current_idx = first_unscored
        st.session_state._gen_path = str(path)

    with st.sidebar:
        st.metric("Scored", f"{len(done_ids)} / {len(prompt_ids)}")
        jump_options = [f"{'✅' if pid in done_ids else '•'} {pid}" for pid in prompt_ids]
        jump_choice = st.selectbox(
            "Jump to prompt",
            options=range(len(prompt_ids)),
            format_func=lambda i: jump_options[i],
            index=min(st.session_state.current_idx, len(prompt_ids) - 1),
        )
        if jump_choice != st.session_state.current_idx:
            st.session_state.current_idx = jump_choice

    if condition == "nocard":
        st.info(
            "Scoring **nocard** (model saw only a bare sentence). "
            "The fact list below is for *your* faithfulness judgment — "
            "it was **not** in the model’s prompt."
        )

    if len(done_ids) == len(prompt_ids):
        st.success("All prompts have been scored.")

    idx = min(st.session_state.current_idx, len(prompt_ids) - 1)
    prompt_id = prompt_ids[idx]
    rows = generations[prompt_id]
    prompt_text = rows[0]["prompt_text"]
    by_system = {r["system_id"]: r for r in rows}
    order = shuffled_order(prompt_id, list(by_system.keys()))
    labels = [f"Model {chr(ord('A') + i)}" for i in range(len(order))]

    st.subheader(f"Prompt `{prompt_id}`")
    facts = rows[0].get("facts") or []
    if facts:
        st.caption(
            f"Topic: **{rows[0].get('topic', '?')}** · condition: `{rows[0].get('condition', '?')}`"
        )
        st.markdown("\n".join(f"{i + 1}. {fact}" for i, fact in enumerate(facts)))
        with st.expander("Exact model input"):
            st.text(prompt_text)
    else:
        st.write(prompt_text)

    is_scored = prompt_id in done_ids
    if is_scored:
        with st.expander("Automated metrics (revealed — this prompt is already scored)"):
            for label, system_id in zip(labels, order):
                ppl = by_system[system_id]["perplexity"]
                if isinstance(ppl, (int, float)):
                    st.write(f"**{label}** ({system_id}): perplexity = {ppl:.2f}")
                else:
                    st.write(f"**{label}** ({system_id}): perplexity = {ppl}")

    cols = st.columns(len(order))
    responses: Dict[str, Dict[str, Any]] = {}
    for col, label, system_id in zip(cols, labels, order):
        with col:
            st.markdown(f"#### {label}")
            with st.container(border=True):
                st.write(by_system[system_id]["story_text"])

            values: Dict[str, Any] = {}
            for field, axis_label in LIKERT_AXES:
                values[field] = st.radio(
                    axis_label, options=[1, 2, 3, 4, 5], index=None, horizontal=True,
                    key=f"{path.name}_{prompt_id}_{system_id}_{field}",
                )
            values["error_tags"] = st.multiselect(
                "Error tags (optional)", options=ERROR_TAGS,
                key=f"{path.name}_{prompt_id}_{system_id}_tags",
            )
            responses[system_id] = values

    st.divider()
    if st.button("Save & Next", type="primary", disabled=is_scored):
        missing = [
            system_id for system_id, values in responses.items()
            if any(values[field] is None for field, _ in LIKERT_AXES)
        ]
        if missing:
            st.error(f"Fill in all four ratings for every model before saving ({len(missing)} incomplete).")
        else:
            now = datetime.now(timezone.utc).isoformat()
            to_save = []
            for label, system_id in zip(labels, order):
                r = by_system[system_id]
                v = responses[system_id]
                to_save.append({
                    "timestamp": now,
                    "prompt_id": prompt_id,
                    "card_id": r.get("card_id", ""),
                    "condition": r.get("condition", condition),
                    "prompt_text": prompt_text,
                    "shown_label": label,
                    "system_id": system_id,
                    "story_text": r["story_text"],
                    "grammar": v["grammar"],
                    "factual_correctness": v["factual_correctness"],
                    "storytelling_creativity": v["storytelling_creativity"],
                    "coherence": v["coherence"],
                    "error_tags": ";".join(v["error_tags"]),
                    "perplexity": r["perplexity"],
                    "num_tokens": r["num_tokens"],
                })
            append_scores(scores_path, to_save)
            next_unscored = next(
                (i for i in range(idx + 1, len(prompt_ids)) if prompt_ids[i] not in done_ids | {prompt_id}),
                None,
            )
            if next_unscored is None:
                next_unscored = next(
                    (i for i, pid in enumerate(prompt_ids) if pid not in done_ids | {prompt_id}),
                    idx,
                )
            st.session_state.current_idx = next_unscored
            st.rerun()

    if is_scored:
        st.info("This prompt is already fully scored. Use 'Jump to prompt' to review, or pick an unscored one.")


if __name__ == "__main__":
    main()
