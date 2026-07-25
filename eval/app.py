"""Blind human-eval scoring UI (Lane C).

Reads a generations file produced by generate_samples.py (or the shipped
eval/generations/placeholder.jsonl for a dry run), shows each prompt's
stories side by side under randomized "Model A/B/..." labels (never the real
system id), collects the rubric in eval/rubric.md, and appends one row per
system per prompt to eval/scores.csv.

Run with: streamlit run eval/app.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATIONS_DIR = REPO_ROOT / "eval" / "generations"
SCORES_CSV = REPO_ROOT / "eval" / "scores.csv"
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
    "timestamp", "prompt_id", "prompt_text", "shown_label", "system_id",
    "story_text", "grammar", "factual_correctness", "storytelling_creativity",
    "coherence", "error_tags", "perplexity", "num_tokens",
]

st.set_page_config(page_title="LLM Story Eval", layout="wide")


def newest_generations_file() -> Path | None:
    files = sorted(GENERATIONS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


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


def load_scores() -> List[Dict[str, str]]:
    if not SCORES_CSV.exists():
        return []
    with SCORES_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def scored_prompt_ids(scores: List[Dict[str, str]], generations: Dict[str, List[Dict[str, Any]]]) -> set:
    by_prompt: Dict[str, set] = {}
    for row in scores:
        by_prompt.setdefault(row["prompt_id"], set()).add(row["system_id"])
    done = set()
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


def append_scores(rows: List[Dict[str, Any]]) -> None:
    is_new = not SCORES_CSV.exists() or SCORES_CSV.stat().st_size == 0
    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SCORES_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    st.title("LLM Story Output — Blind Human Evaluation")

    default_path = newest_generations_file()
    with st.sidebar:
        st.header("Data source")
        path_str = st.text_input(
            "Generations file",
            value=str(default_path) if default_path else "",
            help="A JSONL file written by eval/generate_samples.py",
        )

    if not path_str or not Path(path_str).exists():
        st.warning(
            "No generations file found. Run `python eval/generate_samples.py ...` to generate real "
            "stories, or point this at `eval/generations/placeholder.jsonl` to try the UI first."
        )
        return

    generations = load_generations(path_str)
    prompt_ids = list(generations.keys())
    scores = load_scores()
    done_ids = scored_prompt_ids(scores, generations)

    if not prompt_ids:
        st.info("Generations file is empty.")
        return

    if "current_idx" not in st.session_state:
        first_unscored = next((i for i, pid in enumerate(prompt_ids) if pid not in done_ids), 0)
        st.session_state.current_idx = first_unscored

    with st.sidebar:
        st.metric("Scored", f"{len(done_ids)} / {len(prompt_ids)}")
        jump_options = [f"{'✅' if pid in done_ids else '•'} {pid}" for pid in prompt_ids]
        jump_choice = st.selectbox("Jump to prompt", options=range(len(prompt_ids)),
                                    format_func=lambda i: jump_options[i],
                                    index=st.session_state.current_idx)
        if jump_choice != st.session_state.current_idx:
            st.session_state.current_idx = jump_choice

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
    st.write(prompt_text)

    is_scored = prompt_id in done_ids
    if is_scored:
        with st.expander("Automated metrics (revealed — this prompt is already scored)"):
            for label, system_id in zip(labels, order):
                ppl = by_system[system_id]["perplexity"]
                st.write(f"**{label}** ({system_id}): perplexity = {ppl:.2f}" if isinstance(ppl, (int, float)) else f"**{label}** ({system_id}): perplexity = {ppl}")

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
                    key=f"{prompt_id}_{system_id}_{field}",
                )
            values["error_tags"] = st.multiselect(
                "Error tags (optional)", options=ERROR_TAGS,
                key=f"{prompt_id}_{system_id}_tags",
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
            append_scores(to_save)
            next_unscored = next(
                (i for i in range(idx + 1, len(prompt_ids)) if prompt_ids[i] not in done_ids | {prompt_id}),
                None,
            )
            if next_unscored is None:
                next_unscored = next((i for i, pid in enumerate(prompt_ids) if pid not in done_ids | {prompt_id}), idx)
            st.session_state.current_idx = next_unscored
            st.rerun()

    if is_scored:
        st.info("This prompt is already fully scored. Use 'Jump to prompt' to review, or pick an unscored one.")


if __name__ == "__main__":
    main()
