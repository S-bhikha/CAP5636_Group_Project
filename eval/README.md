# Lane C — human evaluation

Blind side-by-side scoring of B0/B1/M2 (or any set of checkpoints) on a
frozen eval prompt set. See [`rubric.md`](./rubric.md) for the scoring
criteria and blind-scoring protocol.

## Runbook

Picks up after [`TRAINING.md`](../TRAINING.md) has produced the three
checkpoints. Run from the repo root with `.venv` active; `RUN=$(date +%Y%m%d)`
just keeps the filenames consistent across the commands.

```bash
RUN=$(date +%Y%m%d)

# 1. Build the prompt packs from Lane A's frozen eval cards (regenerate whenever
#    data/fact_cards/eval.jsonl changes).
python eval/build_eval_prompts.py --condition card   --out eval/prompts/eval_prompts.jsonl
python eval/build_eval_prompts.py --condition nocard --out eval/prompts/eval_prompts_nocard.jsonl

# 2. Generate stories for every system under identical decoding (primary set).
python eval/generate_samples.py \
  --system B0=results/b0_full_768/checkpoint.pt \
  --system B1=results/b1_cpt_full_768/checkpoint.pt \
  --system M2=results/m2_sft_full_768/checkpoint.pt \
  --prompts eval/prompts/eval_prompts.jsonl \
  --out eval/generations/run_$RUN.jsonl

# 3. Score blind, by hand. Point the sidebar at eval/generations/run_$RUN.jsonl.
streamlit run eval/app.py

# 4. Aggregate the ratings into the paper's tables.
python eval/summarize_scores.py --generations eval/generations/run_$RUN.jsonl

# 5. Prompt ablation (optional, same three checkpoints, its OWN files).
python eval/generate_samples.py \
  --system B0=results/b0_full_768/checkpoint.pt \
  --system B1=results/b1_cpt_full_768/checkpoint.pt \
  --system M2=results/m2_sft_full_768/checkpoint.pt \
  --prompts eval/prompts/eval_prompts_nocard.jsonl \
  --out eval/generations/run_${RUN}_nocard.jsonl
```

**Check before moving on**

| After | Verify |
| --- | --- |
| Step 1 | Each command prints `Wrote 100 ... prompts`; the `card` pack reports a max of ~223 tokens, inside the 340-token prompt budget |
| Step 2 | 300 rows written (100 prompts x 3 systems), no context-budget error, and few stories hitting `max_new_tokens` |
| Step 3 | Sidebar shows `Scored 100 / 100`; apply error tags as you go (they are optional in the UI but required by the paper) |
| Step 4 | Condition line reads `card`, not `MIXED`; coverage reads 100/100 |

Scoring is deliberately manual. Step 5's ablation is scored the same way, but
keep it in a separate `scores.csv` (move the primary file aside first) so the
two conditions never mix in one table.

## 1. Eval prompts

Prompt packs are **generated**, never hand-written, by
[`build_eval_prompts.py`](./build_eval_prompts.py) from Lane A's frozen
held-out cards (`data/fact_cards/eval.jsonl`) using the same
`render_model_input()` that builds M2's SFT inputs:

```bash
# primary set: Topic + numbered facts + frozen instruction (what M2 is trained on)
python eval/build_eval_prompts.py --condition card \
  --out eval/prompts/eval_prompts.jsonl

# ablation set: first fact only, no fact card in context
python eval/build_eval_prompts.py --condition nocard \
  --out eval/prompts/eval_prompts_nocard.jsonl
```

Both conditions use the same 100 topics under the same prompt ids
(`prompts/frozen_eval_ids.txt` maps `prompt_id -> card_id`), so the two runs
are **paired per topic** and answer the README's prompt ablation directly.

> **Why this matters.** M2 is supervised on the rendered
> "Topic / Facts / Instruction" format. An earlier pack held bare one-sentence
> prompts, which tested M2 off-distribution and understated it. The `card`
> condition is the primary set for all systems; `nocard` is the ablation, not
> the headline result. Do not edit a pack by hand — rebuild it.

Each row carries `id`, `card_id`, `condition`, `topic`, `facts`, `prompt`. The
`facts` list is what the scoring UI shows raters so faithfulness and the
**Omission** tag can be judged against the card.

## 2. Generate stories for each system

Once checkpoints exist under `results/<run_id>/checkpoint.pt`:

```bash
python eval/generate_samples.py \
  --system B0=results/b0_full_768/checkpoint.pt \
  --system B1=results/b1_cpt_full_768/checkpoint.pt \
  --system M2=results/m2_sft_full_768/checkpoint.pt \
  --prompts eval/prompts/eval_prompts.jsonl \
  --out eval/generations/run_YYYYMMDD.jsonl

# ablation: same command, other pack, its OWN output file
python eval/generate_samples.py --system ... \
  --prompts eval/prompts/eval_prompts_nocard.jsonl \
  --out eval/generations/run_YYYYMMDD_nocard.jsonl
```

This reuses `FIXED_EVAL_DECODING` from `scripts/lab_gpt/generation.py` so
every system is generated under identical decoding, and records each story's
perplexity under its own generating model.

Two guards will stop a run rather than produce numbers that can't be compared:

- **Context budget** — a rendered card is ~160 tokens (max 223), so with
  `max_new_tokens=300` the checkpoint needs `block_size >= ~525`; the project
  trains at 640. Decoding keeps only the last `block_size` tokens, so an
  overlong prompt does not crash — the card silently scrolls out of context
  mid-story and the model is graded on facts it can no longer see. Override
  only for debugging with `--allow-context-overflow`.
- **Mixed conditions** — one condition per generations file, so `card` and
  `nocard` scores never land in the same table.

To try the scoring UI before any checkpoint exists, use the shipped
[`generations/placeholder.jsonl`](./generations/placeholder.jsonl) (fake
stories, 3 systems x 3 prompts).

## 3. Score

```bash
streamlit run eval/app.py
```

The sidebar lets you pick which generations file to load (defaults to the
newest file in `eval/generations/`). Stories are shown under randomized
"Model A/B/..." labels — the app never reveals which real system produced a
story while you're scoring. Perplexity is hidden until a prompt is fully
scored.

Click **Save & Next** once all four ratings are filled in for every model on
a prompt; progress auto-resumes from the first unscored prompt on reload.

## 4. Output

Scores accumulate in `eval/scores.csv` (schema in
[`score_sheet_template.csv`](./score_sheet_template.csv)) — one row per
`(prompt, system)`, with the real `system_id`, shown blind label, `card_id`,
`condition`, all four Likert scores, optional error tags, and perplexity. This
is the data source for the paper's faithfulness/quality tables, main figure,
and error analysis. Join on `system_id` and `condition`, never on
`shown_label` (the blind label differs per prompt).

`eval/scores_nocard_ablation.csv` holds the 50 prompts x 3 systems scored
against the old bare-sentence pack. Those prompts are byte-identical to the
`nocard` pack, so the file stands as the ablation arm's ratings — it is not
the primary result and its column layout predates `card_id`/`condition`.

## 5. Aggregate

[`summarize_scores.py`](./summarize_scores.py) turns the saved ratings into the
paper's tables, so reported numbers come from a command rather than a
spreadsheet:

```bash
python eval/summarize_scores.py                                   # eval/scores.csv
python eval/summarize_scores.py --generations eval/generations/run_20260726.jsonl
python eval/summarize_scores.py --scores eval/scores_nocard_ablation.csv
```

It prints per-system Likert means, the error-tag counts that feed the required
error-analysis section, perplexity/length, and — with `--generations` — which
prompt ids are still unscored. It warns instead of averaging silently when a
scores file mixes `card` and `nocard` rows.
