# Lane C — human evaluation

Blind side-by-side scoring of B0/B1/M2 (or any set of checkpoints) on a
frozen eval prompt set. See [`rubric.md`](./rubric.md) for the scoring
criteria and blind-scoring protocol.

## 1. Eval prompts

[`prompts/eval_prompts.jsonl`](./prompts/eval_prompts.jsonl) currently holds
3 placeholder rows. **Replace it with the real ~100-prompt set** before final
scoring — same schema, one JSON object per line:

```json
{"id": "p001", "prompt": "Write a short story about ..."}
```

These are independent of the training fact cards in `data/fact_cards/` — no
gold facts list is attached to an eval prompt.

## 2. Generate stories for each system

Once checkpoints exist under `results/<run_id>/checkpoint.pt`:

```bash
python eval/generate_samples.py \
  --system B0=results/b0_full/checkpoint.pt \
  --system B1=results/b1_cpt/checkpoint.pt \
  --system M2=results/m2_sft_full/checkpoint.pt \
  --prompts eval/prompts/eval_prompts.jsonl \
  --out eval/generations/run_YYYYMMDD.jsonl
```

This reuses `FIXED_EVAL_DECODING` from `scripts/lab_gpt/generation.py` so
every system is generated under identical decoding, and records each story's
perplexity under its own generating model.

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
`(prompt, system)`, with the real `system_id`, shown blind label, all four
Likert scores, optional error tags, and perplexity. This is the data source
for the paper's faithfulness/quality tables, main figure, and error analysis.
