# Lane A tracker — fact cards & handoff

Owner: **Data (Lane A)**  
Contract: [`SCHEMA.md`](./SCHEMA.md) · Env: [`ENV_SETUP.md`](./ENV_SETUP.md) · Licenses: [`LICENSES.md`](./LICENSES.md)

This file is the build log for the card/SFT set. Day-to-day paths for B/C are summarized in [`README.md`](./README.md) (Status section).

---

## Current snapshot

| Artifact | Path | Notes |
| --- | --- | --- |
| Approved train cards | `fact_cards/train.jsonl` | **866** approved |
| Approved eval cards | `fact_cards/eval.jsonl` | **235** held-out — no SFT |
| SFT stories | `sft_pairs/train.jsonl` | **866** approved (1:1 with train) |
| Prompt pack | `prompts/templates.json` | Shared B + C render |
| Draft queues | `fact_cards/drafts/wiki_candidates*.json` | Batches 1–5 (historical review) |

**Scale note:** Early course lock was ~80–200 train / 40–60 eval. We oversized the pool for the Stage-1 geometry in `configs/b0_full.yaml` (~78M params: 10L / 768d / 12H). Lane C scores a frozen **100-prompt** subset (`eval/prompts/frozen_eval_ids.txt`), not all 235 eval cards.

---

## Checklist

### 1. Review drafts

- [x] Batches 1–5 (+5b) reviewed and promoted (see log)
- [x] Facts paraphrased for clarity / story-coverability (4–7 bullets)
- [x] `split` set train/eval; thin/broken cards rejected
- [x] `false_claims` filled on approved cards

### 2. Promote approved drafts → JSONL

- [x] Promoted into `fact_cards/train.jsonl` and `eval.jsonl`
- [x] Spot-check: eval IDs are **not** used for SFT

### 3. Write SFT stories (train only)

- [x] Every approved train card has one ~120–180 word story
- [x] `data/sft_pairs/train.jsonl` (`sft_<slug>_a`, `review_status: approved`)
- [x] No SFT pairs for `split: eval` cards

### 4. Handoff for teammates (B / C)

- [x] **Schema + prompts:** `data/SCHEMA.md`, `data/prompts/templates.json`
- [x] **Approved cards:** `data/fact_cards/train.jsonl`, `data/fact_cards/eval.jsonl`
- [x] **SFT pairs (M2):** `data/sft_pairs/train.jsonl`
- [x] **Corpora / rebuild:** `data/ENV_SETUP.md` + `python scripts/download_data.py`
- [x] **Licenses:** `data/LICENSES.md`

**Lane B (M2):** approved train JSONL + SFT stories.  
**Lane C (eval):** frozen eval JSONL + same prompt template; scored subset via `eval/prompts/frozen_eval_ids.txt`.

---

## Done when (minimum handoff)

- [x] ≥10 approved train cards in JSONL — **866 train**
- [x] Eval IDs frozen — **235** in JSONL; **100** in the scored prompt pack
- [x] Every train card used for M2 has ≥1 approved SFT story (866/866)
- [x] Teammates have stable paths; schema not thrashing

---

## Log

| Date | What changed |
| --- | --- |
| 2026-07-22 | Tracker created; seeds: 2 train, 1 eval, 1 SFT pair; 25 wiki drafts unreviewed |
| 2026-07-24 | Batch 1 promoted: 20 train + 5 eval + 20 SFT stories |
| 2026-07-24 | Batch 2 extracted: 50 drafts; QA 44 approved / 6 rejected; SFT stories appended |
| 2026-07-24 | Batch 3: 40 approved / 10 rejected; +30 train / +10 eval / +30 SFT |
| 2026-07-24 | Batch 4: 162 approved / 38 rejected; totals **213 / 58 / 213** |
| 2026-07-24 | Batch 5 (+5b): 830 approved / 250 rejected; totals **866 / 235 / 866** |
| 2026-07-26 | Docs pass: marked handoff complete; corrected model-size note (~78M, not ~35M) |
