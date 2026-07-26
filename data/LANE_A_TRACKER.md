# Lane A tracker — fact cards & handoff

Owner: **Data (Lane A)**  
Contract: [`SCHEMA.md`](./SCHEMA.md) · Env: [`ENV_SETUP.md`](./ENV_SETUP.md) · Licenses: [`LICENSES.md`](./LICENSES.md)

Update checkboxes as you go. Teammates can use this as the “what’s ready?” board.

---

## Current snapshot

| Artifact | Path | Notes |
| --- | --- | --- |
| Draft queue | `fact_cards/drafts/wiki_candidates.json` | Batch 1 (promoted) |
| Draft queue (batch 2) | `fact_cards/drafts/wiki_candidates_batch2.json` | 44 approved / 6 rejected |
| Draft queue (batch 3) | `fact_cards/drafts/wiki_candidates_batch3.json` | 40 approved / 10 rejected |
| Draft queue (batch 4) | `fact_cards/drafts/wiki_candidates_batch4.json` | 162 approved / 38 rejected |
| Draft queue (batch 5) | `fact_cards/drafts/wiki_candidates_batch5.json` | **830 approved / 250 rejected** (incl. 5b top-up) |
| Draft queue (batch 5b) | `fact_cards/drafts/wiki_candidates_batch5b.json` | 62 approved / 18 rejected (merged into batch 5) |
| Approved train cards | `fact_cards/train.jsonl` | **866** approved (~4.1× pre-batch-5) |
| Approved eval cards | `fact_cards/eval.jsonl` | **235** frozen IDs — no SFT (~4.1×) |
| SFT stories | `sft_pairs/train.jsonl` | **866** approved (1:1 with train) |
| Prompt pack | `prompts/templates.json` | Shared B + C render |

**Scale note:** Course lock was ~80–200 train / 40–60 eval. We oversized for a ~35M model (less SFT memorization at fixed M2 budget).

---

## Checklist

### 1. Review drafts

- [x] Batch 1 `wiki_candidates.json` reviewed and promoted
- [x] Batch 2 `wiki_candidates_batch2.json` QA + approve/reject
- [x] Batch 3 `wiki_candidates_batch3.json` QA + approve/reject
- [x] Batch 4 `wiki_candidates_batch4.json` QA + approve/reject
- [x] Batch 5 (+5b) `wiki_candidates_batch5.json` QA + approve/reject (1080 drafts)
- [x] Facts paraphrased for clarity / story-coverability (4–7 bullets)
- [x] `split` set train/eval; thin/broken cards rejected
- [x] `false_claims` filled on approved cards
- [ ] Optional: another extract batch  
  `python scripts/extract_fact_card_drafts.py --max-drafts 50 --seed-seq 1406 ...`

### 2. Promote approved drafts → JSONL

- [x] Promote batches 2–5 into train/eval JSONL
- [x] Confirm rows in `fact_cards/train.jsonl` and `eval.jsonl`
- [x] Spot-check: eval IDs are **not** used for SFT

### 3. Write SFT stories (train only)

- [x] Every approved train card has one ~120–180 word story
- [x] Appended to `data/sft_pairs/train.jsonl` (`sft_<slug>_a`, `review_status: approved`)
- [x] No SFT pairs for `split: eval` cards

### 4. Handoff for teammates (B / C)

Point people at:

- [ ] **Schema + prompts:** `data/SCHEMA.md`, `data/prompts/templates.json`
- [ ] **Approved cards:** `data/fact_cards/train.jsonl`, `data/fact_cards/eval.jsonl`
- [ ] **SFT pairs (M2):** `data/sft_pairs/train.jsonl`
- [ ] **Corpora / rebuild:** `data/raw/…` or `data/ENV_SETUP.md` + `python scripts/download_data.py`
- [ ] **Licenses:** `data/LICENSES.md`
- [ ] Share the short process blurb (drafts = JSON → promote → JSONL; B loads JSONL only)

**Lane B needs for M2:** approved train JSONL + SFT stories.  
**Lane C needs for eval:** frozen eval JSONL + same prompt template (stories optional for scoring model outputs).

---

## Done when (minimum handoff)

- [x] ≥10 approved train cards in JSONL — **866 train**
- [x] Eval IDs frozen — **235 eval**
- [x] Every train card used for M2 has ≥1 approved SFT story (866/866)
- [ ] Teammates know paths above; schema not thrashing

---

## Log (optional)

| Date | What changed |
| --- | --- |
| 2026-07-22 | Tracker created; seeds: 2 train, 1 eval, 1 SFT pair; 25 wiki drafts unreviewed |
| 2026-07-24 | Batch 1 promoted: 20 train + 5 eval + 20 SFT stories |
| 2026-07-24 | Batch 2 extracted: 50 drafts in `wiki_candidates_batch2.json` (40 train / 10 eval assigned) |
| 2026-07-24 | Batch 2 QA pass: subject-first facts, false_claims filled, 44 approved / 6 thin-meta rejected |
| 2026-07-24 | Batch 2 SFT: 34 gold stories appended to `sft_pairs/train.jsonl` (~120–180 words each) |
| 2026-07-24 | Batch 3 extracted + QA: 40 approved / 10 rejected; promoted 30 train + 10 eval; 30 SFT stories |
| 2026-07-24 | Batch 4 extracted 200 drafts; QA 162 approved / 38 rejected; promoted +129 train / +33 eval; 129 SFT stories (totals **213 / 58 / 213**) |
| 2026-07-24 | Batch 5 (+5b): 1080 drafts; 830 approved / 250 rejected; +653 train / +177 eval / +653 SFT → totals **866 / 235 / 866** (~4.1×) |
