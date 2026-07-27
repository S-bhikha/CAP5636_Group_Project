# Data (Lane A)

Local corpora and task artifacts for **Fact-Constrained Story Generation**.

## Layout

```text
data/
  SCHEMA.md               # fact-card / SFT / prompt contract (Lane A↔B↔C)
  LANE_A_TRACKER.md       # build log + counts (historical checklist)
  ENV_SETUP.md            # venv + download
  LICENSES.md             # corpus license obligations
  raw/
    tinystories/          # Stage 1 pretraining source
    wikipedia/            # Stage 2 B1 CPT source
  manifests/
    local_corpora.json
  fact_cards/
    train.jsonl           # 866 approved train cards (JSONL — B/C load)
    eval.jsonl            # 235 approved held-out cards (JSONL)
    drafts/               # review queues (JSON arrays; batches 1–5)
  sft_pairs/
    train.jsonl           # 866 approved card_id → story (M2)
  prompts/
    templates.json        # frozen instruction + model-input render
```

Raw dumps under `data/raw/` are **gitignored** (regenerate with the script).  
Schema, approved cards, templates, manifests, and license docs are tracked.

**Drafts vs approved:** edit drafts as pretty **JSON**; promote to **JSONL** for training/eval. B and C load JSONL only.

## Corpora we use

| Corpus | HF id | Role | Default local pull |
| --- | --- | --- | --- |
| **TinyStories** | [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) | Stage 1 next-token pretraining (B0) | full train + validation; always also smoke 10k/1k |
| **Wikipedia (Simple English)** | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) config `20231101.simple` | Stage 2 B1 encyclopedic CPT | full ~242k articles; always also smoke 5k |

**Why Simple English for B1?** Full English Wikipedia is multi‑GB and far larger than a matched Stage‑2 student budget. Simple English is still encyclopedic, smaller (~242k articles), and closer in language complexity to TinyStories. Swap to `20231101.en` with a row cap if the team wants “full enwiki language” instead:

```bash
python scripts/download_data.py --skip-tinystories --wiki-config 20231101.en --wiki-max-rows 50000
```

## Setup

```bash
# from repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download

```bash
source .venv/bin/activate

# 1) Smoke only — enough for Stage-1 smoke + B1 dry-run (minutes)
python scripts/download_data.py --smoke-only

# 2) Full TinyStories + full Simple English Wikipedia (longer; needs disk)
python scripts/download_data.py

# Re-download even if files exist
python scripts/download_data.py --force

# One corpus only
python scripts/download_data.py --skip-wikipedia
python scripts/download_data.py --skip-tinystories
```

Each line of a `*.jsonl` file is one JSON object:

- TinyStories: `{"text": "..."}`
- Wikipedia: `{"id": "...", "url": "...", "title": "...", "text": "..."}`

## Handoff paths (A → B)

| Need | Path |
| --- | --- |
| Stage-1 smoke | `data/raw/tinystories/train_smoke.jsonl` |
| Stage-1 full | `data/raw/tinystories/train.jsonl` (+ `validation.jsonl`) |
| B1 smoke | `data/raw/wikipedia/20231101_simple_smoke.jsonl` |
| B1 full | `data/raw/wikipedia/20231101_simple.jsonl` |
| Manifest | `data/manifests/local_corpora.json` |

## Licenses & acknowledgment (required)

**Authoritative checklist:** [`data/LICENSES.md`](./LICENSES.md)  
**Machine-readable metadata:** [`data/licenses.json`](./licenses.json)

| Corpus | License | What we must do |
| --- | --- | --- |
| **TinyStories** | **[CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/)** ([HF card](https://huggingface.co/datasets/roneneldan/TinyStories)) | Cite Eldan & Li (2023) + HF id; if we *publish* the data (or enhanced data), keep CDLA-Sharing + attribution. Model/metrics “Results” are not share-alike-locked by the license text. Do not commit raw dumps. |
| **Wikipedia** | **CC BY-SA** + **GFDL** ([HF](https://huggingface.co/datasets/wikimedia/wikipedia): `cc-by-sa-3.0`, `gfdl`; [dumps legal](https://dumps.wikimedia.org/legal.html) also documents dual free licensing) | Attribute Wikipedia/Wikimedia contributors; link license; note processing changes; ShareAlike when redistributing adapted *text* corpora. Do not claim ownership of article text. Do not commit raw dumps. |

### Paper-ready acknowledgment (draft)

```text
Data and licenses. Stage-1 pretraining uses the TinyStories dataset
(Eldan & Li, 2023; Hugging Face: roneneldan/TinyStories), distributed under
the Community Data License Agreement – Sharing 1.0 (CDLA-Sharing-1.0).
Stage-2 encyclopedic continued pretraining (B1) uses Simple English Wikipedia
articles from the wikimedia/wikipedia packaging (config 20231101.simple),
derived from Wikimedia dumps. Wikipedia textual content is available under
Creative Commons Attribution-ShareAlike and the GNU Free Documentation License
(see the dataset card and https://dumps.wikimedia.org/legal.html). We attribute
Wikipedia/Wikimedia contributors and retain license notices for any
redistributed text extracts. We do not redistribute full raw dumps in the
repository; rebuild commands are provided instead.
```

Full obligations, share-alike scope, and the hand-in checklist live in **`data/LICENSES.md`**.

## Fact cards & Wiki draft extraction

**Contract:** [`SCHEMA.md`](./SCHEMA.md) (fields, model render, split rules, review flow).

```bash
source .venv/bin/activate

# 1) Propose draft cards (JSON array for easy IDE review)
python scripts/extract_fact_card_drafts.py \
  --input data/raw/wikipedia/20231101_simple_smoke.jsonl \
  --max-drafts 30
# → data/fact_cards/drafts/wiki_candidates.json

# 2) Edit that JSON: fix facts, set split, set review_status to "approved"

# 3) Export approved cards to train/eval JSONL
python scripts/promote_fact_cards.py --dry-run   # preview
python scripts/promote_fact_cards.py             # write JSONL
```

| File | Format | Role |
| --- | --- | --- |
| `fact_cards/drafts/wiki_candidates.json` | JSON array | Human review queue |
| `fact_cards/train.jsonl` / `eval.jsonl` | JSONL | Approved cards for B/C |

## Status

Lane A handoff is **done** for the reported runs:

| Artifact | Count |
| --- | --- |
| `fact_cards/train.jsonl` | 866 |
| `fact_cards/eval.jsonl` | 235 |
| `sft_pairs/train.jsonl` | 866 |

Eval scoring uses a frozen **100-prompt** subset of the eval cards (`eval/prompts/frozen_eval_ids.txt`), not all 235. Build log and batch history: [`LANE_A_TRACKER.md`](./LANE_A_TRACKER.md). Contract: [`SCHEMA.md`](./SCHEMA.md).
