# Fact-Constrained Story Generation

**CAP 5636 Final Project** — TinyStories-scale LM + factual narrative adaptation

**Team (2–3):** Sahil Bhikha · Thomas Belyakov · David Almeida II

---

## One-line summary

**Pretrain** a small decoder-only Transformer on **TinyStories**, then **adapt** it so it can write short **stories that stay faithful to a provided fact card**, and measure the **faithfulness–narrative quality tradeoff** against matched controls.

## Research question

Under a fixed small-model and token budget, which adaptation strategy best improves **fact faithfulness** of short educational stories **without destroying narrative quality**, relative to TinyStories-only and pure encyclopedic continued pretraining?

## Contribution (paper framing)

We study **fact-constrained story generation** with a TinyStories-scale decoder-only LM. We build an evaluation suite of topic–fact-card–prompt triples and compare matched-budget adaptation recipes (story-only baseline, Wiki continued pretraining, and task-specific SFT). We characterize the **faithfulness vs story-quality** tradeoff and analyze failure modes (contradiction, omission, unconstrained invention, encyclopedia dump).

This is a **controlled empirical study**. We do **not** claim general truthfulness, educational safety, or open-world hallucination reduction.

---

## Task definition

Each item is closed-world and scoreable:

| Field | Role |
| --- | --- |
| **Topic** | e.g. water cycle, seasons, a simple historical figure |
| **Fact card** | 4–7 gold bullets the story may teach; optional notes on common false claims |
| **Prompt** | Rendered as `Topic` + numbered `Facts` + a frozen instruction (same string used for M2 SFT and primary eval) |
| **Output** | One short story (~120–180 words) |

**Success axes** (how the paper talks about them) map to the human rubric in [`eval/rubric.md`](./eval/rubric.md):

| Paper axis | Rubric scores |
| --- | --- |
| **Faithfulness** | Factual correctness (+ error tags: omission, contradiction, unconstrained invention) |
| **Story quality** | Grammar, storytelling creativity, coherence (+ tags: encyclopedia dump, story domination) |

## Systems

| ID | System | Role |
| --- | --- | --- |
| **B0** | TinyStories-only (Stage 1) | Narrative prior; faithfulness floor |
| **B1** | B0 + Wikipedia continued pretraining | Encyclopedic / “more world text” control (**required baseline**) |
| **M2** | B0 + SFT on (fact card + prompt → story) | Primary task adaptation |

Optional if time allows: **M1** (CPT on factual narratives), **M3** (B1 then light SFT), **prompt ablation** (same checkpoints with vs without the fact card in context — `card` vs `nocard` packs in [`eval/`](./eval/)).

**Deadline triage:** ship **B0, B1, M2** under equal Stage-2 budgets. That set is the empirical study.

## Data (what is in the repo)

| Artifact | Path | Count |
| --- | --- | --- |
| Train fact cards | `data/fact_cards/train.jsonl` | **866** approved |
| Held-out eval cards | `data/fact_cards/eval.jsonl` | **235** approved (never used for SFT) |
| SFT pairs | `data/sft_pairs/train.jsonl` | **866** (1:1 with train) |
| Frozen eval prompts | `eval/prompts/frozen_eval_ids.txt` | **100** topics (subset of eval cards) |

Early planning targeted ~80–200 train / 40–60 eval. We oversized the card pool for a larger Stage-1 model (~78M params in `configs/b0_full.yaml`) and froze a **100-prompt** scored subset for the paper tables. Details: [`data/SCHEMA.md`](./data/SCHEMA.md), [`data/LANE_A_TRACKER.md`](./data/LANE_A_TRACKER.md).

Raw TinyStories / Wikipedia dumps are **gitignored** — rebuild with `python scripts/download_data.py` ([`data/README.md`](./data/README.md)).

## Training stages

| Stage | What happens | Starts from |
| --- | --- | --- |
| **1 — Pretraining (B0)** | Next-token LM on TinyStories | Random init |
| **2 — Adaptation** | Matched-budget Wiki CPT (**B1**) and/or SFT (**M2**) | Stage-1 checkpoint |

**Locked:** Stage 1 trains from scratch on TinyStories (lab-scale GPT port). Do not start from a public pretrained model unless Stage 1 fails after smoke.

Geometry for reported runs lives in YAML (`configs/b0_full.yaml`: 10L / 768d / 12H, `vocab_size=10000`, `block_size=640`). `block_size` is set by Stage 1 so M2 can fit a full rendered card + gold story in one window — see [`TRAINING.md`](./TRAINING.md).

**Runbook:** [`TRAINING.md`](./TRAINING.md).

## Evaluation

Primary claim is **blind human scoring** on the frozen 100-prompt `card` pack (fact card in context for all systems). Same decoding for every system via `FIXED_EVAL_DECODING` in `scripts/lab_gpt/generation.py`.

| Piece | Where |
| --- | --- |
| Rubric + blind protocol | [`eval/rubric.md`](./eval/rubric.md) |
| Build prompts / generate / score / aggregate | [`eval/README.md`](./eval/README.md) |
| Main figure | Faithfulness vs story quality for B0 / B1 / M2 |
| Error analysis | Omission, contradiction, unconstrained invention, encyclopedia dump, story domination |

Supporting automatic metrics (length, self-perplexity) are secondary only — not a substitute for the rubric.

**Runbook:** [`eval/README.md`](./eval/README.md).

---

## Repo layout

```text
configs/                 # b0_smoke, b0_full, b1_cpt, m2_sft
data/                    # schema, fact cards, SFT pairs, licenses, download docs
eval/                    # prompts, generate, Streamlit scorer, rubric, snapshots
scripts/                 # download_data, train_stage1/2, lab_gpt/, fact-card tools
results/<run_id>/        # local only (gitignored): checkpoint.pt, RUN_CARD.md, metrics
TRAINING.md              # Lane B runbook
CAP5636_W6_Transformer(LLM).ipynb   # lab reference
```

## Reproduce key results

From the repo root, with `.venv` active (`pip install -r requirements.txt`; install a CUDA-matched `torch` first — see `requirements.txt`).

```bash
# Data
python scripts/download_data.py

# Stage 1 (B0), then Stage 2 (B1 + M2) — matched budgets
python scripts/train_stage1.py --config configs/b0_full.yaml --retrain-tokenizer
python scripts/train_stage2.py --config configs/b1_cpt.yaml
python scripts/train_stage2.py --config configs/m2_sft.yaml

# Eval (primary: fact card in context)
python eval/build_eval_prompts.py --condition card --out eval/prompts/eval_prompts.jsonl
python eval/generate_samples.py \
  --system B0=results/b0_full_768/checkpoint.pt \
  --system B1=results/b1_cpt_full_768/checkpoint.pt \
  --system M2=results/m2_sft_full_768/checkpoint.pt \
  --prompts eval/prompts/eval_prompts.jsonl \
  --out eval/generations/run_YYYYMMDD.jsonl
streamlit run eval/app.py
python eval/summarize_scores.py --generations eval/generations/run_YYYYMMDD.jsonl
```

Full step checks (tokenizer warning, context budget, smoke path): [`TRAINING.md`](./TRAINING.md) and [`eval/README.md`](./eval/README.md).

---

## Course deliverables

| Component | Weight | Artifact |
| --- | --- | --- |
| **Final paper** | **60%** | 6–8 pp NeurIPS-style PDF (Webcourses); repo link; named contributions; **AI Tools** section |
| **Code & reproducibility** | **20%** | This repo + deps + the runbooks above |
| **Oral presentation** | **20%** | 15 min (10 + 5), ~10–12 slides, **all members speak** |

**Paper sections to include:** Abstract, Intro, Related Work, Methods, Experiments, Results & Discussion (main figure + error analysis), Conclusion, References, Contributions, AI Tools.

**Hard deadline:** 2026-07-27.

### What we are not claiming

- Open-world truthfulness or “no hallucinations” in general
- Perplexity alone as proof of factual stories
- Reproducing large-LM token counts or multi-GPU training as requirements
- RAG, chat UI, agents, or RL alignment as core deliverables
- Real educational product readiness or child-safety guarantees

### Use of generative AI

Allowed for coding, debugging, and writing support. **Must** document tool + purpose in the paper **AI Tools** section.

---

## Status

| Area | State |
| --- | --- |
| Project type / task | Locked: controlled empirical study; B0 / B1 / M2 |
| Data (Lane A) | Done — 866 train / 235 eval / 866 SFT; schema frozen |
| Training code (Lane B) | Done — configs + Stage 1/2 scripts; see `TRAINING.md` |
| Eval harness (Lane C) | Done — prompt builder, generate, Streamlit scorer, summarize |
| Human scoring | In progress — 60 / 100 primary (`card`) prompts scored |
| Paper PDF / slides | In progress |
| Lane owners → Contributions | Assign names before paper freeze |

Work split (assign before the Contributions section is final):

| Lane | Owns |
| --- | --- |
| **A — Data** | Fact cards, SFT pairs, corpora download, licenses |
| **B — Train** | Model/configs, Stage 1 + 2, checkpoints, run cards |
| **C — Eval / paper** | Rubric, scoring, figures, paper/slides, README repro |

---

## Data corpora & licenses

Local download/packaging: [`data/README.md`](./data/README.md) · rebuild: `python scripts/download_data.py`

| Corpus | Role | License (declared) |
| --- | --- | --- |
| **TinyStories** (`roneneldan/TinyStories`) | Stage 1 / B0 | **[CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/)** |
| **Simple English Wikipedia** (`wikimedia/wikipedia`, `20231101.simple`) | Stage 2 B1 CPT | **CC BY-SA** + **GFDL** |

**Full obligations and paper copy-paste block:** [`data/LICENSES.md`](./data/LICENSES.md).

## Relation to the Week 6 LLM lab

Lab notebook: [`CAP5636_W6_Transformer(LLM).ipynb`](./CAP5636_W6_Transformer%28LLM%29.ipynb)

| Lab module | Project use |
| --- | --- |
| Decoder-only GPT + BPE + TinyStories pretrain | Stage 1 architecture / tokenizer / loop |
| Temperature / top-k / top-p | Eval decoding (**fixed across systems**) |
| Adaptation after pretraining | Stage 2 (CPT and/or SFT) |

## References

1. CAP 5636 Final Project Guidelines (Webcourses)
2. CAP 5636 Week 6 LLM lab notebook (this repo)
3. Eldan & Li, *TinyStories* ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759)); [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) under [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/)
4. Wikimedia Wikipedia dumps; [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (CC BY-SA + GFDL)
5. [karpathy/nanochat](https://github.com/karpathy/nanochat)
6. Vaswani et al., *Attention Is All You Need*
7. Gunasekar et al., *Textbooks Are All You Need*
8. Continued pretraining, instruction tuning, and factuality evaluation literature (expand in paper Related Work)
