# Training (Lane B)

Model/config, Stage 1 + Stage 2 training scripts, smoke + full runs, checkpoints, run cards.

Owns: `scripts/lab_gpt/` (model, tokenizer, datasets, decoding, trainer, run cards), `scripts/train_stage1.py`, `scripts/train_stage2.py`, `configs/*.yaml`, everything under `results/<run_id>/`.

Architecture and training loop are a direct port of `CAP5636_W6_Transformer(LLM).ipynb` (Modules 1, 2, 3, 4) into standalone scripts -- see that notebook for the pedagogical walkthrough of each component.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
# Install torch first, matched to your GPU/CUDA (see requirements.txt comment), then:
pip install -r requirements.txt
```

## 0. Data

Stage 1 and Stage 2 need local data from Lane A's pipeline first:

```bash
python scripts/download_data.py --smoke-only   # fast: enough for smoke runs
python scripts/download_data.py                # full TinyStories + Simple English Wikipedia
```

See [`data/README.md`](./data/README.md) and [`data/ENV_SETUP.md`](./data/ENV_SETUP.md).

## 1. Stage 1 -- pretrain from scratch (B0)

```bash
# Smoke: sanity-check the whole pipeline in minutes
python scripts/train_stage1.py --config configs/b0_smoke.yaml

# Full pretrain
python scripts/train_stage1.py --config configs/b0_full.yaml
```

Trains (or reuses) a Byte-Level BPE tokenizer at `bpe_tokenizer/` (shared by every later stage), then pretrains the lab-scale GPT (`n_layer=6, n_embd=256, n_head=8, vocab_size=8000, block_size=256` by default -- override via CLI flags or the YAML config; see `python scripts/train_stage1.py --help`).

Output: `results/<run_id>/` containing `config.yaml`, `metrics.json`, `RUN_CARD.md` (hardware, tokens, wall time), `samples/`, and `checkpoint.pt`. That `checkpoint.pt` is the required `--init-ckpt` for Stage 2.

## 2. Stage 2 -- adaptation from the B0 checkpoint

Two arms, same script, matched Stage-2 token budget (`configs/b1_cpt.yaml` and `configs/m2_sft.yaml` share `batch_size`/`max_steps` on purpose -- **keep them in sync if you edit either**):

```bash
# B1: Wikipedia continued pretraining (required baseline)
python scripts/train_stage2.py --mode cpt --init-ckpt results/<b0_run_id>/checkpoint.pt \
  --config configs/b1_cpt.yaml

# M2: SFT on fact-card -> story pairs (primary task adaptation)
python scripts/train_stage2.py --mode sft --init-ckpt results/<b0_run_id>/checkpoint.pt \
  --config configs/m2_sft.yaml
```

Before running, edit `init_ckpt:` in both `configs/b1_cpt.yaml` and `configs/m2_sft.yaml` to point at your actual Stage-1 run directory.

M2 loads `data/fact_cards/train.jsonl` + `data/sft_pairs/train.jsonl` (approved, `split: train` only -- see `data/SCHEMA.md`), renders each card with the same `render_model_input` function Lane C's eval harness should reuse (`scripts/lab_gpt/prompts.py`), and masks the loss over the prompt so only the story tokens are supervised. Right now there is only 1 approved SFT pair and 2 approved train fact cards -- enough to smoke-test the M2 path end to end, but Lane A needs to land more pairs before a real M2 run is meaningful.

B1 packs raw Wikipedia text the same way Stage 1 packs TinyStories (no masking -- every token is a training target).

Each run again writes `results/<run_id>/{config.yaml, metrics.json, RUN_CARD.md, samples/, checkpoint.pt}`. `RUN_CARD.md` prints the realized token budget (`max_steps * batch_size * block_size`) so mismatched B1/M2 budgets are easy to catch before reporting results.

## Fixed decoding

`scripts/lab_gpt/generation.py` exposes `FIXED_EVAL_DECODING` (temperature 0.85, top-p 0.9). Lane C's eval harness should import this rather than redefining decoding settings, so B0/B1/M2 are compared under identical decoding (README requirement).

## Smoke-testing the M2 path early

Per the project's parallel-work plan, M2 training-code work can start on toy pairs during Stage 1, ahead of Lane A's full SFT set:

```bash
python scripts/train_stage2.py --mode sft --init-ckpt results/<b0_smoke_run_id>/checkpoint.pt \
  --fact-cards data/fact_cards/train.jsonl --sft-pairs data/sft_pairs/train.jsonl \
  --max-steps 20 --batch-size 1 --eval-every 5 --gen-every 10
```

## Notes

- `bpe_tokenizer/`, `results/`, and `*.pt` are gitignored -- regenerate locally, don't commit them.
- `--ckpt-every N` saves intermediate checkpoints under `results/<run_id>/checkpoints/step_<N>.pt` for a cheap duration/data ablation later.
- Checkpoints store `config` as a plain dict (not a pickled dataclass) so they load safely under `torch.load(..., weights_only=True)`.
