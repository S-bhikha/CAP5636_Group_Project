#!/usr/bin/env python3
"""
Stage 1 pretraining (B0): next-token LM training on TinyStories, from scratch.

Examples
--------
  # Smoke run (fast sanity check; needs data/raw/tinystories/train_smoke.jsonl)
  python scripts/train_stage1.py --config configs/b0_smoke.yaml

  # Full run
  python scripts/train_stage1.py --config configs/b0_full.yaml

Outputs land under results/<run_id>/: config.yaml, metrics.json, RUN_CARD.md,
samples/, checkpoint.pt. checkpoint.pt (plus its recorded tokenizer_dir) is the
required --init-ckpt input for scripts/train_stage2.py (B1 CPT / M2 SFT).

Data must already exist locally -- see data/ENV_SETUP.md / scripts/download_data.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from lab_gpt.data import PackedTextDataset
from lab_gpt.generation import DecodingConfig, generate
from lab_gpt.model import GPTConfig, build_model
from lab_gpt.run_card import write_run_card
from lab_gpt.tokenizer_utils import load_or_train_tokenizer
from lab_gpt.trainer import build_optimizer_and_scheduler, eval_loss, run_training

DEFAULT_TOKENIZER_DIR = REPO_ROOT / "bpe_tokenizer"
SMOKE_DATA = REPO_ROOT / "data" / "raw" / "tinystories" / "train_smoke.jsonl"
FULL_DATA = REPO_ROOT / "data" / "raw" / "tinystories" / "train.jsonl"
SMOKE_VAL = REPO_ROOT / "data" / "raw" / "tinystories" / "validation_smoke.jsonl"
FULL_VAL = REPO_ROOT / "data" / "raw" / "tinystories" / "validation.jsonl"
RESULTS_ROOT = REPO_ROOT / "results"

SAMPLE_PROMPTS = ["Once upon a time", "One sunny day, a"]


def _load_file_defaults(argv: Optional[List[str]]) -> Dict[str, Any]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    ns, _ = pre.parse_known_args(argv)
    if ns.config is None:
        return {}
    return yaml.safe_load(ns.config.read_text(encoding="utf-8")) or {}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    d = _load_file_defaults(argv)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="YAML file providing defaults for any flag below")
    p.add_argument("--smoke", action="store_true", help="Use smoke TinyStories files as the --data/--val-data default")
    p.add_argument("--run-id", type=str, default=d.get("run_id"), help="Defaults to b0_<smoke|full>_<timestamp>")
    p.add_argument("--data", type=Path, default=Path(d["data"]) if "data" in d else None, help="TinyStories jsonl (one 'text' field per line)")
    p.add_argument("--val-data", type=Path, default=Path(d["val_data"]) if "val_data" in d else None)
    p.add_argument("--tokenizer-dir", type=Path, default=Path(d.get("tokenizer_dir", DEFAULT_TOKENIZER_DIR)))
    p.add_argument("--retrain-tokenizer", action="store_true", help="Retrain even if --tokenizer-dir already has a saved BPE model")
    p.add_argument("--tokenizer-train-docs", type=int, default=d.get("tokenizer_train_docs", 20_000))
    p.add_argument("--vocab-size", type=int, default=d.get("vocab_size", 8_000))
    p.add_argument("--block-size", type=int, default=d.get("block_size", 256))
    p.add_argument("--n-layer", type=int, default=d.get("n_layer", 6))
    p.add_argument("--n-head", type=int, default=d.get("n_head", 8))
    p.add_argument("--n-embd", type=int, default=d.get("n_embd", 256))
    p.add_argument("--dropout", type=float, default=d.get("dropout", 0.10))
    p.add_argument("--pos-encoding", choices=["learned", "sinusoidal"], default=d.get("pos_encoding", "learned"))
    p.add_argument("--batch-size", type=int, default=d.get("batch_size", 32))
    p.add_argument("--max-steps", type=int, default=d.get("max_steps", 2_000))
    p.add_argument("--lr", type=float, default=d.get("lr", 3e-4))
    p.add_argument("--warmup-steps", type=int, default=d.get("warmup_steps", 100))
    p.add_argument("--weight-decay", type=float, default=d.get("weight_decay", 0.1))
    p.add_argument("--max-tokenize-tokens", type=int, default=d.get("max_tokenize_tokens"), help="Cap tokens read from --data (None = whole file)")
    p.add_argument("--eval-every", type=int, default=d.get("eval_every", 100))
    p.add_argument("--gen-every", type=int, default=d.get("gen_every", 500))
    p.add_argument("--ckpt-every", type=int, default=d.get("ckpt_every", 0), help="Save an intermediate checkpoint every N steps (0 = final only)")
    p.add_argument("--seed", type=int, default=d.get("seed", 0))
    p.add_argument("--device", type=str, default=d.get("device"), help="Defaults to cuda if available else cpu")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    if args.data is None:
        args.data = SMOKE_DATA if args.smoke else FULL_DATA
    if args.val_data is None:
        candidate = SMOKE_VAL if args.smoke else FULL_VAL
        args.val_data = candidate if candidate.exists() else None

    if not args.data.exists():
        print(f"[error] data file not found: {args.data}\nRun scripts/download_data.py first (see data/ENV_SETUP.md).", file=sys.stderr)
        return 1

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device      : {device}")
    print(f"Data        : {args.data}")

    run_id = args.run_id or f"b0_{'smoke' if args.smoke else 'full'}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = RESULTS_ROOT / run_id
    print(f"Run id      : {run_id}")
    print(f"Out dir     : {out_dir}")

    print(f"Tokenizer   : {args.tokenizer_dir}")
    tokenizer = load_or_train_tokenizer(
        args.tokenizer_dir,
        args.data,
        vocab_size=args.vocab_size,
        max_docs=args.tokenizer_train_docs,
        force_retrain=args.retrain_tokenizer,
    )
    vocab_size = tokenizer.get_vocab_size()

    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = build_model(cfg, pos_encoding=args.pos_encoding).to(device)
    print(f"Model params: {model.n_params():,}")

    dataset = PackedTextDataset(args.data, tokenizer, block_size=cfg.block_size, max_tokens=args.max_tokenize_tokens)
    print(f"Train tokens: {dataset.n_tokens:,} -> {len(dataset):,} sequences [{dataset.elapsed_s:.1f}s]")

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr=args.lr, weight_decay=args.weight_decay, warmup_steps=args.warmup_steps, max_steps=args.max_steps,
    )

    def _save_ckpt(step: int) -> None:
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "pos_encoding": args.pos_encoding,
                "tokenizer_dir": str(args.tokenizer_dir),
                "step": step,
                "system_id": "B0",
            },
            ckpt_dir / f"step_{step}.pt",
        )

    history = run_training(
        model, dataset, optimizer, scheduler, device,
        max_steps=args.max_steps, batch_size=args.batch_size,
        eval_every=args.eval_every, gen_every=args.gen_every,
        gen_prompts=SAMPLE_PROMPTS, tokenizer=tokenizer,
        ckpt_every=args.ckpt_every, ckpt_fn=_save_ckpt if args.ckpt_every else None,
    )

    val_loss = None
    if args.val_data is not None and args.val_data.exists():
        val_dataset = PackedTextDataset(args.val_data, tokenizer, block_size=cfg.block_size, max_tokens=args.max_tokenize_tokens)
        val_loss = eval_loss(model, val_dataset, device, batch_size=args.batch_size)
        print(f"Held-out TinyStories val loss: {val_loss:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "pos_encoding": args.pos_encoding,
            "tokenizer_dir": str(args.tokenizer_dir),
            "step": args.max_steps,
            "system_id": "B0",
        },
        out_dir / "checkpoint.pt",
    )

    samples = [
        generate(model, tokenizer, prompt, DecodingConfig(temperature=0.85, top_p=0.9, max_new_tokens=150), device)
        for prompt in SAMPLE_PROMPTS
    ]

    tokens_seen = args.max_steps * args.batch_size * cfg.block_size
    metrics = {
        "stage": 1,
        "system_id": "B0",
        "final_step": args.max_steps,
        "final_loss": history["loss"][-1] if history["loss"] else None,
        "final_ppl": history["ppl"][-1] if history["ppl"] else None,
        "tokens_seen": tokens_seen,
        "elapsed_s": history["elapsed_s"],
        "val_loss": val_loss,
        "history": {"step": history["step"], "loss": history["loss"], "ppl": history["ppl"]},
    }

    config_dump = {
        "stage": 1,
        "system_id": "B0",
        "run_id": run_id,
        "data": str(args.data),
        "val_data": str(args.val_data) if args.val_data else None,
        "tokenizer_dir": str(args.tokenizer_dir),
        "vocab_size": vocab_size,
        "block_size": cfg.block_size,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "dropout": cfg.dropout,
        "pos_encoding": args.pos_encoding,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }

    write_run_card(
        out_dir, run_id=run_id, stage="1-pretrain", system_id="B0",
        config=config_dump, metrics=metrics, samples=samples,
        notes=f"Checkpoint at results/{run_id}/checkpoint.pt is the --init-ckpt for Stage 2 (B1/M2).",
    )

    print(f"\nDone. Checkpoint: {out_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
