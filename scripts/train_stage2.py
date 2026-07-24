#!/usr/bin/env python3
"""
Stage 2 adaptation from a Stage-1 (B0) checkpoint.

Two modes, selected with --mode:
  cpt  B1: continued pretraining on Simple English Wikipedia (encyclopedic control)
  sft  M2: supervised fine-tuning on (fact card + prompt) -> story pairs

Pass the SAME --max-steps and --batch-size to the B1 and M2 runs so Stage-2
token budgets stay matched (README requirement) -- configs/b1_cpt.yaml and
configs/m2_sft.yaml are pre-matched; keep them in sync if you edit either.

Examples
--------
  python scripts/train_stage2.py --mode cpt --init-ckpt results/b0_full_.../checkpoint.pt \
      --config configs/b1_cpt.yaml

  python scripts/train_stage2.py --mode sft --init-ckpt results/b0_full_.../checkpoint.pt \
      --config configs/m2_sft.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from lab_gpt.data import FactCardSFTDataset, PackedTextDataset
from lab_gpt.generation import DecodingConfig, generate
from lab_gpt.model import GPTConfig, build_model
from lab_gpt.prompts import load_templates, render_model_input
from lab_gpt.run_card import write_run_card
from lab_gpt.tokenizer_utils import load_tokenizer
from lab_gpt.trainer import build_optimizer_and_scheduler, run_training

SMOKE_WIKI = REPO_ROOT / "data" / "raw" / "wikipedia" / "20231101_simple_smoke.jsonl"
FULL_WIKI = REPO_ROOT / "data" / "raw" / "wikipedia" / "20231101_simple.jsonl"
DEFAULT_FACT_CARDS = REPO_ROOT / "data" / "fact_cards" / "train.jsonl"
DEFAULT_EVAL_CARDS = REPO_ROOT / "data" / "fact_cards" / "eval.jsonl"
DEFAULT_SFT_PAIRS = REPO_ROOT / "data" / "sft_pairs" / "train.jsonl"
DEFAULT_TEMPLATES = REPO_ROOT / "data" / "prompts" / "templates.json"
RESULTS_ROOT = REPO_ROOT / "results"

CPT_SAMPLE_PROMPTS = ["Once upon a time", "One sunny day, a"]


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
    p.add_argument("--mode", choices=["cpt", "sft"], default=d.get("mode"))
    p.add_argument("--init-ckpt", type=Path, default=Path(d["init_ckpt"]) if "init_ckpt" in d else None, help="Stage-1 (B0) checkpoint.pt")
    p.add_argument("--run-id", type=str, default=d.get("run_id"))
    p.add_argument("--system-id", type=str, default=d.get("system_id"), help="Defaults to B1 (cpt) or M2 (sft)")
    p.add_argument("--smoke", action="store_true", help="Use the smoke Wikipedia file as the --data default (cpt mode)")
    p.add_argument("--data", type=Path, default=Path(d["data"]) if "data" in d else None, help="Wikipedia jsonl (cpt mode only)")
    p.add_argument("--fact-cards", type=Path, default=Path(d.get("fact_cards", DEFAULT_FACT_CARDS)))
    p.add_argument("--eval-cards", type=Path, default=Path(d.get("eval_cards", DEFAULT_EVAL_CARDS)))
    p.add_argument("--sft-pairs", type=Path, default=Path(d.get("sft_pairs", DEFAULT_SFT_PAIRS)))
    p.add_argument("--templates", type=Path, default=Path(d.get("templates", DEFAULT_TEMPLATES)))
    p.add_argument("--batch-size", type=int, default=d.get("batch_size", 16))
    p.add_argument("--max-steps", type=int, default=d.get("max_steps", 500))
    p.add_argument("--lr", type=float, default=d.get("lr", 1e-4))
    p.add_argument("--warmup-steps", type=int, default=d.get("warmup_steps", 50))
    p.add_argument("--weight-decay", type=float, default=d.get("weight_decay", 0.1))
    p.add_argument("--max-tokenize-tokens", type=int, default=d.get("max_tokenize_tokens"), help="Cap tokens read from --data (cpt mode; None = whole file)")
    p.add_argument("--eval-every", type=int, default=d.get("eval_every", 50))
    p.add_argument("--gen-every", type=int, default=d.get("gen_every", 200))
    p.add_argument("--ckpt-every", type=int, default=d.get("ckpt_every", 0))
    p.add_argument("--seed", type=int, default=d.get("seed", 0))
    p.add_argument("--device", type=str, default=d.get("device"))
    return p.parse_args(argv)


def _sft_eval_prompts(eval_cards_path: Path, templates: Dict[str, Any], k: int = 2) -> List[str]:
    if not eval_cards_path.exists():
        return []
    prompts = []
    with eval_cards_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            if card.get("review_status") == "approved":
                prompts.append(render_model_input(card, templates))
            if len(prompts) >= k:
                break
    return prompts


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    if args.mode is None:
        print("[error] --mode {cpt,sft} is required (CLI flag or --config yaml)", file=sys.stderr)
        return 1
    if args.init_ckpt is None:
        print("[error] --init-ckpt is required (CLI flag or --config yaml)", file=sys.stderr)
        return 1
    if not args.init_ckpt.exists():
        print(f"[error] init checkpoint not found: {args.init_ckpt}", file=sys.stderr)
        return 1

    if args.mode == "cpt" and args.data is None:
        args.data = SMOKE_WIKI if args.smoke else FULL_WIKI

    system_id = args.system_id or ("B1" if args.mode == "cpt" else "M2")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Device      : {device}")
    print(f"Mode        : {args.mode}  (system_id={system_id})")
    print(f"Init ckpt   : {args.init_ckpt}")

    ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=True)
    cfg = GPTConfig(**ckpt["config"])
    pos_encoding = ckpt.get("pos_encoding", "learned")
    tokenizer_dir = Path(ckpt["tokenizer_dir"])
    tokenizer = load_tokenizer(tokenizer_dir)

    model = build_model(cfg, pos_encoding=pos_encoding).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model params: {model.n_params():,}  (resumed from step {ckpt.get('step')}, system {ckpt.get('system_id')})")

    templates = load_templates(args.templates) if args.mode == "sft" else None

    if args.mode == "cpt":
        if not args.data.exists():
            print(f"[error] data file not found: {args.data}\nRun scripts/download_data.py first.", file=sys.stderr)
            return 1
        dataset = PackedTextDataset(args.data, tokenizer, block_size=cfg.block_size, max_tokens=args.max_tokenize_tokens)
        gen_prompts = CPT_SAMPLE_PROMPTS
    else:
        if not args.sft_pairs.exists() or not args.fact_cards.exists():
            print(f"[error] SFT data not found: {args.sft_pairs} / {args.fact_cards}", file=sys.stderr)
            return 1
        dataset = FactCardSFTDataset(args.sft_pairs, args.fact_cards, templates, tokenizer, block_size=cfg.block_size)
        gen_prompts = _sft_eval_prompts(args.eval_cards, templates) or CPT_SAMPLE_PROMPTS

    print(f"Examples    : {len(dataset):,}")
    if len(dataset) == 0:
        print("[error] dataset is empty -- nothing to train on", file=sys.stderr)
        return 1

    run_id = args.run_id or f"{system_id.lower()}_{'smoke' if args.smoke else 'full'}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = RESULTS_ROOT / run_id
    print(f"Run id      : {run_id}")

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
                "pos_encoding": pos_encoding,
                "tokenizer_dir": str(tokenizer_dir),
                "step": step,
                "system_id": system_id,
                "init_ckpt": str(args.init_ckpt),
            },
            ckpt_dir / f"step_{step}.pt",
        )

    history = run_training(
        model, dataset, optimizer, scheduler, device,
        max_steps=args.max_steps, batch_size=args.batch_size,
        eval_every=args.eval_every, gen_every=args.gen_every,
        gen_prompts=gen_prompts, tokenizer=tokenizer,
        ckpt_every=args.ckpt_every, ckpt_fn=_save_ckpt if args.ckpt_every else None,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "pos_encoding": pos_encoding,
            "tokenizer_dir": str(tokenizer_dir),
            "step": args.max_steps,
            "system_id": system_id,
            "init_ckpt": str(args.init_ckpt),
        },
        out_dir / "checkpoint.pt",
    )

    sample_prompts_final = (gen_prompts or CPT_SAMPLE_PROMPTS)[:3]
    samples = [
        generate(model, tokenizer, prompt, DecodingConfig(temperature=0.85, top_p=0.9, max_new_tokens=150), device)
        for prompt in sample_prompts_final
    ]

    tokens_seen = args.max_steps * args.batch_size * cfg.block_size
    metrics = {
        "stage": 2,
        "system_id": system_id,
        "mode": args.mode,
        "final_step": args.max_steps,
        "final_loss": history["loss"][-1] if history["loss"] else None,
        "final_ppl": history["ppl"][-1] if history["ppl"] else None,
        "tokens_seen": tokens_seen,
        "elapsed_s": history["elapsed_s"],
        "history": {"step": history["step"], "loss": history["loss"], "ppl": history["ppl"]},
    }

    config_dump = {
        "stage": 2,
        "system_id": system_id,
        "mode": args.mode,
        "run_id": run_id,
        "init_ckpt": str(args.init_ckpt),
        "data": str(args.data) if args.mode == "cpt" else None,
        "fact_cards": str(args.fact_cards) if args.mode == "sft" else None,
        "sft_pairs": str(args.sft_pairs) if args.mode == "sft" else None,
        "tokenizer_dir": str(tokenizer_dir),
        "block_size": cfg.block_size,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "dropout": cfg.dropout,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }

    write_run_card(
        out_dir, run_id=run_id, stage=f"2-{args.mode}", system_id=system_id,
        config=config_dump, metrics=metrics, samples=samples,
        notes=(
            f"Matched Stage-2 budget check: max_steps={args.max_steps}, batch_size={args.batch_size}, "
            f"block_size={cfg.block_size} -> tokens_seen={tokens_seen:,}. "
            "Compare against the paired B1/M2 run card before reporting results."
        ),
    )

    print(f"\nDone. Checkpoint: {out_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
