"""Offline eval generation (Lane C).

Loads one or more system checkpoints, generates a story for every prompt in
the eval prompt set using the project's FIXED_EVAL_DECODING (so B0/B1/M2 are
compared under identical decoding), scores each story's perplexity under its
own generating model, and writes one JSONL file the scoring UI (eval/app.py)
reads.

Usage:
    python eval/generate_samples.py \
        --system B0=results/b0_full/checkpoint.pt \
        --system B1=results/b1_cpt/checkpoint.pt \
        --system M2=results/m2_sft_full/checkpoint.pt \
        --prompts eval/prompts/eval_prompts.jsonl \
        --out eval/generations/run_$(date +%Y%m%d).jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from scripts.lab_gpt.generation import FIXED_EVAL_DECODING, generate
from scripts.lab_gpt.model import GPT, GPTConfig, IGNORE_INDEX, build_model
from scripts.lab_gpt.tokenizer_utils import load_tokenizer


def parse_system_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--system must be NAME=path/to/checkpoint.pt, got {raw!r}")
    name, path = raw.split("=", 1)
    return name, Path(path)


def load_system(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = GPTConfig(**ckpt["config"])
    model = build_model(cfg, pos_encoding=ckpt.get("pos_encoding", "learned")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tokenizer = load_tokenizer(Path(ckpt["tokenizer_dir"]))
    return model, tokenizer


@torch.no_grad()
def story_perplexity(model: GPT, tokenizer, prompt: str, story_full_text: str, device: str) -> float:
    """Perplexity of the generated continuation only, under its own model."""
    prompt_ids = tokenizer.encode(prompt).ids
    full_ids = tokenizer.encode(story_full_text).ids
    if len(full_ids) <= len(prompt_ids):
        return float("nan")

    full_ids = full_ids[: model.config.block_size]
    idx = torch.tensor([full_ids], dtype=torch.long, device=device)

    targets = list(full_ids[1:]) + [IGNORE_INDEX]
    n_prompt = min(len(prompt_ids), len(targets))
    for i in range(n_prompt):
        targets[i] = IGNORE_INDEX
    targets_t = torch.tensor([targets], dtype=torch.long, device=device)

    _, loss = model(idx, targets=targets_t)
    return math.exp(loss.item())


def iter_prompts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", type=parse_system_arg, required=True, dest="systems",
                     help="NAME=path/to/checkpoint.pt, repeatable (e.g. --system B0=... --system M2=...)")
    ap.add_argument("--prompts", type=Path, default=REPO_ROOT / "eval" / "prompts" / "eval_prompts.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    prompts = list(iter_prompts(args.prompts))
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for name, ckpt_path in args.systems:
        print(f"[{name}] loading {ckpt_path} ...")
        model, tokenizer = load_system(ckpt_path, args.device)

        for item in prompts:
            # generate() returns the full decoded sequence (prompt + continuation).
            full_text = generate(model, tokenizer, item["prompt"], FIXED_EVAL_DECODING, args.device)
            story = full_text[len(item["prompt"]):] if full_text.startswith(item["prompt"]) else full_text
            ppl = story_perplexity(model, tokenizer, item["prompt"], full_text, args.device)
            rows.append({
                "prompt_id": item["id"],
                "prompt_text": item["prompt"],
                "system_id": name,
                "story_text": story.strip(),
                "perplexity": ppl,
                "num_tokens": len(tokenizer.encode(story).ids),
                "decoding": {
                    "temperature": FIXED_EVAL_DECODING.temperature,
                    "top_p": FIXED_EVAL_DECODING.top_p,
                    "max_new_tokens": FIXED_EVAL_DECODING.max_new_tokens,
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"  {item['id']}: ppl={ppl:.2f}")

    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
