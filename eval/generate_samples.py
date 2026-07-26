"""Offline eval generation (Lane C).

Loads one or more system checkpoints, generates a story for every prompt in the
eval prompt set using the project's FIXED_EVAL_DECODING (so B0/B1/M2 are
compared under identical decoding), scores each story's perplexity, and writes
one JSONL file the scoring UI (eval/app.py) reads.

The prompt pack must come from eval/build_eval_prompts.py, which renders Lane
A's held-out fact cards with the same `render_model_input` used for M2's SFT
inputs. Do not hand-write bare prompts here: M2 is trained on the rendered
"Topic / Facts / Instruction" format, so a bare sentence tests it
off-distribution and understates the primary system.

Usage:
    python eval/generate_samples.py \
        --system B0=results/b0_full_768/checkpoint.pt \
        --system B1=results/b1_cpt_full_768/checkpoint.pt \
        --system M2=results/m2_sft_full_768/checkpoint.pt \
        --prompts eval/prompts/eval_prompts.jsonl \
        --out eval/generations/run_$(date +%Y%m%d).jsonl

    # prompt ablation (same cards, no card in context) -- keep it in its own file
    python eval/generate_samples.py --system ... \
        --prompts eval/prompts/eval_prompts_nocard.jsonl \
        --out eval/generations/run_$(date +%Y%m%d)_nocard.jsonl
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

from scripts.lab_gpt.generation import FIXED_EVAL_DECODING, generate_ids
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


def check_context_budget(
    model: GPT, tokenizer, prompts: List[Dict[str, Any]], system: str, allow_overflow: bool
) -> None:
    """Fail loudly if a rendered prompt + the generation budget exceeds block_size.

    generate() keeps only the last block_size tokens of context, so an overlong
    prompt does not crash -- the fact card silently scrolls out of the window
    mid-story, and the model is judged on facts it can no longer see.
    """
    budget = model.config.block_size - FIXED_EVAL_DECODING.max_new_tokens
    lengths = [(p["id"], len(tokenizer.encode(p["prompt"]).ids)) for p in prompts]
    over = [(pid, n) for pid, n in lengths if n > budget]
    if not over:
        return
    worst = max(n for _, n in over)
    msg = (
        f"[{system}] {len(over)}/{len(prompts)} prompts do not fit the context window: "
        f"block_size={model.config.block_size} - max_new_tokens={FIXED_EVAL_DECODING.max_new_tokens} "
        f"leaves {budget} prompt tokens, but the longest prompt is {worst} tokens "
        f"(e.g. {', '.join(pid for pid, _ in over[:5])}). "
        "Retrain with a larger block_size, or shorten the prompt template."
    )
    if not allow_overflow:
        raise SystemExit("[error] " + msg + " Pass --allow-context-overflow to generate anyway.")
    print("[warn] " + msg)


@torch.no_grad()
def story_perplexity(model: GPT, tokenizer, all_ids: List[int], n_prompt: int, device: str) -> float:
    """Perplexity of the generated continuation only, under its own model.

    Takes the exact ids returned by generate_ids() so the prompt/continuation
    boundary used for masking is the same one used to slice the story text.

    NOTE (known limitation): this is each sample's perplexity under the model
    that produced it, which rewards low-entropy degeneration -- a system that
    loops "They are happy. They are happy." scores well. Report it as a
    fluency/confidence proxy only; see eval/rubric.md.
    """
    if len(all_ids) <= n_prompt:
        return float("nan")

    # Scoring is capped at block_size; check_context_budget() keeps us clear of it.
    ids = all_ids[: model.config.block_size]
    if len(all_ids) > len(ids):
        targets = list(all_ids[1 : len(ids) + 1])
    else:
        targets = list(all_ids[1:]) + [IGNORE_INDEX]

    # Position i predicts token i+1, so the first story token is predicted at
    # position n_prompt-1 -- mask only positions that predict prompt tokens.
    for i in range(min(n_prompt - 1, len(targets))):
        targets[i] = IGNORE_INDEX
    if all(t == IGNORE_INDEX for t in targets):
        return float("nan")

    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    targets_t = torch.tensor([targets], dtype=torch.long, device=device)
    _, loss = model(ids_t, targets=targets_t)
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
    ap.add_argument("--allow-context-overflow", action="store_true",
                    help="Generate even if prompts exceed block_size - max_new_tokens (not comparable; debugging only)")
    args = ap.parse_args()

    prompts = list(iter_prompts(args.prompts))
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts}")

    conditions = {p.get("condition", "unspecified") for p in prompts}
    if len(conditions) > 1:
        raise SystemExit(f"Prompt pack mixes conditions {conditions}; generate one condition per run.")
    condition = conditions.pop()
    print(f"Prompts     : {len(prompts)} ({condition}) from {args.prompts}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for name, ckpt_path in args.systems:
        print(f"[{name}] loading {ckpt_path} ...")
        model, tokenizer = load_system(ckpt_path, args.device)
        check_context_budget(model, tokenizer, prompts, name, args.allow_context_overflow)

        for item in prompts:
            all_ids, n_prompt = generate_ids(model, tokenizer, item["prompt"], FIXED_EVAL_DECODING, args.device)
            story_ids = all_ids[n_prompt:]
            story = tokenizer.decode(story_ids)
            ppl = story_perplexity(model, tokenizer, all_ids, n_prompt, args.device)
            rows.append({
                "prompt_id": item["id"],
                "card_id": item.get("card_id", ""),
                "condition": condition,
                "topic": item.get("topic", ""),
                "facts": item.get("facts", []),
                "prompt_text": item["prompt"],
                "system_id": name,
                "story_text": story.strip(),
                "perplexity": ppl,
                "num_tokens": len(story_ids),
                "prompt_tokens": n_prompt,
                "hit_token_cap": len(story_ids) >= FIXED_EVAL_DECODING.max_new_tokens,
                "decoding": {
                    "temperature": FIXED_EVAL_DECODING.temperature,
                    "top_p": FIXED_EVAL_DECODING.top_p,
                    "max_new_tokens": FIXED_EVAL_DECODING.max_new_tokens,
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"  {item['id']}: {n_prompt} prompt tok -> {len(story_ids)} story tok, ppl={ppl:.2f}")

    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    capped = sum(1 for r in rows if r["hit_token_cap"])
    print(f"\nWrote {len(rows)} rows -> {args.out}")
    if capped:
        print(f"[warn] {capped}/{len(rows)} stories hit max_new_tokens (never emitted <eos>)")


if __name__ == "__main__":
    main()
