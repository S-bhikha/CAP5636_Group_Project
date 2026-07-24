"""Shared training loop for Stage 1 pretraining and Stage 2 CPT/SFT.

All three (B0, B1, M2) are next-token cross-entropy training over (x, y)
pairs -- label masking for SFT is already baked into the dataset, so one loop
covers every arm.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .generation import DecodingConfig, generate


def build_optimizer_and_scheduler(model, lr: float, weight_decay: float, warmup_steps: int, max_steps: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def run_training(
    model,
    dataset: Dataset,
    optimizer,
    scheduler,
    device,
    max_steps: int,
    batch_size: int,
    eval_every: int = 100,
    gen_every: int = 0,
    gen_prompts: Optional[List[str]] = None,
    tokenizer=None,
    ckpt_every: int = 0,
    ckpt_fn: Optional[Callable[[int], None]] = None,
    log_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    history: Dict[str, Any] = {"step": [], "loss": [], "ppl": []}
    data_iter = iter(loader)
    t0 = time.time()

    model.train()
    log_fn(f"Training {max_steps} steps | batch={batch_size} | examples={len(dataset):,}")

    for step in range(1, max_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_every == 0 or step == 1:
            ppl = math.exp(min(loss.item(), 20))
            cur_lr = optimizer.param_groups[0]["lr"]
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["ppl"].append(ppl)
            log_fn(
                f"step {step:6d}/{max_steps} | loss={loss.item():.4f} | ppl={ppl:8.1f} "
                f"| lr={cur_lr:.2e} | {time.time() - t0:.0f}s"
            )

        if gen_every and step % gen_every == 0 and tokenizer is not None and gen_prompts:
            for prompt in gen_prompts:
                sample = generate(
                    model, tokenizer, prompt,
                    DecodingConfig(temperature=0.85, top_p=0.9, max_new_tokens=60),
                    device,
                )
                log_fn(f"  [sample @ step {step}] {sample[:200]}")

        if ckpt_every and step % ckpt_every == 0 and ckpt_fn is not None:
            ckpt_fn(step)

    elapsed = time.time() - t0
    log_fn(f"Done in {elapsed:.0f}s")
    history["elapsed_s"] = elapsed
    history["final_step"] = max_steps
    return history


@torch.no_grad()
def eval_loss(model, dataset: Dataset, device, batch_size: int = 32, max_batches: int = 50) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    total_loss, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    if was_training:
        model.train()
    return total_loss / max(n, 1)
