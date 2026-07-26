"""Autoregressive decoding shared by training-time sampling and later eval (Lane C).

FIXED_EVAL_DECODING is the frozen decoding config the README requires to stay
the same across B0/B1/M2 when generating the final eval stories -- Lane C's
harness should import it rather than redefining its own settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DecodingConfig:
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    greedy: bool = False
    max_new_tokens: int = 200


# max_new_tokens must exceed the gold story length (~200-260 tokens for the
# 120-180 word target in data/prompts/templates.json), otherwise M2 is cut off
# before it can emit <eos> and every system looks truncated.
FIXED_EVAL_DECODING = DecodingConfig(temperature=0.85, top_p=0.9, max_new_tokens=300)


@torch.no_grad()
def generate_ids(
    model, tokenizer, prompt: str, config: DecodingConfig, device
) -> Tuple[List[int], int]:
    """Sample a continuation; return (prompt_ids + new_ids, n_prompt_tokens).

    Callers that need the prompt/continuation boundary should use this rather
    than string-slicing `generate()`'s output: BPE decode(encode(x)) is not
    guaranteed to reproduce `x` byte-for-byte, and with a long rendered fact
    card a few characters of slippage would silently leak card text into the
    scored story (and into the perplexity mask).
    """
    was_training = model.training
    model.eval()
    eos_id = tokenizer.token_to_id("<eos>")
    prompt_ids = tokenizer.encode(prompt).ids
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(config.max_new_tokens):
        ids_cond = ids[:, -model.config.block_size:]
        logits, _ = model(ids_cond)
        logits = logits[:, -1, :]

        if config.greedy:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / max(config.temperature, 1e-8)
            if config.top_k is not None:
                k = min(config.top_k, logits.size(-1))
                vals, _ = torch.topk(logits, k)
                logits[logits < vals[:, [-1]]] = float("-inf")
            if config.top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > config.top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == eos_id:
            break

    if was_training:
        model.train()
    return ids[0].tolist(), len(prompt_ids)


def generate(model, tokenizer, prompt: str, config: DecodingConfig, device) -> str:
    """Full decoded sequence (prompt + continuation)."""
    all_ids, _ = generate_ids(model, tokenizer, prompt, config, device)
    return tokenizer.decode(all_ids)
