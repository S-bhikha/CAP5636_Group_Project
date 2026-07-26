"""Datasets for Stage 1 pretraining (B0), Stage 2 CPT (B1), and Stage 2 SFT (M2)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .model import IGNORE_INDEX
from .prompts import render_model_input


class PackedTextDataset(Dataset):
    """Concatenate raw-text jsonl rows into fixed-length next-token windows.

    Used for Stage 1 (TinyStories, B0) and Stage 2 CPT (Simple English
    Wikipedia, B1) -- same packing scheme, different source file.
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer,
        block_size: int,
        text_field: str = "text",
        max_tokens: Optional[int] = None,
    ):
        self.block_size = block_size
        eos_id = tokenizer.token_to_id("<eos>")

        t0 = time.time()
        all_ids: List[int] = []
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(text_field)
                if not text:
                    continue
                all_ids.extend(tokenizer.encode(text).ids)
                all_ids.append(eos_id)
                if max_tokens is not None and len(all_ids) >= max_tokens:
                    break

        n = block_size + 1
        trim = (len(all_ids) // n) * n
        if trim == 0:
            raise ValueError(
                f"Not enough tokens in {data_path} for one window of block_size={block_size} "
                f"(got {len(all_ids)} tokens)"
            )
        self.data = torch.tensor(all_ids[:trim], dtype=torch.long)
        self.n_tokens = len(self.data)
        self.elapsed_s = time.time() - t0

    def __len__(self) -> int:
        return len(self.data) // (self.block_size + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.block_size + 1
        chunk = self.data[idx * n: (idx + 1) * n]
        return chunk[:-1], chunk[1:]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class FactCardSFTDataset(Dataset):
    """(fact card + prompt) -> story pairs for Stage 2 SFT (M2).

    Loss is masked over the rendered prompt (SCHEMA.md "model input"); only
    story tokens plus the trailing <eos> are supervised, matching the lab's
    prompt-masking convention (Module 5.2.1).
    """

    def __init__(
        self,
        sft_pairs_path: Path,
        fact_cards_path: Path,
        templates: Dict[str, Any],
        tokenizer,
        block_size: int,
    ):
        self.block_size = block_size
        pad_id = tokenizer.token_to_id("<pad>")
        eos_id = tokenizer.token_to_id("<eos>")
        seq_len = block_size + 1

        cards_by_id = {
            c["id"]: c
            for c in _load_jsonl(fact_cards_path)
            if c.get("review_status") == "approved" and c.get("split") == "train"
        }

        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        n_skipped_no_card = 0
        n_skipped_no_room = 0
        n_truncated = 0
        worst_len = 0

        for pair in _load_jsonl(sft_pairs_path):
            if pair.get("review_status") != "approved":
                continue
            card = cards_by_id.get(pair.get("card_id"))
            if card is None:
                n_skipped_no_card += 1
                continue

            prompt_ids = tokenizer.encode(render_model_input(card, templates)).ids
            story_ids = tokenizer.encode(pair["story"]).ids + [eos_id]

            if len(prompt_ids) >= seq_len:
                n_skipped_no_room += 1
                continue

            needed = len(prompt_ids) + len(story_ids)
            worst_len = max(worst_len, needed)
            if needed > seq_len:
                n_truncated += 1

            full_ids = (prompt_ids + story_ids)[:seq_len]
            is_target = ([False] * len(prompt_ids) + [True] * len(story_ids))[:seq_len]

            n_pad = seq_len - len(full_ids)
            if n_pad > 0:
                full_ids = full_ids + [pad_id] * n_pad
                is_target = is_target + [False] * n_pad

            label_ids = [tid if keep else IGNORE_INDEX for tid, keep in zip(full_ids, is_target)]

            full_t = torch.tensor(full_ids, dtype=torch.long)
            label_t = torch.tensor(label_ids, dtype=torch.long)
            self.examples.append((full_t[:-1], label_t[1:]))

        if n_skipped_no_card:
            print(f"[sft-data] skipped {n_skipped_no_card} pair(s) with no matching approved train card")
        if n_skipped_no_room:
            print(f"[sft-data] skipped {n_skipped_no_room} pair(s) where the rendered prompt alone exceeds block_size")
        if n_truncated:
            # A truncated story loses its ending and its <eos>, so M2 never learns
            # to stop -- and at eval the fact card scrolls out of the window.
            print(
                f"[sft-data] WARNING: {n_truncated}/{len(self.examples)} example(s) truncated at "
                f"block_size={block_size}; longest prompt+story needs {worst_len} tokens "
                f"(have {seq_len}). Retrain Stage 1 with a larger block_size."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]
