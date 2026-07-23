"""Byte-Level BPE tokenizer training/loading, shared by Stage 1 and Stage 2.

Stage 2 (B1/M2) must reuse the exact tokenizer trained for Stage 1 (B0) --
vocab and ids are baked into the resumed checkpoint's embedding table.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterable, Optional

from tokenizers import ByteLevelBPETokenizer

SPECIAL_TOKENS = ["<pad>", "<eos>", "<unk>"]


def iter_jsonl_field(path: Path, field: str = "text", limit: Optional[int] = None) -> Iterable[str]:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(field)
            if text:
                yield text
            n += 1
            if limit is not None and n >= limit:
                return


def train_tokenizer(
    data_path: Path,
    out_dir: Path,
    vocab_size: int = 8_000,
    min_frequency: int = 2,
    text_field: str = "text",
    max_docs: Optional[int] = 20_000,
) -> ByteLevelBPETokenizer:
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = list(iter_jsonl_field(data_path, field=text_field, limit=max_docs))
    if not texts:
        raise ValueError(f"No '{text_field}' rows found in {data_path}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(texts))
        corpus_path = Path(f.name)

    try:
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[str(corpus_path)],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
    finally:
        corpus_path.unlink(missing_ok=True)

    tokenizer.save_model(str(out_dir))
    return tokenizer


def load_tokenizer(tok_dir: Path) -> ByteLevelBPETokenizer:
    vocab_path = tok_dir / "vocab.json"
    merges_path = tok_dir / "merges.txt"
    if not (vocab_path.exists() and merges_path.exists()):
        raise FileNotFoundError(f"Tokenizer files not found in {tok_dir}")
    return ByteLevelBPETokenizer(str(vocab_path), str(merges_path))


def load_or_train_tokenizer(
    tok_dir: Path,
    data_path: Path,
    vocab_size: int = 8_000,
    text_field: str = "text",
    max_docs: Optional[int] = 20_000,
    force_retrain: bool = False,
) -> ByteLevelBPETokenizer:
    if not force_retrain and (tok_dir / "vocab.json").exists() and (tok_dir / "merges.txt").exists():
        return load_tokenizer(tok_dir)
    return train_tokenizer(data_path, tok_dir, vocab_size=vocab_size, text_field=text_field, max_docs=max_docs)
