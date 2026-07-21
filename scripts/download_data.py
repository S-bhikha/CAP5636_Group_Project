#!/usr/bin/env python3
"""
Download and materialize local corpora for CAP 5636 Stage 1 / Stage 2 B1.

Sources
-------
* TinyStories  — roneneldan/TinyStories  (Stage 1 pretraining + smoke)
* Wikipedia    — wikimedia/wikipedia     (Stage 2 B1 continued pretraining)

Default Wikipedia subset is Simple English (20231101.simple): ~242k articles,
simpler language, and a better vocabulary match to TinyStories than full enwiki.
Use --wiki-config 20231101.en with a row cap if you need full English instead.

Examples
--------
  # Smoke subsets only (fast; enough for Lane B Stage-1 smoke)
  python scripts/download_data.py --smoke-only

  # Full TinyStories + full Simple English Wikipedia
  python scripts/download_data.py

  # TinyStories full + English Wiki capped at 50k articles
  python scripts/download_data.py --wiki-config 20231101.en --wiki-max-rows 50000

Outputs land under data/raw/ and a JSON manifest under data/manifests/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
MANIFEST_DIR = DATA_ROOT / "manifests"

TINYSTORIES_HF = "roneneldan/TinyStories"
WIKI_HF = "wikimedia/wikipedia"
DEFAULT_WIKI_CONFIG = "20231101.simple"

# License metadata (authoritative human doc: data/LICENSES.md).
# Keep in sync with data/licenses.json when ids change on the Hub.
TINYSTORIES_LICENSE = {
    "spdx": "CDLA-Sharing-1.0",
    "hf_id": "cdla-sharing-1.0",
    "name": "Community Data License Agreement – Sharing, Version 1.0",
    "url": "https://cdla.dev/sharing-1-0/",
    "homepage": f"https://huggingface.co/datasets/{TINYSTORIES_HF}",
    "paper": "https://arxiv.org/abs/2305.07759",
    "attribution": (
        "TinyStories dataset (Eldan & Li, 2023; "
        f"https://huggingface.co/datasets/{TINYSTORIES_HF}), "
        "licensed under CDLA-Sharing-1.0 (https://cdla.dev/sharing-1-0/)."
    ),
    "notes": (
        "Synthetic stories (GPT-3.5/GPT-4). Publishing Data/Enhanced Data requires "
        "CDLA-Sharing share-alike + attribution; Computational Use Results are not "
        "share-alike-locked by §3.5. See data/LICENSES.md."
    ),
}
WIKIPEDIA_LICENSE = {
    "hf_ids": ["cc-by-sa-3.0", "gfdl"],
    "names": [
        "Creative Commons Attribution-ShareAlike (HF card: 3.0; dumps legal also cites 4.0)",
        "GNU Free Documentation License (GFDL)",
    ],
    "urls": {
        "cc_by_sa_3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
        "cc_by_sa_4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
        "gfdl": "https://www.gnu.org/licenses/fdl-1.3.html",
        "dumps_legal": "https://dumps.wikimedia.org/legal.html",
        "hf_dataset": f"https://huggingface.co/datasets/{WIKI_HF}",
        "terms_of_use": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
    },
    "homepage": f"https://huggingface.co/datasets/{WIKI_HF}",
    "attribution": (
        "Wikipedia text from Wikimedia dumps via wikimedia/wikipedia on Hugging Face. "
        "Available under CC BY-SA and GFDL; see https://dumps.wikimedia.org/legal.html. "
        "Attribution: Wikipedia/Wikimedia contributors. Processing changes: HF dump "
        "cleaning; project subset fields id/url/title/text."
    ),
    "notes": (
        "Dual-licensed free content. Attribute + link license + indicate changes when "
        "sharing text; ShareAlike for redistributed adapted text. See data/LICENSES.md."
    ),
}

# Lab notebook smoke size; full train uses entire split when --smoke-only is off.
SMOKE_TS_TRAIN = 10_000
SMOKE_TS_VAL = 1_000
SMOKE_WIKI = 5_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dirs() -> None:
    for p in (
        RAW_ROOT / "tinystories",
        RAW_ROOT / "wikipedia",
        MANIFEST_DIR,
        DATA_ROOT / "fact_cards",
        DATA_ROOT / "sft_pairs",
        DATA_ROOT / "prompts",
        DATA_ROOT / "processed",
    ):
        p.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _file_mb(path: Path) -> float:
    return path.stat().st_size / 1e6 if path.exists() else 0.0


def download_tinystories(
    *,
    smoke_only: bool,
    smoke_train: int,
    smoke_val: int,
    force: bool,
) -> Dict[str, Any]:
    from datasets import load_dataset

    out_dir = RAW_ROOT / "tinystories"
    train_path = out_dir / ("train_smoke.jsonl" if smoke_only else "train.jsonl")
    val_path = out_dir / ("validation_smoke.jsonl" if smoke_only else "validation.jsonl")
    # Always also emit lab-compatible smoke files when doing full download.
    smoke_train_path = out_dir / "train_smoke.jsonl"
    smoke_val_path = out_dir / "validation_smoke.jsonl"

    info: Dict[str, Any] = {
        "source": TINYSTORIES_HF,
        "license": TINYSTORIES_LICENSE,
        "license_doc": "data/LICENSES.md",
        "files": {},
    }

    if train_path.exists() and val_path.exists() and not force:
        print(f"[tinystories] exists, skip (use --force to re-download): {train_path}")
        info["files"]["train"] = {
            "path": str(train_path.relative_to(REPO_ROOT)),
            "rows": _count_lines(train_path),
            "size_mb": round(_file_mb(train_path), 2),
        }
        info["files"]["validation"] = {
            "path": str(val_path.relative_to(REPO_ROOT)),
            "rows": _count_lines(val_path),
            "size_mb": round(_file_mb(val_path), 2),
        }
        return info

    print(f"[tinystories] loading {TINYSTORIES_HF} ...")
    t0 = time.time()

    if smoke_only:
        train_split = f"train[:{smoke_train}]"
        val_split = f"validation[:{smoke_val}]"
    else:
        train_split = "train"
        val_split = "validation"

    # Parquet-backed on Hub; no loading script / trust_remote_code needed (datasets>=3).
    train_ds = load_dataset(TINYSTORIES_HF, split=train_split)
    val_ds = load_dataset(TINYSTORIES_HF, split=val_split)

    def _rows(ds) -> List[Dict[str, Any]]:
        # Dataset is a single "text" column.
        return [{"text": t} for t in ds["text"]]

    print(f"[tinystories] writing {train_path} ({len(train_ds):,} rows)...")
    n_train = _write_jsonl(train_path, _rows(train_ds))
    print(f"[tinystories] writing {val_path} ({len(val_ds):,} rows)...")
    n_val = _write_jsonl(val_path, _rows(val_ds))

    # Lab-sized smoke always available under fixed names.
    if not smoke_only:
        print(f"[tinystories] also writing smoke slices → {smoke_train_path.name}, {smoke_val_path.name}")
        _write_jsonl(
            smoke_train_path,
            [{"text": t} for t in train_ds.select(range(min(smoke_train, len(train_ds))))["text"]],
        )
        _write_jsonl(
            smoke_val_path,
            [{"text": t} for t in val_ds.select(range(min(smoke_val, len(val_ds))))["text"]],
        )
        info["files"]["train_smoke"] = {
            "path": str(smoke_train_path.relative_to(REPO_ROOT)),
            "rows": min(smoke_train, n_train),
            "size_mb": round(_file_mb(smoke_train_path), 2),
        }
        info["files"]["validation_smoke"] = {
            "path": str(smoke_val_path.relative_to(REPO_ROOT)),
            "rows": min(smoke_val, n_val),
            "size_mb": round(_file_mb(smoke_val_path), 2),
        }

    elapsed = time.time() - t0
    print(f"[tinystories] done in {elapsed:.1f}s")

    info["files"]["train"] = {
        "path": str(train_path.relative_to(REPO_ROOT)),
        "rows": n_train,
        "size_mb": round(_file_mb(train_path), 2),
    }
    info["files"]["validation"] = {
        "path": str(val_path.relative_to(REPO_ROOT)),
        "rows": n_val,
        "size_mb": round(_file_mb(val_path), 2),
    }
    info["elapsed_s"] = round(elapsed, 1)
    return info


def download_wikipedia(
    *,
    config: str,
    smoke_only: bool,
    smoke_rows: int,
    max_rows: Optional[int],
    force: bool,
) -> Dict[str, Any]:
    from datasets import load_dataset

    out_dir = RAW_ROOT / "wikipedia"
    # Safe filename from config, e.g. 20231101.simple → 20231101_simple
    tag = config.replace(".", "_")
    if smoke_only:
        out_path = out_dir / f"{tag}_smoke.jsonl"
        n_limit = smoke_rows
    elif max_rows is not None:
        out_path = out_dir / f"{tag}_n{max_rows}.jsonl"
        n_limit = max_rows
    else:
        out_path = out_dir / f"{tag}.jsonl"
        n_limit = None

    info: Dict[str, Any] = {
        "source": WIKI_HF,
        "config": config,
        "license": WIKIPEDIA_LICENSE,
        "license_doc": "data/LICENSES.md",
        "files": {},
    }

    if out_path.exists() and not force:
        print(f"[wikipedia] exists, skip (use --force to re-download): {out_path}")
        info["files"]["articles"] = {
            "path": str(out_path.relative_to(REPO_ROOT)),
            "rows": _count_lines(out_path),
            "size_mb": round(_file_mb(out_path), 2),
            "row_cap": n_limit,
        }
        return info

    print(f"[wikipedia] loading {WIKI_HF} config={config!r} (streaming)...")
    t0 = time.time()

    # Stream so we never require the full multi-GB dump in memory for capped pulls.
    ds = load_dataset(WIKI_HF, config, split="train", streaming=True)

    rows_out: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        # Keep fields useful for CPT packing; drop huge unused metadata later if needed.
        rows_out.append(
            {
                "id": ex.get("id"),
                "url": ex.get("url"),
                "title": ex.get("title"),
                "text": ex.get("text", ""),
            }
        )
        if (i + 1) % 2000 == 0:
            print(f"[wikipedia]   streamed {i + 1:,} articles...")
        if n_limit is not None and (i + 1) >= n_limit:
            break

    print(f"[wikipedia] writing {out_path} ({len(rows_out):,} rows)...")
    n = _write_jsonl(out_path, rows_out)
    elapsed = time.time() - t0
    print(f"[wikipedia] done in {elapsed:.1f}s")

    # Always keep a small smoke file for B1 dry-runs.
    smoke_path = out_dir / f"{tag}_smoke.jsonl"
    if not smoke_only:
        smoke_n = min(smoke_rows, n)
        _write_jsonl(smoke_path, rows_out[:smoke_n])
        info["files"]["smoke"] = {
            "path": str(smoke_path.relative_to(REPO_ROOT)),
            "rows": smoke_n,
            "size_mb": round(_file_mb(smoke_path), 2),
        }

    info["files"]["articles"] = {
        "path": str(out_path.relative_to(REPO_ROOT)),
        "rows": n,
        "size_mb": round(_file_mb(out_path), 2),
        "row_cap": n_limit,
    }
    info["elapsed_s"] = round(elapsed, 1)
    return info


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def write_manifest(payload: Dict[str, Any]) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / "local_corpora.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[manifest] wrote {path.relative_to(REPO_ROOT)}")
    return path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--smoke-only",
        action="store_true",
        help=f"Download small slices only (TS train={SMOKE_TS_TRAIN}, val={SMOKE_TS_VAL}, wiki={SMOKE_WIKI})",
    )
    p.add_argument(
        "--skip-tinystories",
        action="store_true",
        help="Skip TinyStories download",
    )
    p.add_argument(
        "--skip-wikipedia",
        action="store_true",
        help="Skip Wikipedia download",
    )
    p.add_argument(
        "--wiki-config",
        default=DEFAULT_WIKI_CONFIG,
        help=f"wikimedia/wikipedia config name (default: {DEFAULT_WIKI_CONFIG})",
    )
    p.add_argument(
        "--wiki-max-rows",
        type=int,
        default=None,
        help="Cap Wikipedia articles (ignored with --smoke-only). Default: full config.",
    )
    p.add_argument(
        "--smoke-ts-train",
        type=int,
        default=SMOKE_TS_TRAIN,
        help=f"TinyStories smoke train rows (default {SMOKE_TS_TRAIN})",
    )
    p.add_argument(
        "--smoke-ts-val",
        type=int,
        default=SMOKE_TS_VAL,
        help=f"TinyStories smoke validation rows (default {SMOKE_TS_VAL})",
    )
    p.add_argument(
        "--smoke-wiki",
        type=int,
        default=SMOKE_WIKI,
        help=f"Wikipedia smoke rows (default {SMOKE_WIKI})",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output files already exist",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _ensure_dirs()

    print(f"Repo root : {REPO_ROOT}")
    print(f"Data root : {DATA_ROOT}")
    print(f"Mode      : {'SMOKE ONLY' if args.smoke_only else 'FULL (or capped wiki)'}")
    print("-" * 60)

    manifest: Dict[str, Any] = {
        "created_at": _utc_now(),
        "smoke_only": args.smoke_only,
        "license_doc": "data/LICENSES.md",
        "license_metadata": "data/licenses.json",
        "corpora": {},
    }

    if not args.skip_tinystories:
        manifest["corpora"]["tinystories"] = download_tinystories(
            smoke_only=args.smoke_only,
            smoke_train=args.smoke_ts_train,
            smoke_val=args.smoke_ts_val,
            force=args.force,
        )
    else:
        print("[tinystories] skipped")

    if not args.skip_wikipedia:
        manifest["corpora"]["wikipedia"] = download_wikipedia(
            config=args.wiki_config,
            smoke_only=args.smoke_only,
            smoke_rows=args.smoke_wiki,
            max_rows=args.wiki_max_rows,
            force=args.force,
        )
    else:
        print("[wikipedia] skipped")

    write_manifest(manifest)
    print("-" * 60)
    print("Done. See data/README.md for paths and license notes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
