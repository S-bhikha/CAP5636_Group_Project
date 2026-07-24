"""Write results/<run_id>/ artifacts: config.yaml, metrics.json, samples/, RUN_CARD.md.

RUN_CARD.md is the human-readable hardware/tokens/wall-time summary the
README requires per Stage-1/Stage-2 run (Lane B deliverable).
"""
from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml


def gpu_info() -> Dict[str, Any]:
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return {"device": "cuda", "name": p.name, "vram_gb": round(p.total_memory / 1e9, 1)}
    return {"device": "cpu", "name": platform.processor() or "cpu", "vram_gb": None}


def write_run_card(
    out_dir: Path,
    *,
    run_id: str,
    stage: str,
    system_id: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    samples: Optional[List[str]] = None,
    notes: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if samples:
        samples_dir = out_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(samples):
            (samples_dir / f"sample_{i:02d}.txt").write_text(text, encoding="utf-8")

    hw = gpu_info()
    if hw["vram_gb"] is not None:
        hw_desc = f"{hw['name']} ({hw['device']}, {hw['vram_gb']} GB VRAM)"
    else:
        hw_desc = f"{hw['name']} ({hw['device']})"

    tokens_seen = metrics.get("tokens_seen")
    wall_s = metrics.get("elapsed_s")

    lines = [
        f"# Run card: {run_id}",
        "",
        f"- **Stage**: {stage}",
        f"- **System**: {system_id}",
        f"- **Hardware**: {hw_desc}",
        f"- **Steps**: {metrics.get('final_step')}",
        f"- **Tokens seen**: {tokens_seen:,}" if tokens_seen is not None else "- **Tokens seen**: n/a",
        f"- **Wall time**: {wall_s:.0f}s" if wall_s is not None else "- **Wall time**: n/a",
        f"- **Final loss / ppl**: {metrics.get('final_loss')} / {metrics.get('final_ppl')}",
    ]
    if metrics.get("val_loss") is not None:
        lines.append(f"- **Held-out TinyStories val loss**: {metrics['val_loss']}")
    if notes:
        lines += ["", notes]

    (out_dir / "RUN_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
