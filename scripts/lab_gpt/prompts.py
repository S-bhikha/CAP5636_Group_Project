"""Fact-card -> model-input rendering (Lane B side of the SCHEMA.md contract).

`render_model_input` must stay identical to whatever Lane C's eval harness
uses, so every system (B0/B1/M2) is scored on matched prompts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATES_PATH = REPO_ROOT / "data" / "prompts" / "templates.json"


def load_templates(path: Path = DEFAULT_TEMPLATES_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_model_input(card: Dict[str, Any], templates: Dict[str, Any]) -> str:
    """Topic + numbered facts + frozen instruction -> the SFT/eval input string."""
    numbered_facts = "\n".join(f"{i + 1}. {fact}" for i, fact in enumerate(card["facts"]))
    return templates["model_input"]["template"].format(
        topic=card["topic"],
        numbered_facts=numbered_facts,
        instruction=templates["instruction"],
    )
