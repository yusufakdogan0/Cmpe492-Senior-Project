"""
Load and inspect training checkpoints saved under checkpoints/.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from models.baseline_agent import Vocabulary


def load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    """Load a ``.pt`` checkpoint and validate required fields."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint {path} is missing 'model_state_dict'. "
            "Expected a file from train_lgrl.py, train_lgrl_rule.py, or train_baseline.py."
        )
    if "vocab" not in ckpt:
        raise KeyError(
            f"Checkpoint {path} is missing 'vocab' (word2idx). "
            "Cannot tokenize missions for evaluation."
        )
    return ckpt


def load_vocab_from_checkpoint(ckpt: dict[str, Any]) -> Vocabulary:
    return Vocabulary.load_from_dict(ckpt["vocab"])


def describe_checkpoint(ckpt: dict[str, Any], path: str) -> str:
    """Human-readable summary of checkpoint metadata."""
    lines = [
        f"  Path          : {path}",
        f"  Train env     : {ckpt.get('env', '(not recorded)')}",
        f"  Train mix     : {ckpt.get('mix', '(none)')}",
        f"  Train planner : {ckpt.get('planner', '(not recorded)')}",
        f"  Updates       : {ckpt.get('update', '?')}",
        f"  Total frames  : {ckpt.get('total_frames', '?')}",
        f"  Vocab size    : {len(ckpt.get('vocab', {}))}",
        f"  Has optimizer : {'optimizer_state_dict' in ckpt}",
    ]
    return "\n".join(lines)
