"""
Load and inspect training checkpoints saved under checkpoints/.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import gymnasium as gym
import minigrid  # noqa: F401
import torch

from models.baseline_agent import Vocabulary
from utils.env_parser import parse_env_description
from utils.rule_based_planner import RuleBasedPlanner


def load_checkpoint(
    path: str,
    device: torch.device,
    *,
    require_vocab: bool = False,
) -> dict[str, Any]:
    """Load a ``.pt`` checkpoint and validate required fields."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint {path} is missing 'model_state_dict'. "
            "Expected a file from train_lgrl.py, train_lgrl_rule.py, or train_baseline.py."
        )
    if require_vocab and not ckpt.get("vocab"):
        raise KeyError(
            f"Checkpoint {path} is missing 'vocab' (word2idx). "
            "Pass --vocab-checkpoint or rely on automatic vocab rebuild."
        )
    return ckpt


def load_vocab_from_checkpoint(ckpt: dict[str, Any]) -> Vocabulary:
    if not ckpt.get("vocab"):
        raise KeyError("Checkpoint has no 'vocab' field.")
    return Vocabulary.load_from_dict(ckpt["vocab"])


def rebuild_vocab_for_eval(
    env_ids: list[str],
    *,
    agent: str = "lgrl",
    max_mission_len: int = 32,
    num_seeds: int = 512,
) -> Vocabulary:
    """Rebuild a vocabulary compatible with experiment / training scripts.

    Experiment runners (``run_experiment1.py`` etc.) bootstrap vocab with one
    ``tokenize`` call then grow it during rollouts. Checkpoints from those
    scripts historically omitted ``vocab``; this function replays the same
    bootstrap and walks rule-planner subgoals over many env seeds so token
    indices stay close to training.
    """
    if not env_ids:
        raise ValueError("rebuild_vocab_for_eval requires at least one env_id")

    vocab = Vocabulary()
    max_len = max_mission_len if agent == "lgrl" else 16
    planner = RuleBasedPlanner()
    bootstrapped = False
    seen_envs: set[str] = set()

    for env_id in env_ids:
        if env_id in seen_envs:
            continue
        seen_envs.add(env_id)
        env = gym.make(env_id)
        try:
            for seed in range(num_seeds):
                obs, _ = env.reset(seed=seed)
                mission = obs["mission"]
                uw = env.unwrapped

                if not bootstrapped:
                    if agent == "lgrl":
                        vocab.tokenize(
                            f"{mission} [SEP] search for the yellow key",
                            max_len=max_len,
                        )
                    else:
                        vocab.tokenize(mission, max_len=max_len)
                    bootstrapped = True

                if agent == "lgrl":
                    env_json = parse_env_description(obs["image"], uw.carrying)
                    n_stages = RuleBasedPlanner.num_stages(mission)
                    for stage in range(n_stages):
                        subgoal, _ = planner.get_subgoal(
                            mission, env_json, obs.get("direction", 0), stage,
                        )
                        vocab.tokenize(
                            f"{mission} [SEP] {subgoal}", max_len=max_len,
                        )
                    for fallback in (
                        "search for the target",
                        "search for the key",
                    ):
                        vocab.tokenize(
                            f"{mission} [SEP] {fallback}", max_len=max_len,
                        )
                else:
                    vocab.tokenize(mission, max_len=max_len)
        finally:
            env.close()

    return vocab


def resolve_vocab(
    ckpt: dict[str, Any],
    *,
    env_ids: list[str],
    agent: str,
    device: torch.device,
    vocab_checkpoint_path: Optional[str] = None,
) -> tuple[Vocabulary, str]:
    """Load or reconstruct vocabulary for evaluation."""
    if vocab_checkpoint_path:
        vckpt = load_checkpoint(
            vocab_checkpoint_path, device, require_vocab=True,
        )
        return (
            load_vocab_from_checkpoint(vckpt),
            f"loaded from --vocab-checkpoint ({vocab_checkpoint_path})",
        )

    if ckpt.get("vocab"):
        return load_vocab_from_checkpoint(ckpt), "loaded from checkpoint"

    vocab = rebuild_vocab_for_eval(env_ids, agent=agent)
    return (
        vocab,
        "rebuilt automatically (checkpoint has no vocab; see run_experiment*.py)",
    )


def describe_checkpoint(ckpt: dict[str, Any], path: str) -> str:
    """Human-readable summary of checkpoint metadata."""
    vocab = ckpt.get("vocab")
    lines = [
        f"  Path          : {path}",
        f"  Train env     : {ckpt.get('env', '(not recorded)')}",
        f"  Train mix     : {ckpt.get('mix', '(none)')}",
        f"  Train planner : {ckpt.get('planner', '(not recorded)')}",
        f"  Updates       : {ckpt.get('update', '?')}",
        f"  Total frames  : {ckpt.get('total_frames', '?')}",
        f"  Vocab size    : {len(vocab) if vocab else '(not saved)'}",
        f"  Has optimizer : {'optimizer_state_dict' in ckpt}",
    ]
    return "\n".join(lines)


def experiment_checkpoint_payload(
    model,
    vocab: Vocabulary,
    update: int,
    total_frames: int,
    *,
    env: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Standard fields for ``run_experiment*.py`` checkpoint saves."""
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab.word2idx,
        "update": update,
        "total_frames": total_frames,
    }
    if env is not None:
        payload["env"] = env
    if extra:
        payload.update(extra)
    return payload
