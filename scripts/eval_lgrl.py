"""
eval_lgrl.py — Evaluate a trained LGRL (or baseline) checkpoint.

Designed for the rule-train / LLM-test workflow: train with
train_lgrl_rule.py, then evaluate with --planner llm (Ollama).

Runs each benchmark environment for N episodes, reports success rate,
average raw environment return, average steps, and mean policy entropy,
and writes per-episode JSONL traces for inspection.

Usage:
    python scripts/eval_lgrl.py --checkpoint checkpoints/lgrl_rule.pt
    python scripts/eval_lgrl.py --checkpoint checkpoints/lgrl_rule.pt \\
        --planner llm --episodes 100 --envs gotodoor,unlockpickup
    python scripts/eval_lgrl.py --checkpoint checkpoints/baseline.pt \\
        --agent baseline --episodes 500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import requests
import torch
import torch_ac

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.baseline_agent import BaselineAgent, Vocabulary
from models.lgrl_agent import LGRLAgent
from utils.checkpoint_utils import describe_checkpoint, load_checkpoint, load_vocab_from_checkpoint
from utils.env_parser import parse_env_description
from utils.eval_config import (
    DEFAULT_NUM_EPISODES,
    DEFAULT_SEED_START,
    SUBGOAL_TIMEOUT_MULT,
    BenchmarkEnvSpec,
    get_eval_suite,
)
from utils.llm_planner import DEFAULT_TIMEOUT_SEC, LLMPlanner
from utils.rule_based_planner import RuleBasedPlanner
from utils.subgoal_tracker import SubgoalTracker

LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "eval")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Single-env hierarchy (eval only; mirrors train_lgrl.HierarchyState)
# ---------------------------------------------------------------------------

class EvalHierarchy:
  def __init__(self, env, planner, n_subgoals: int, t_max: int):
    self.env = env
    self.planner = planner
    self.n_subgoals = n_subgoals
    self.t_max = t_max
    self.tracker = SubgoalTracker()
    self.reset_episode()

  def reset_episode(self):
    self.active_subgoal = ""
    self.stage_index = 0
    self.step_counter = 0
    self.episode_steps = 0
    self.subgoal_trace: list[dict[str, Any]] = []

  def subgoal_budget(self) -> float:
    i = min(self.stage_index + 1, self.n_subgoals)
    return (i / self.n_subgoals) * self.t_max

  def init_subgoal(self, obs: dict):
    uw = self.env.unwrapped
    env_json = parse_env_description(obs["image"], uw.carrying)
    subgoal, stage = self.planner.get_subgoal(
      obs["mission"], env_json, obs.get("direction", 0), stage_index=0,
    )
    self.active_subgoal = subgoal
    self.stage_index = stage
    self.step_counter = 0
    self._log_subgoal_event("init", subgoal, env_json)

  def advance(self, obs: dict, status: str):
    self.subgoal_trace.append({
      "subgoal": self.active_subgoal,
      "status": status,
      "steps": self.step_counter,
      "stage": self.stage_index,
    })
    next_stage = self.stage_index + 1
    if next_stage >= self.n_subgoals:
      self.stage_index = self.n_subgoals
      self.step_counter = 0
      return

    uw = self.env.unwrapped
    env_json = parse_env_description(obs["image"], uw.carrying)
    subgoal, stage = self.planner.get_subgoal(
      obs["mission"], env_json, obs.get("direction", 0), stage_index=next_stage,
    )
    self.active_subgoal = subgoal
    self.stage_index = stage
    self.step_counter = 0
    self._log_subgoal_event("new", subgoal, env_json, after=status)

  def _log_subgoal_event(self, event: str, subgoal: str, env_json: str, after: str = ""):
    entry = {
      "event": event,
      "subgoal": subgoal,
      "stage": self.stage_index,
      "budget": self.subgoal_budget(),
    }
    if after:
      entry["after"] = after
    raw = getattr(self.planner, "last_raw_response", None)
    if raw:
      entry["raw_llm"] = raw
    self.subgoal_trace.append(entry)

  def after_step(self, obs: dict, action: int) -> None:
    self.step_counter += 1
    self.episode_steps += 1

    if self.stage_index >= self.n_subgoals:
      return
    if not self.active_subgoal:
      self.init_subgoal(obs)
      return

    uw = self.env.unwrapped
    completed = self.tracker.check_completion(
      self.active_subgoal, uw, action, obs_image=obs["image"],
    )
    timed_out = self.step_counter > SUBGOAL_TIMEOUT_MULT * self.subgoal_budget()

    if completed:
      self.advance(obs, "completed")
    elif timed_out:
      self.advance(obs, "timed_out")


# ---------------------------------------------------------------------------
# Observation / action helpers
# ---------------------------------------------------------------------------

def preprocess_obs(
    obs: dict,
    vocab: Vocabulary,
    model,
    *,
    subgoal: str = "",
    device: torch.device = DEVICE,
):
    max_len = getattr(model, "MAX_MISSION_LEN", 32)
    image = torch.tensor(obs["image"], dtype=torch.float32, device=device)
    image = image.permute(2, 0, 1).unsqueeze(0) / 255.0

    if subgoal:
        text_str = f"{obs['mission']} [SEP] {subgoal}"
    else:
        text_str = obs["mission"]
    tokens = vocab.tokenize(text_str, max_len=max_len)
    text = torch.tensor([tokens], dtype=torch.long, device=device)
    return torch_ac.DictList({"image": image, "text": text})


@dataclass
class EpisodeResult:
  episode_index: int
  seed: int
  success: bool
  steps: int
  raw_return: float
  mean_entropy: float
  mission: str
  subgoal_trace: list[dict[str, Any]]


@dataclass
class EnvEvalSummary:
  env_key: str
  display_name: str
  env_id: str
  num_episodes: int
  success_rate: float
  avg_return: float
  avg_steps: float
  mean_entropy: float
  std_return: float
  std_steps: float
  std_entropy: float


def run_episode_lgrl(
    env,
    model,
    vocab: Vocabulary,
    hierarchy: EvalHierarchy,
    *,
    seed: int,
    deterministic: bool,
    device: torch.device,
) -> EpisodeResult:
  obs, _ = env.reset(seed=seed)
  hierarchy.reset_episode()
  memory = torch.zeros(1, model.memory_size, device=device)
  raw_return = 0.0
  entropies: list[float] = []

  done = False
  while not done:
    subgoal = hierarchy.active_subgoal or "search for the target"
    obss = preprocess_obs(obs, vocab, model, subgoal=subgoal, device=device)
    with torch.no_grad():
      dist, _value, memory = model(obss, memory)

    if deterministic:
      action = int(dist.probs.argmax(dim=-1).item())
    else:
      action = int(dist.sample().item())

    entropies.append(float(dist.entropy().item()))

    obs, reward, terminated, truncated, _info = env.step(action)
    raw_return += float(reward)
    done = terminated or truncated
    if not done:
      hierarchy.after_step(obs, action)

  success = raw_return > 0
  return EpisodeResult(
    episode_index=-1,
    seed=seed,
    success=success,
    steps=hierarchy.episode_steps,
    raw_return=raw_return,
    mean_entropy=float(np.mean(entropies)) if entropies else 0.0,
    mission=obs.get("mission", ""),
    subgoal_trace=hierarchy.subgoal_trace,
  )


def run_episode_baseline(
    env,
    model,
    vocab: Vocabulary,
    *,
    seed: int,
    deterministic: bool,
    device: torch.device,
) -> EpisodeResult:
  obs, _ = env.reset(seed=seed)
  memory = torch.zeros(1, model.memory_size, device=device)
  raw_return = 0.0
  entropies: list[float] = []
  steps = 0

  done = False
  while not done:
    obss = preprocess_obs(obs, vocab, model, subgoal="", device=device)
    with torch.no_grad():
      dist, _value, memory = model(obss, memory)

    if deterministic:
      action = int(dist.probs.argmax(dim=-1).item())
    else:
      action = int(dist.sample().item())

    entropies.append(float(dist.entropy().item()))
    obs, reward, terminated, truncated, _info = env.step(action)
    raw_return += float(reward)
    steps += 1
    done = terminated or truncated

  return EpisodeResult(
    episode_index=-1,
    seed=seed,
    success=raw_return > 0,
    steps=steps,
    raw_return=raw_return,
    mean_entropy=float(np.mean(entropies)) if entropies else 0.0,
    mission=obs.get("mission", ""),
    subgoal_trace=[],
  )


def summarize_env_results(
    spec: BenchmarkEnvSpec,
    episodes: list[EpisodeResult],
) -> EnvEvalSummary:
  n = len(episodes)
  successes = [e.success for e in episodes]
  returns = [e.raw_return for e in episodes]
  steps = [e.steps for e in episodes]
  entropies = [e.mean_entropy for e in episodes]
  return EnvEvalSummary(
    env_key=spec.key,
    display_name=spec.display_name,
    env_id=spec.env_id,
    num_episodes=n,
    success_rate=float(np.mean(successes)) if n else 0.0,
    avg_return=float(np.mean(returns)) if n else 0.0,
    avg_steps=float(np.mean(steps)) if n else 0.0,
    mean_entropy=float(np.mean(entropies)) if n else 0.0,
    std_return=float(np.std(returns)) if n else 0.0,
    std_steps=float(np.std(steps)) if n else 0.0,
    std_entropy=float(np.std(entropies)) if n else 0.0,
  )


def evaluate_env(
    spec: BenchmarkEnvSpec,
    model,
    vocab: Vocabulary,
    planner,
    *,
    num_episodes: int,
    seed_start: int,
    agent_type: str,
    deterministic: bool,
    device: torch.device,
    episodes_path: str,
) -> tuple[EnvEvalSummary, list[EpisodeResult]]:
  try:
    env = gym.make(spec.env_id)
  except gym.error.Error as exc:
    raise SystemExit(
      f"Cannot create env {spec.env_id!r} ({spec.display_name}). "
      f"Is it registered in your MiniGrid install?\n  {exc}"
    ) from exc

  sample_obs, _ = env.reset(seed=seed_start)
  mission = sample_obs["mission"]
  n_subgoals = spec.resolve_n_subgoals(mission)
  t_max = spec.resolve_t_max()

  episodes: list[EpisodeResult] = []
  with open(episodes_path, "w", encoding="utf-8") as ep_file:
    for i in range(num_episodes):
      seed = seed_start + i
      if agent_type == "baseline":
        result = run_episode_baseline(
          env, model, vocab, seed=seed, deterministic=deterministic, device=device,
        )
      else:
        hierarchy = EvalHierarchy(env, planner, n_subgoals, t_max)
        result = run_episode_lgrl(
          env, model, vocab, hierarchy,
          seed=seed, deterministic=deterministic, device=device,
        )
      result.episode_index = i
      episodes.append(result)
      ep_file.write(json.dumps({
        "episode": i,
        "seed": seed,
        "success": result.success,
        "steps": result.steps,
        "raw_return": result.raw_return,
        "mean_entropy": result.mean_entropy,
        "mission": result.mission,
        "subgoal_trace": result.subgoal_trace,
      }) + "\n")

  env.close()
  return summarize_env_results(spec, episodes), episodes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
  p = argparse.ArgumentParser(description="Evaluate LGRL/baseline checkpoints.")
  p.add_argument(
    "--checkpoint", required=True,
    help="Path to .pt file (e.g. checkpoints/lgrl_rule.pt)",
  )
  p.add_argument(
    "--agent", choices=["lgrl", "baseline"], default="lgrl",
    help="Model architecture (default: lgrl). Auto-detected from checkpoint if omitted.",
  )
  p.add_argument(
    "--planner", choices=["llm", "rule_based"], default="llm",
    help="Subgoal planner at eval time (default: llm / Ollama).",
  )
  p.add_argument(
    "--llm-model", default="qwen2.5:7b",
    help="Ollama model name when --planner llm.",
  )
  p.add_argument(
    "--ollama-host", default="http://127.0.0.1:11434",
    help="Ollama API base URL (match ollama serve, usually 127.0.0.1:11434).",
  )
  p.add_argument(
    "--llm-timeout", type=float, default=DEFAULT_TIMEOUT_SEC,
    help=(
      f"HTTP timeout per subgoal request in seconds (default: {DEFAULT_TIMEOUT_SEC}). "
      "Increase on CPU-only Ollama (7B can take 30–90s per call)."
    ),
  )
  p.add_argument(
    "--warmup", action=argparse.BooleanOptionalAction, default=True,
    help="Send one Ollama request before eval to load the model (default: on).",
  )
  p.add_argument(
    "--llm-fewshot", action="store_true",
    help="Use the long few-shot system prompt (slower on CPU, default: compact).",
  )
  p.add_argument(
    "--episodes", type=int, default=DEFAULT_NUM_EPISODES,
    help=f"Episodes per environment (default: {DEFAULT_NUM_EPISODES}).",
  )
  p.add_argument(
    "--seed-start", type=int, default=DEFAULT_SEED_START,
    help="First episode seed; episode i uses seed_start + i.",
  )
  p.add_argument(
    "--envs",
    default=None,
    help="Comma-separated suite keys (default: all). "
         "Keys: gotodoor, gotoobject, doorkey5x5, unlockpickup, keycorridor",
  )
  p.add_argument(
    "--output-dir", default=None,
    help="Directory for results (default: logs/eval/<stem>_<timestamp>).",
  )
  p.add_argument(
    "--run-name", default=None,
    help="Optional suffix for the output folder name.",
  )
  p.add_argument(
    "--deterministic", action=argparse.BooleanOptionalAction, default=True,
    help="Greedy argmax actions (default). Use --no-deterministic to sample.",
  )
  p.add_argument(
    "--device", default=None,
    help="cuda or cpu (default: auto).",
  )
  return p.parse_args()


def main():
  args = parse_args()
  device = torch.device(args.device) if args.device else DEVICE

  ckpt_path = os.path.abspath(args.checkpoint)
  ckpt = load_checkpoint(ckpt_path, device)
  vocab = load_vocab_from_checkpoint(ckpt)

  print("=" * 70)
  print("  LGRL Evaluation")
  print("=" * 70)
  print(describe_checkpoint(ckpt, ckpt_path))
  print("=" * 70)

  train_planner = ckpt.get("planner")
  if args.agent == "lgrl" and train_planner == "rule_based" and args.planner == "llm":
    print("  Note: checkpoint was trained with rule_based planner; eval uses LLM.")
  if args.agent == "lgrl" and args.planner == "llm":
    print(f"  Eval planner : LLM ({args.llm_model} @ {args.ollama_host})")
  elif args.agent == "lgrl":
    print("  Eval planner : rule_based oracle")

  env_keys = [k.strip() for k in args.envs.split(",")] if args.envs else None
  suite = get_eval_suite(env_keys)

  stem = os.path.splitext(os.path.basename(ckpt_path))[0]
  ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  run_label = f"{stem}_{ts}"
  if args.run_name:
    run_label = f"{run_label}_{args.run_name}"
  out_dir = args.output_dir or os.path.join(LOG_DIR, run_label)
  os.makedirs(out_dir, exist_ok=True)

  # Build model
  probe_env = gym.make(suite[0].env_id)
  obs_space = probe_env.observation_space
  act_space = probe_env.action_space
  probe_env.close()

  if args.agent == "baseline":
    model = BaselineAgent(obs_space, act_space, vocab)
  else:
    model = LGRLAgent(obs_space, act_space, vocab)
  model.load_state_dict(ckpt["model_state_dict"])
  model.to(device)
  model.eval()

  if args.agent == "lgrl":
    if args.planner == "llm":
      planner = LLMPlanner(
        model_name=args.llm_model,
        host=args.ollama_host,
        timeout=args.llm_timeout,
        compact_prompt=not args.llm_fewshot,
      )
      print(
        f"  LLM timeout    : {args.llm_timeout}s  |  "
        f"prompt: {'few-shot' if args.llm_fewshot else 'compact'}"
      )
      if args.warmup:
        print("  Warming up Ollama (loads model; can take 30–90s on CPU)…")
        try:
          warmup_sec = planner.warmup()
          print(f"  Warmup OK in {warmup_sec:.1f}s")
        except requests.RequestException as exc:
          raise SystemExit(
            f"Ollama warmup failed: {exc}\n"
            "Ensure `ollama serve` is running and the model is pulled:\n"
            f"  ollama pull {args.llm_model}"
          ) from exc
    else:
      planner = RuleBasedPlanner()
  else:
    planner = None

  config_record = {
    "checkpoint": ckpt_path,
    "agent": args.agent,
    "planner": args.planner if args.agent == "lgrl" else None,
    "llm_model": args.llm_model if args.planner == "llm" else None,
    "llm_timeout": args.llm_timeout if args.planner == "llm" else None,
    "llm_compact_prompt": not args.llm_fewshot if args.planner == "llm" else None,
    "llm_warmup": args.warmup if args.planner == "llm" else None,
    "episodes_per_env": args.episodes,
    "seed_start": args.seed_start,
    "deterministic": args.deterministic,
    "device": str(device),
    "environments": [asdict(s) for s in suite],
    "checkpoint_meta": {
      "env": ckpt.get("env"),
      "mix": ckpt.get("mix"),
      "planner": ckpt.get("planner"),
      "update": ckpt.get("update"),
      "total_frames": ckpt.get("total_frames"),
    },
  }
  with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config_record, f, indent=2)

  summaries: list[EnvEvalSummary] = []
  t0 = time.time()

  for spec in suite:
    print(f"\n  Evaluating {spec.display_name} ({spec.env_id}) …")
    ep_path = os.path.join(out_dir, f"episodes_{spec.key}.jsonl")
    summary, _episodes = evaluate_env(
      spec, model, vocab, planner,
      num_episodes=args.episodes,
      seed_start=args.seed_start,
      agent_type=args.agent,
      deterministic=args.deterministic,
      device=device,
      episodes_path=ep_path,
    )
    summaries.append(summary)
    print(
      f"    success_rate={summary.success_rate:.3f}  "
      f"avg_return={summary.avg_return:.4f}  "
      f"avg_steps={summary.avg_steps:.1f}  "
      f"mean_entropy={summary.mean_entropy:.4f}"
    )
    print(f"    → {ep_path}")

  elapsed = time.time() - t0

  # summary.csv
  csv_path = os.path.join(out_dir, "summary.csv")
  fieldnames = list(asdict(summaries[0]).keys()) if summaries else []
  with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for s in summaries:
      writer.writerow(asdict(s))

  summary_json = {
    "elapsed_sec": elapsed,
    "results": [asdict(s) for s in summaries],
  }
  with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=2)

  print("\n" + "=" * 70)
  print("  Summary (success_rate / avg_return / avg_steps / mean_entropy)")
  print("=" * 70)
  for s in summaries:
    print(
      f"  {s.display_name:14s}  "
      f"{s.success_rate:.3f}  "
      f"{s.avg_return:7.4f}  "
      f"{s.avg_steps:7.1f}  "
      f"{s.mean_entropy:.4f}"
    )
  print("=" * 70)
  print(f"  Output directory: {out_dir}")
  print(f"  summary.csv       : {csv_path}")
  print(f"  Elapsed           : {elapsed:.1f}s")
  print("=" * 70)


if __name__ == "__main__":
  main()
