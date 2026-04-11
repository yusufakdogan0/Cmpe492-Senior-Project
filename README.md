# An Experimental Evaluation of LLM-Assisted Hierarchical Reinforcement Learning

This repository contains the codebase for our CMPE 492 Senior Project at Bogazici University. We implement and evaluate an LLM-assisted hierarchical reinforcement learning architecture where a large language model decomposes high-level missions into subgoals that guide a PPO agent's exploration and decision-making in interactive grid-world environments.


**Team:** Onur Kucuk & Yusuf Akdogan
**Advisor:** Emre Ugur

## Documentation

Official project documentation, timeline, and milestones are in the [Repository Wiki](https://github.com/yusufakdogan0/Cmpe492-Senior-Project/wiki).

## Project Structure

```
492proj/
├── models/
│   ├── __init__.py
│   ├── baseline_agent.py       # Recurrent actor-critic (mission-only baseline)
│   └── lgrl_agent.py           # LGRL actor-critic (mission + subgoal via [SEP])
├── scripts/
│   ├── train_baseline.py       # Baseline PPO training (no subgoal guidance)
│   ├── train_lgrl.py           # LGRL training (LLM planner by default)
│   └── train_lgrl_rule.py      # LGRL rule oracle (standalone script)
├── utils/
│   ├── __init__.py
│   ├── env_parser.py           # MiniGrid 7x7 observation -> JSON for the LLM
│   ├── llm_planner.py          # LLM subgoal generation (Ollama / Qwen 2.5 7B)
│   ├── rule_based_planner.py   # Stage-based deterministic oracle planner
│   ├── subgoal_tracker.py      # Subgoal completion verification
│   ├── subgoal_logger.py       # Per-environment JSONL subgoal logging
│   └── sequential_env.py       # Single-process env stepper for torch-ac
├── checkpoints/                # Saved model weights (git-ignored)
├── logs/                       # Metrics CSV, plots & subgoal logs (git-ignored)
├── requirements.txt
└── .gitignore
```

## Setup

**Prerequisites:** Python 3.13, NVIDIA GPU with CUDA support, [Ollama](https://ollama.com/download).

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

ollama pull qwen2.5:7b
```

## Training

### Baseline (no subgoal guidance)

Standard PPO agent conditioned only on the mission string. Used as the control condition.

```bash
python scripts/train_baseline.py
```

| Artifact | Path |
|----------|------|
| Metrics CSV | `logs/baseline_metrics.csv` |
| Checkpoint | `checkpoints/baseline.pt` |
| Plot | `logs/plots/training_curves.png` |

### LGRL with LLM planner (default)

The LLM (Qwen 2.5 7B via Ollama) generates subgoals at each decision point during training. Requires the Ollama server running on `localhost:11434`.

```bash
python scripts/train_lgrl.py
python scripts/train_lgrl.py --subgoal-log          # enable per-env subgoal logging
python scripts/train_lgrl.py --planner rule_based   # oracle ablation via same script
python scripts/train_lgrl.py --resume               # resume from checkpoint
```

| Artifact | Path |
|----------|------|
| Metrics CSV | `logs/lgrl_metrics.csv` |
| Checkpoint | `checkpoints/lgrl.pt` |
| Plot | `logs/plots/lgrl_training_curves.png` |
| Subgoal logs (with `--subgoal-log`) | `logs/lgrl_subgoal_log/env_00.jsonl` … `env_15.jsonl` |

### LGRL with rule-based oracle (ablation)

A stage-based deterministic planner that produces the same subgoal format as the LLM using hand-coded heuristics. Isolates LLM quality from hierarchical reward and text conditioning. Same PPO loop and `LGRLAgent` as the LLM run.

```bash
python scripts/train_lgrl_rule.py
python scripts/train_lgrl_rule.py --subgoal-log     # enable per-env subgoal logging
python scripts/train_lgrl_rule.py --resume           # resume from checkpoint
```

Rule training uses this script only (it does not import `train_lgrl.py`).

| Artifact | Path |
|----------|------|
| Metrics CSV | `logs/lgrl_rule_metrics.csv` |
| Checkpoint | `checkpoints/lgrl_rule.pt` |
| Plot | `logs/plots/lgrl_rule_training_curves.png` |
| Subgoal logs (with `--subgoal-log`) | `logs/lgrl_rule_subgoal_log/env_00.jsonl` … `env_15.jsonl` |

## Architecture

The agent operates in a bi-level hierarchy:

- **High level:** A planner (LLM or rule-based oracle) observes the environment state as a JSON description and generates the next subgoal (e.g. "pickup the yellow key").
- **Low level:** A recurrent PPO agent receives the concatenated text `"mission [SEP] subgoal"` alongside the visual observation and selects low-level actions.

### Subgoal Types 

Only these subgoal forms are used:

| Subgoal | Format | Example |
|---------|--------|---------|
| Search | `search for the [color] [object]` | `search for the yellow key` |
| Pickup | `pickup the [color] [object]` | `pickup the yellow key` |
| Open | `open the [status] [color] door` | `open the locked yellow door` |
| Close | `close the [status] [color] door` | `close the open yellow door` |
| Drop | `drop the [color] [object]` | `drop the yellow key` |

### Stage-Based Subgoal Progression (DoorKey-5x5)

The rule-based planner implements a forward-only 5-stage machine. Stages whose preconditions are already met are automatically skipped:

| Stage | Subgoal | Skipped if… |
|-------|---------|-------------|
| 0 | `search for the [color] key` | Key already visible → jump to stage 1 |
| 1 | `pickup the [color] key` | Already carrying key → jump to stage 2 |
| 2 | `search for the [color] door` | Door already visible → jump to stage 3 |
| 3 | `open the locked [color] door` | Door already open → jump to stage 4 |
| 4 | `search for the goal` | — |

### Reward Scaffolding

| Symbol | Formula | Default |
|--------|---------|---------|
| Mission reward | `r_m = R_MISSION * (1 - MISSION_TIME_COEF * T_used / MAX_ENV_STEPS)` | 0.5 * (1 - 0.5 * ratio) |
| Subgoal reward | `r_i = R_SUBGOAL * (1 - SUBGOAL_TIME_COEF * T_used / T_i)` | 0.5 * (1 - 0.5 * ratio) |
| Subgoal budget | `T_i = ((stage+1) / N_SUBGOALS) * MAX_SUBGOAL_STEPS` | (stage+1)/5 * 250 |
| Subgoal timeout | `T_used > SUBGOAL_TIMEOUT_MULT * T_i` | 2.0 * T_i |
| Episode total  | `r = r_m + (1/N_SUBGOALS) * sum(r_i)` | |

If a subgoal times out, its reward is 0 and the agent advances to the next stage. Maximum possible episode reward is `R_MISSION + R_SUBGOAL = 1.0` (when all subgoals and the mission are completed instantly).

### Reward & Budget Configuration

All reward-shaping parameters are defined at the top of `train_lgrl.py` and `train_lgrl_rule.py`. They are kept consistent across both scripts.

| Parameter | Default | Description | Dependencies / notes |
|-----------|---------|-------------|----------------------|
| `R_MISSION` | 0.5 | Max mission-completion reward | `R_MISSION + R_SUBGOAL` sets the theoretical max episode return (1.0 by default) |
| `R_SUBGOAL` | 0.5 | Max per-subgoal reward (before 1/n normalization) | See above |
| `MISSION_TIME_COEF` | 0.5 | Steepness of mission time penalty (paper Eq. 5) | At 0.5 an agent finishing at `T_max` still gets 0.25 mission reward; at 0.9 it gets only 0.05 |
| `SUBGOAL_TIME_COEF` | 0.5 | Steepness of subgoal time penalty (paper Eq. 6) | Higher values punish slow subgoal completion more aggressively |
| `MAX_ENV_STEPS` | 250 | `T_max` used in the mission reward ratio | Only affects mission reward; independent of subgoal budgets |
| `MAX_SUBGOAL_STEPS` | 250 | `T_max` used in subgoal budget calculation `T_i` | Increase to give the agent more time per subgoal without changing mission reward |
| `SUBGOAL_TIMEOUT_MULT` | 2.0 | Subgoal times out when `steps > mult * T_i` | Also caps the ratio in the subgoal reward formula |
| `N_SUBGOALS` | 5 | Number of stages (derived from `RuleBasedPlanner.NUM_STAGES`) | Changing this requires modifying the planner's stage machine |

### Subgoal Logging

When `--subgoal-log` is enabled, per-environment JSONL files are written under `logs/<stem>_subgoal_log/`. Each file traces the full subgoal lifecycle for that environment:

| Event | When | Key fields |
|-------|------|------------|
| `init` | First subgoal assigned at episode start | `subgoal`, `stage`, `env_state` |
| `completed` | Subgoal verified as done | `steps_used`, `budget`, `reward` |
| `timed_out` | Budget exceeded (2×T_i steps) | `steps_used`, `budget` |
| `new` | Next subgoal issued | `subgoal`, `stage`, `env_state`, `raw_llm` |
| `episode_end` | Episode terminates | `success`, `episode_steps` |

Each line includes a timestamp and episode counter for grouping.

## Tech Stack

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.6.0 (CUDA 12.4) | Deep learning framework |
| torch-ac | 1.4.0 | Actor-critic RL algorithms (PPO) |
| MiniGrid | latest | Grid-world environments |
| Gymnasium | latest | Environment interface |
| Ollama + Qwen 2.5 7B | q4_K_M | Local LLM inference |