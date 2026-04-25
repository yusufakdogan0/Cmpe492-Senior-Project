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
│   ├── env_utils.py            # Supported env list, Tmax, artifact-stem helpers
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

## Supported Environments

All three training scripts accept `--env <env_id>`. Supported environments mirror the curriculum from the LGRL paper:

| Env ID                              | Mission template                                        | Stages | `T_max` (from `env.max_steps`) |
|-------------------------------------|---------------------------------------------------------|:------:|:------------------------------:|
| `MiniGrid-DoorKey-5x5-v0` (default) | `use the key to open the door and then get to the goal` | 5      | 250                            |
| `MiniGrid-GoToDoor-5x5-v0`          | `go to the <color> door`                                | 1      | 100                            |
| `MiniGrid-GoToDoor-6x6-v0`          | `go to the <color> door`                                | 1      | 144                            |
| `MiniGrid-GoToDoor-8x8-v0`          | `go to the <color> door`                                | 1      | 256                            |
| `MiniGrid-GoToObject-6x6-N2-v0`     | `go to the <color> <key\|ball\|box>`                    | 1      | 180                            |
| `MiniGrid-GoToObject-8x8-N2-v0`     | `go to the <color> <key\|ball\|box>`                    | 1      | 320                            |

The `GoToDoor` and `GoToObject` families end when the agent issues the `done` action (MiniGrid action 6) while adjacent to the correct target; reaching that state gives the positive environment reward that triggers our mission-level reward (Eq. 5).

Mission-family detection is done from the mission string, not the env name, so custom envs using the same mission templates will also work.

### Artifact Naming

To preserve backward compatibility with existing DoorKey-5x5 runs:

- **`--env MiniGrid-DoorKey-5x5-v0`** (the default) keeps the legacy stems (`baseline`, `lgrl`, `lgrl_rule`).
- Every other `--env` value uses a tagged stem: `{base}_{envtag}` (e.g. `lgrl_rule_gotodoor5x5`).

The env tag is the env id with `MiniGrid-` and `-v0` stripped, hyphens removed, lowercased:

```
MiniGrid-GoToDoor-5x5-v0       -> gotodoor5x5
MiniGrid-GoToObject-6x6-N2-v0  -> gotoobject6x6n2
```

## Training

### Baseline (no subgoal guidance)

Standard PPO agent conditioned only on the mission string. Used as the control condition.

```bash
python scripts/train_baseline.py                                  # DoorKey-5x5 (default)
python scripts/train_baseline.py --env MiniGrid-GoToDoor-5x5-v0
python scripts/train_baseline.py --env MiniGrid-GoToObject-6x6-N2-v0
python scripts/train_baseline.py --resume                         # resume matching checkpoint
```

Artifacts for DoorKey-5x5 (default env):

| Artifact    | Path                                |
|-------------|-------------------------------------|
| Metrics CSV | `logs/baseline_metrics.csv`         |
| Checkpoint  | `checkpoints/baseline.pt`           |
| Plot        | `logs/plots/training_curves.png`    |

For any other env, replace `baseline` with `baseline_<envtag>` (e.g. `baseline_gotodoor5x5.pt`, `logs/plots/baseline_gotodoor5x5_training_curves.png`).

### LGRL with LLM planner (default)

The LLM (Qwen 2.5 7B via Ollama) generates subgoals at each decision point during training. Requires the Ollama server running on `localhost:11434`.

```bash
python scripts/train_lgrl.py                                        # DoorKey-5x5, LLM planner
python scripts/train_lgrl.py --env MiniGrid-GoToDoor-5x5-v0
python scripts/train_lgrl.py --env MiniGrid-GoToObject-6x6-N2-v0
python scripts/train_lgrl.py --subgoal-log                          # enable per-env subgoal logging
python scripts/train_lgrl.py --planner rule_based                   # oracle ablation via same script
python scripts/train_lgrl.py --resume                               # resume from checkpoint
```

Artifacts for DoorKey-5x5:

| Artifact                            | Path                                                          |
|-------------------------------------|---------------------------------------------------------------|
| Metrics CSV                         | `logs/lgrl_metrics.csv`                                       |
| Checkpoint                          | `checkpoints/lgrl.pt`                                         |
| Plot                                | `logs/plots/lgrl_training_curves.png`                         |
| Subgoal logs (with `--subgoal-log`) | `logs/lgrl_subgoal_log/env_00.jsonl` … `env_15.jsonl`         |

For any other env, swap `lgrl` → `lgrl_<envtag>`.

### LGRL with rule-based oracle (ablation)

A stage-based deterministic planner that produces the same subgoal format as the LLM using hand-coded heuristics. Isolates LLM quality from hierarchical reward and text conditioning. Same PPO loop and `LGRLAgent` as the LLM run.

```bash
python scripts/train_lgrl_rule.py                                     # DoorKey-5x5
python scripts/train_lgrl_rule.py --env MiniGrid-GoToDoor-5x5-v0
python scripts/train_lgrl_rule.py --env MiniGrid-GoToObject-6x6-N2-v0
python scripts/train_lgrl_rule.py --subgoal-log                       # enable per-env subgoal logging
python scripts/train_lgrl_rule.py --resume                            # resume from checkpoint
```

Rule training uses this script only (it does not import `train_lgrl.py`).

Artifacts for DoorKey-5x5:

| Artifact                            | Path                                                           |
|-------------------------------------|----------------------------------------------------------------|
| Metrics CSV                         | `logs/lgrl_rule_metrics.csv`                                   |
| Checkpoint                          | `checkpoints/lgrl_rule.pt`                                     |
| Plot                                | `logs/plots/lgrl_rule_training_curves.png`                     |
| Subgoal logs (with `--subgoal-log`) | `logs/lgrl_rule_subgoal_log/env_00.jsonl` … `env_15.jsonl`     |

For any other env, swap `lgrl_rule` → `lgrl_rule_<envtag>`.

## Architecture

The agent operates in a bi-level hierarchy:

- **High level:** A planner (LLM or rule-based oracle) observes the environment state as a JSON description and generates the next subgoal (e.g. "pickup the yellow key").
- **Low level:** A recurrent PPO agent receives the concatenated text `"mission [SEP] subgoal"` alongside the visual observation and selects low-level actions.

### Subgoal Types

Only these subgoal forms are used:

| Subgoal | Format                             | Example                    |
|---------|------------------------------------|----------------------------|
| Search  | `search for the [color] [object]`  | `search for the blue door` |
| Pickup  | `pickup the [color] [object]`      | `pickup the yellow key`    |
| Open    | `open the [status] [color] door`   | `open the locked yellow door` |
| Close   | `close the [status] [color] door`  | `close the open yellow door` |
| Drop    | `drop the [color] [object]`        | `drop the yellow key`      |

### Stage Machines (per env family)

The rule-based planner dispatches on the mission string. The forward-only stage index never decreases, preventing reward farming on earlier subgoals.

#### DoorKey (5 stages)

| Stage | Subgoal                        | Skipped if…                                 |
|:-----:|--------------------------------|---------------------------------------------|
| 0     | `search for the [color] key`   | Key already visible → jump to stage 1       |
| 1     | `pickup the [color] key`       | Already carrying key → jump to stage 2      |
| 2     | `search for the [color] door`  | Door already visible → jump to stage 3      |
| 3     | `open the locked [color] door` | Door already open → jump to stage 4         |
| 4     | `search for the goal`          | —                                           |

#### GoToDoor (1 stage)

| Stage | Subgoal                        |
|:-----:|--------------------------------|
| 0     | `search for the [color] door`  |

The mission-completion reward (Eq. 5) is granted when the agent issues `done` next to the correct door.

#### GoToObject (1 stage)

| Stage | Subgoal                                    |
|:-----:|--------------------------------------------|
| 0     | `search for the [color] [key\|ball\|box]`  |

Same pattern as GoToDoor — the mission terminates on `done` next to the correct object.

### Reward Scaffolding (LGRL paper Eqs. 5–7)

| Symbol           | Formula                                                                        | Default                            |
|------------------|--------------------------------------------------------------------------------|------------------------------------|
| Mission reward   | `r_m = R_MISSION * (1 - MISSION_TIME_COEF * T_used / T_max)`                   | `0.5 * (1 - 0.5 * ratio)`          |
| Subgoal reward   | `r_i = R_SUBGOAL * (1 - SUBGOAL_TIME_COEF * T_used / T_i)`                     | `0.5 * (1 - 0.5 * ratio)`          |
| Subgoal budget   | `T_i = ((stage + 1) / N) * T_max`                                              | Per-env: see table above           |
| Subgoal timeout  | `T_used > SUBGOAL_TIMEOUT_MULT * T_i`  (then `r_i = 0`)                        | `2.0 * T_i`                        |
| Episode total    | `r = r_m + (1/N) * sum(r_i)`                                                   | —                                  |

`T_max` is pulled from `env.unwrapped.max_steps` per-env, so reward scaling is faithful to each env's step budget. Maximum possible episode reward is `R_MISSION + R_SUBGOAL = 1.0` when both the mission and every subgoal are completed instantly.

### Reward & Budget Configuration

All reward-shaping parameters are defined at the top of `train_lgrl.py` and `train_lgrl_rule.py`. They are kept consistent across both scripts.

| Parameter              | Default             | Description                                                     | Notes                                                                 |
|------------------------|---------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------|
| `R_MISSION`            | `0.5`               | Max mission-completion reward (`R_m`)                           | `R_MISSION + R_SUBGOAL` sets the max episode return (1.0 by default)  |
| `R_SUBGOAL`            | `0.5`               | Max per-subgoal reward (`R_t`) before 1/N normalization         | See above                                                             |
| `MISSION_TIME_COEF`    | `0.5`               | 0.5 factor in Eq. 5                                             | At 0.5 an agent finishing at `T_max` still gets `0.25 * R_MISSION`    |
| `SUBGOAL_TIME_COEF`    | `0.5`               | 0.5 factor in Eq. 6                                             | Same shape as mission penalty                                         |
| `T_max`                | `env.max_steps`     | Paper's `T_max`, derived from the env at startup                | Per-env: 250 / 100 / 144 / 256 / 180 / 320 (see table)                |
| `SUBGOAL_TIMEOUT_MULT` | `2.0`               | Subgoal times out when `T_used > mult * T_i`                    | Also caps the ratio in the subgoal reward formula                     |
| `N_SUBGOALS`           | derived from env    | Number of stages for this env family (5 / 1 / 1)                | Queried from `RuleBasedPlanner.num_stages(mission)` at startup         |

### PPO Hyperparameters (LGRL paper Section 4.3)

| Parameter          | Value   | Source                           |
|--------------------|---------|----------------------------------|
| `LR`               | `1e-4`  | Paper §4.3                       |
| `DISCOUNT`         | `0.99`  | Paper §4.3                       |
| `GAE_LAMBDA`       | `0.95`  | Paper §4.3                       |
| `CLIP_EPS`         | `0.2`   | Paper §4.3                       |
| `BATCH_SIZE`       | `256`   | Paper §4.3                       |
| `ENTROPY_COEF`     | `0.01`  | torch-ac default (paper silent)  |
| `VALUE_LOSS_COEF`  | `0.5`   | torch-ac default (paper silent)  |
| `MAX_GRAD_NORM`    | `0.5`   | torch-ac default (paper silent)  |
| `EPOCHS`           | `4`     | torch-ac default (paper silent)  |
| `RECURRENCE`       | `4`     | torch-ac default (paper silent)  |

### Subgoal Logging

When `--subgoal-log` is enabled, per-environment JSONL files are written under `logs/<stem>_subgoal_log/`. Each file traces the full subgoal lifecycle for that environment:

| Event         | When                                       | Key fields                          |
|---------------|--------------------------------------------|-------------------------------------|
| `init`        | First subgoal assigned at episode start    | `subgoal`, `stage`, `env_state`     |
| `completed`   | Subgoal verified as done                   | `steps_used`, `budget`, `reward`    |
| `timed_out`   | Budget exceeded (`2 × T_i` steps)          | `steps_used`, `budget`              |
| `new`         | Next subgoal issued                        | `subgoal`, `stage`, `env_state`, `raw_llm` |
| `episode_end` | Episode terminates                         | `success`, `episode_steps`          |

Each line includes a timestamp and episode counter for grouping.

## Tech Stack

| Package              | Version         | Purpose                           |
|----------------------|-----------------|-----------------------------------|
| PyTorch              | 2.6.0 (CUDA 12.4) | Deep learning framework        |
| torch-ac             | 1.4.0           | Actor-critic RL algorithms (PPO)  |
| MiniGrid             | latest          | Grid-world environments           |
| Gymnasium            | latest          | Environment interface             |
| Ollama + Qwen 2.5 7B | q4_K_M          | Local LLM inference               |
