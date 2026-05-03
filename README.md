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
| `MiniGrid-GoToDoor-5x5-v0`          | `go to the <color> door`                                | 2      | 100                            |
| `MiniGrid-GoToDoor-6x6-v0`          | `go to the <color> door`                                | 2      | 144                            |
| `MiniGrid-GoToDoor-8x8-v0`          | `go to the <color> door`                                | 2      | 256                            |
| `MiniGrid-GoToObject-6x6-N2-v0`     | `go to the <color> <key\|ball\|box>`                    | 2      | 180                            |
| `MiniGrid-GoToObject-8x8-N2-v0`     | `go to the <color> <key\|ball\|box>`                    | 2      | 320                            |
| `MiniGrid-UnlockPickup-v0`          | `pick up the <color> box`                               | 9      | 288                            |

The `GoToDoor` and `GoToObject` families end when the agent issues the `done` action (MiniGrid action 6) while adjacent to the correct target; reaching that state gives the positive environment reward that triggers our mission-level reward (Eq. 5).

The `UnlockPickup` family ends when the agent picks up the target object in the locked room. It is the only family in this codebase where the mission text mentions a *target* color while the **key/door color is different and inferred from observation**.

Mission-family detection is done from the mission string, not the env name, so custom envs using the same mission templates will also work.

### Artifact Naming

To preserve backward compatibility with existing DoorKey-5x5 runs:

- **`--env MiniGrid-DoorKey-5x5-v0`** (the default) keeps the legacy stems (`baseline`, `lgrl`, `lgrl_rule`).
- Every other `--env` value uses a tagged stem: `{base}_{envtag}` (e.g. `lgrl_rule_gotodoor5x5`).

The env tag is the env id with `MiniGrid-` and `-v0` stripped, hyphens removed, lowercased:

```
MiniGrid-GoToDoor-5x5-v0       -> gotodoor5x5
MiniGrid-GoToObject-6x6-N2-v0  -> gotoobject6x6n2
MiniGrid-UnlockPickup-v0       -> unlockpickup
```

## Training

### Baseline (no subgoal guidance)

Standard PPO agent conditioned only on the mission string. Used as the control condition.

```bash
python scripts/train_baseline.py                                  # DoorKey-5x5 (default)
python scripts/train_baseline.py --env MiniGrid-GoToDoor-5x5-v0
python scripts/train_baseline.py --env MiniGrid-GoToObject-6x6-N2-v0
python scripts/train_baseline.py --env MiniGrid-UnlockPickup-v0
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
python scripts/train_lgrl.py --env MiniGrid-UnlockPickup-v0
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
python scripts/train_lgrl_rule.py --env MiniGrid-UnlockPickup-v0
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

### Mixed-task training (paper §4.5)

The paper does **not** train UnlockPickup standalone from scratch. The reward is too sparse for random exploration to bootstrap (a uniform-random policy effectively never solves an episode within `T_max`, so PPO has no learning signal). Two setups are reported instead:

- §4.4 (Table 2): UnlockPickup is tested as **zero-shot transfer** from a KeyCorridor-trained agent.
- §4.5 (single-step convergence, Fig. 3): UnlockPickup is trained **mixed** with `GoToObject` at a 1:3 ratio. The easy task gives the agent dense reward signal for navigation while it learns the harder task.

`--mix` is mutually exclusive with `--env`. Format: `env1:r1,env2:r2`. The total ratio must divide `NUM_ENVS=16` evenly. With ratio 1:3 you get 4 UnlockPickup + 12 GoToObject worker envs. All three training scripts support it identically:

```bash
python scripts/train_lgrl_rule.py \
    --mix "MiniGrid-UnlockPickup-v0:1,MiniGrid-GoToObject-6x6-N2-v0:3"

python scripts/train_lgrl.py --planner llm \
    --mix "MiniGrid-UnlockPickup-v0:1,MiniGrid-GoToObject-6x6-N2-v0:3"

python scripts/train_baseline.py \
    --mix "MiniGrid-UnlockPickup-v0:1,MiniGrid-GoToObject-6x6-N2-v0:3"
```

In mix mode, the per-update stdout breakdown shows each family's episode count, return, steps, and success rate independently. The CSV gains four columns per family — `<fam>_episodes`, `<fam>_avg_return`, `<fam>_avg_steps`, `<fam>_success_rate` — so UnlockPickup convergence can be tracked without being washed out by the much-easier GoToObject episodes that dominate the global average. The plot panels for return and steps overlay per-family curves on the global curve.

Mix-mode artifacts:

| Artifact     | Path                                                                       |
|--------------|----------------------------------------------------------------------------|
| Metrics CSV  | `logs/lgrl_rule_mix_unlockpickup_gotoobject6x6n2_1to3_metrics.csv`         |
| Checkpoint   | `checkpoints/lgrl_rule_mix_unlockpickup_gotoobject6x6n2_1to3.pt`           |
| Plot         | `logs/plots/lgrl_rule_mix_unlockpickup_gotoobject6x6n2_1to3_training_curves.png` |

Stem format: `<base>_mix_<env1tag>_<env2tag>_<r1>to<r2>`. Different mixes get distinct artifact paths so multiple ratio experiments don't clobber each other. Resume validates that the saved mix spec matches the requested one.

Single-env runs (`--env`) keep their original CSV format and artifact paths byte-for-byte; mix-mode CSV is a strict superset.

## Architecture

The agent operates in a bi-level hierarchy:

- **High level:** A planner (LLM or rule-based oracle) observes the environment state as a JSON description and generates the next subgoal (e.g. "pickup the yellow key").
- **Low level:** A recurrent PPO agent receives the concatenated text `"mission [SEP] subgoal"` alongside the visual observation and selects low-level actions.

### Subgoal Types

Only these subgoal forms are used:

| Subgoal | Format                             | Example                    |
|---------|------------------------------------|----------------------------|
| Search  | `search for the [color] [object]`  | `search for the blue door` |
| Go near | `go near the [color] [object]`     | `go near the blue key`     |
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

#### GoToDoor (2 stages)

| Stage | Subgoal                        | Skipped if…                                 |
|:-----:|--------------------------------|---------------------------------------------|
| 0     | `search for the [color] door`  | Door already visible → jump to stage 1      |
| 1     | `go near the [color] door`     | —                                           |

The mission-completion reward (Eq. 5) is granted when the agent issues `done` next to the correct door. The `go near` subgoal rewards the agent for becoming cardinally adjacent to the door before that final `done`.

#### GoToObject (2 stages)

| Stage | Subgoal                                    | Skipped if…                                    |
|:-----:|--------------------------------------------|------------------------------------------------|
| 0     | `search for the [color] [key\|ball\|box]`  | Object already visible → jump to stage 1       |
| 1     | `go near the [color] [key\|ball\|box]`     | —                                              |

Same pattern as GoToDoor — the mission terminates on `done` next to the correct object.

#### UnlockPickup (9 stages)

The only family where the mission text mentions a target color while the **key/door color is different and inferred from observation** (a visible key in the starting room, or a visible door if the key is hidden). Each `go near` stage rewards adjacency, then the next stage rewards the actual interaction (pickup or open).

| Stage | Subgoal                                          | Skipped if…                                              |
|:-----:|--------------------------------------------------|----------------------------------------------------------|
| 0     | `search for the [key_color] key`                 | Key visible → jump to 1; carrying key → jump to 3        |
| 1     | `go near the [key_color] key`                    | Already carrying key → jump to stage 3                   |
| 2     | `pickup the [key_color] key`                     | Already carrying key → jump to stage 3                   |
| 3     | `search for the [door_color] door`               | Locked door visible → jump to 4; open door visible → 6   |
| 4     | `go near the [door_color] door`                  | Door already open → jump to stage 6                      |
| 5     | `open the locked [door_color] door`              | Door already open → jump to stage 6                      |
| 6     | `search for the [target_color] [target_object]`  | Target visible → jump to 7; carrying target → jump to 8  |
| 7     | `go near the [target_color] [target_object]`     | Carrying target → jump to stage 8                        |
| 8     | `pickup the [target_color] [target_object]`      | —                                                        |

The mission ends when the agent picks up the target — the env emits a positive reward at that moment, which triggers the mission-level reward (Eq. 5). Note that to pick up the target the agent must first drop the key (MiniGrid's `pickup` action requires an empty inventory); this is left as something the agent learns from the stage-8 reward signal rather than encoded as an explicit `drop` subgoal, matching the paper's decomposition.

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
| `T_max`                | `env.max_steps`     | Paper's `T_max`, derived from the env at startup                | Per-env: 250 / 100 / 144 / 256 / 180 / 320 / 288 (see table)          |
| `SUBGOAL_TIMEOUT_MULT` | `2.0`               | Subgoal times out when `T_used > mult * T_i`                    | Also caps the ratio in the subgoal reward formula                     |
| `N_SUBGOALS`           | derived from env    | Number of stages for this env family (5 / 2 / 2 / 9)            | Queried from `RuleBasedPlanner.num_stages(mission)` at startup         |

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
