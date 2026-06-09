# LGRL on Crafter — LLM-Guided Reinforcement Learning, ported to a survival environment

This folder is a self-contained port of our LGRL (LLM-Guided Reinforcement
Learning) framework from MiniGrid to [Crafter](https://github.com/danijar/crafter),
a 2D open-world survival game with a 22-achievement tech tree. It is part
of our CMPE 492 Senior Project at Boğaziçi University. See the main project
README for the LGRL background and acknowledgment.

**Team:** Onur Küçük & Yusuf Akdoğan
**Advisor:** Emre Uğur

The port keeps the LGRL **spine** identical to the MiniGrid implementation —

> planner → subgoal → tracker verifies → reward scaffold → PPO

— and replaces only the environment-specific I/O. The reward scaffolding,
PPO hyperparameters, forward-only stage logic, and per-subgoal time budgets
are unchanged from the MiniGrid implementation. Everything here imports from
siblings in this folder, so it does **not** depend on the MiniGrid project's
`models/` or `utils/`.

## Why Crafter

The assignment was to apply the framework to an environment that is neither
MiniGrid nor BabyAI. Crafter is a good fit because:

- It has an explicit **tech-tree dependency DAG** (wood → table → wood
  pickaxe → stone → … → iron pickaxe → diamond), which maps directly onto
  LGRL's forward-only subgoal sequence.
- Its reward is **extremely sparse** — a random policy essentially never
  collects even wood within a short horizon (our verification shows 0
  completions across 6 random-seeded rollouts), so the LGRL subgoal
  scaffolding has a real effect to measure.
- It exposes ground-truth symbolic state (`semantic` grid, `inventory`,
  `achievements`) so the parser and tracker need no engine-internal hacks.

## How the port maps to the MiniGrid code

| Concept                | MiniGrid file              | Crafter file              |
|------------------------|----------------------------|---------------------------|
| Env list / Tmax / stems| `utils/env_utils.py`       | `crafter_tasks.py`        |
| Per-family stage tables| `rule_based_planner.py`    | `crafter_tasks.py` (`build_stage_plan`) |
| Observation → JSON     | `utils/env_parser.py`      | `crafter_parser.py`       |
| Rule oracle planner    | `rule_based_planner.py`    | `crafter_planner.py`      |
| LLM planner            | `utils/llm_planner.py`     | `crafter_llm_planner.py`  |
| Subgoal completion     | `utils/subgoal_tracker.py` | `crafter_tracker.py`      |
| Agent (actor-critic)   | `models/lgrl_agent.py`     | `crafter_agent.py`        |
| Subgoal logging        | `utils/subgoal_logger.py`  | `crafter_logger.py`       |
| Parallel-env shim      | `utils/sequential_env.py`  | `seq_env.py`              |
| Baseline training      | `scripts/train_baseline.py`| `train_baseline.py`       |
| LGRL rule training     | `scripts/train_lgrl_rule.py`| `train_lgrl_rule.py`     |
| Curriculum training    | `scripts/train_lgrl_curriculum.py` | `train_lgrl_curriculum.py` |

## The mission abstraction

A "mission" is a single **target achievement**. `crafter_tasks.py` decomposes
it into a forward-only sequence of subgoal **stages** whose preconditions are
quantity-correct: the materials for any craft/place are gathered in the
stage(s) immediately before that action, and the tools/stations needed to
harvest a material are produced before that material is gathered. This makes
each `collect N item` threshold an absolute "top up to N" that the planner's
`inventory >= N` skip can never misfire on. All 20 task plans are simulated
against the recipe graph at import time (`_validate_all_plans`).

Example — `make a stone pickaxe` (7 stages):

```
 0. collect 2 wood
 1. place a table
 2. collect 1 wood
 3. make a wood pickaxe
 4. collect 1 wood
 5. collect 1 stone
 6. make a stone pickaxe
```

`collect a diamond` is the deepest target at 14 stages.

### Subgoal grammar

| Subgoal     | Format                        | Example                |
|-------------|-------------------------------|------------------------|
| Collect N   | `collect <N> <material>`      | `collect 2 wood`       |
| Collect     | `collect <material>`          | `collect diamond`      |
| Place       | `place a <station>`           | `place a furnace`      |
| Make        | `make a/an <tool>`            | `make an iron pickaxe` |
| Eat         | `eat a cow`                   | `eat a cow`            |
| Defeat      | `defeat a <foe>`              | `defeat a zombie`      |

The rule planner decides **which** subgoal to show next (skipping
already-satisfied stages, judged purely from the current observation —
inventory + nearby stations). The tracker decides **completion** via Crafter
achievement diffs (or inventory thresholds for the `collect N` stages),
mirroring the MiniGrid planner/tracker split.

## Episode design

`crafter_env.py` wraps `crafter.Env` (legacy 4-tuple gym) into a Gymnasium
5-tuple env with a MiniGrid-style episodic goal:

- **Terminates with reward +1** the step the target achievement unlocks →
  `success = reward > 0`, so the mission reward transfers unchanged.
- **Terminates with reward 0** on death.
- **Truncates with reward 0** at `T_max`.

`T_max` is per-task: `max(150, 120 + 60 · n)` where `n` is the stage count
(Crafter's native 10,000-step episode is far too long for per-subgoal time
budgeting). The agent observes a `64×64×3` RGB image plus the text stream.

## Setup

**Prerequisites:** Python 3.13, NVIDIA GPU with CUDA support. (Optional, for
the LLM-planner condition: [Ollama](https://ollama.com/download).)

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r crafter/requirements.txt

# Optional, only for the full-framework (LLM planner) condition:
ollama pull qwen2.5:7b
```

## Supported tasks

`crafter_tasks.SUPPORTED_TASKS` (each usable with `--task`):

| Task                | Mission                | Stages |
|---------------------|------------------------|:------:|
| `collect_wood`      | collect wood           | 1      |
| `place_table`       | place a table          | 2      |
| `make_wood_pickaxe` | make a wood pickaxe    | 4      |
| `collect_stone`     | collect stone          | 5      |
| `collect_coal`      | collect coal           | 5      |
| `place_furnace`     | place a furnace        | 6      |
| `make_stone_pickaxe`| make a stone pickaxe   | 7      |
| `collect_iron`      | collect iron           | 8      |
| `make_iron_pickaxe` | make an iron pickaxe   | 13     |
| `collect_diamond`   | collect a diamond      | 14     |

Plus `make_wood_sword`, `make_stone_sword`, `make_iron_sword`,
`place_stone`, `place_plant`, `collect_sapling`, `collect_drink`,
`eat_cow`, `defeat_zombie`, `defeat_skeleton`.

## Training

### Baseline (mission-only control)

Standard PPO conditioned only on the mission string — no planner, no
subgoals, no reward shaping. Average return equals success rate.

```bash
python crafter/train_baseline.py --task collect_wood
python crafter/train_baseline.py --task make_stone_pickaxe --frames 5000000
python crafter/train_baseline.py --task collect_stone --resume
```

Artifacts: `checkpoints/baseline_<task>.pt`, `logs/baseline_<task>_metrics.csv`,
`logs/plots/baseline_<task>_training_curves.png`.

### LGRL with the rule-based oracle

The rule planner is the **training oracle**: it isolates hierarchical reward
+ text conditioning from LLM quality.

```bash
python crafter/train_lgrl_rule.py --task collect_wood
python crafter/train_lgrl_rule.py --task make_stone_pickaxe --frames 5000000
python crafter/train_lgrl_rule.py --task collect_diamond --subgoal-log
python crafter/train_lgrl_rule.py --task collect_stone --resume
```

Artifacts: `checkpoints/lgrl_rule_<task>.pt`,
`logs/lgrl_rule_<task>_metrics.csv`,
`logs/plots/lgrl_rule_<task>_training_curves.png`, and with `--subgoal-log`,
`logs/lgrl_rule_<task>_subgoal_log/env_00.jsonl … env_15.jsonl`.

### LGRL curriculum / transfer learning (the headline experiment)

Trains sequentially through the tech tree from easiest to hardest, **carrying
the agent's weights and optimizer state forward** across stage boundaries so
each harder target inherits everything learned on the shallower ones. This is
the Crafter analogue of the transfer-learning curriculum in our MiniGrid work.

Default curriculum:

```
collect_wood → place_table → make_wood_pickaxe → collect_stone
→ make_stone_pickaxe → place_furnace → collect_coal → collect_iron
→ make_iron_pickaxe → collect_diamond
```

```bash
# Full tech-tree curriculum (default), up to 1M frames per stage
python crafter/train_lgrl_curriculum.py

# Custom curriculum
python crafter/train_lgrl_curriculum.py --curriculum collect_wood,place_table,make_wood_pickaxe

# Tighter advancement (higher bar, more stability required)
python crafter/train_lgrl_curriculum.py --success-threshold 0.85 --success-stability 15

# Hard-cap a stage so wallclock stays predictable
python crafter/train_lgrl_curriculum.py --max-frames-per-stage 500000

# Resume (the saved curriculum spec must match)
python crafter/train_lgrl_curriculum.py --resume
```

**Advancement rule** (same "constantly above threshold, not once" logic as
the MiniGrid curriculum): a stage advances when the rolling success rate over
the last `--success-window` (default 200) episodes stays ≥ `--success-threshold`
(default 0.80) for `--success-stability` (default 10) consecutive PPO updates
— a single dip resets the counter — OR when `--max-frames-per-stage` (default
1,000,000) is reached. Model + vocab + optimizer persist across boundaries;
only the env workers and per-stage config (`n_subgoals`, `T_max`, target) are
rebuilt.

Artifacts (default curriculum): `checkpoints/lgrl_rule_curriculum_techtree.pt`,
`logs/lgrl_rule_curriculum_techtree_metrics.csv` (with `stage_idx` /
`stage_env` columns), and a plot whose panels mark stage boundaries with
vertical lines and show a curriculum-progress staircase.

### LGRL with the LLM planner (full framework)

To test the **full** framework (LLM decomposition + PPO low-level control),
swap `RuleBasedCrafterPlanner` for `CrafterLLMPlanner` (Ollama / Qwen 2.5 7B).
The planner exposes the same `get_subgoal(mission, env_json, direction,
stage_index)` interface, so it is a drop-in replacement; it requires the
Ollama server running on `localhost:11434`. `crafter_llm_planner.py` ships a
Crafter-specific system prompt with the subgoal grammar, tech-tree rules, and
four few-shot examples.

## Reward scaffolding

Identical to the MiniGrid scripts:

| Symbol         | Formula                                                | Default          |
|----------------|--------------------------------------------------------|------------------|
| Mission reward | `r_m = R_MISSION · (1 − 0.5 · T_used / T_max)`         | `R_MISSION=0.5`  |
| Subgoal reward | `r_i = R_SUBGOAL · (1 − 0.5 · T_used / T_i)`           | `R_SUBGOAL=0.5`  |
| Subgoal budget | `T_i = ((stage + 1) / n) · T_max`                      | per-task         |
| Subgoal timeout| `T_used > 2 · T_i`  (then `r_i = 0`)                   | mult `2.0`       |
| Episode total  | `r = r_m + (1/n) · Σ r_i`                              | —                |

Max possible episode return is `R_MISSION + R_SUBGOAL = 1.0`.

## PPO hyperparameters

`NUM_ENVS=16`, `NUM_FRAMES_PER_PROC=128`, `LR=1e-4`, `DISCOUNT=0.99`,
`GAE_LAMBDA=0.95`, `CLIP_EPS=0.2`, `BATCH_SIZE=256`, `ENTROPY_COEF=0.01`,
`VALUE_LOSS_COEF=0.5`, `MAX_GRAD_NORM=0.5`, `EPOCHS=4`, `RECURRENCE=4`.

## Agent architecture

`CrafterACModel` (used by **both** the baseline and LGRL — they differ only
in their text input):

- **Visual:** `64×64×3` → Conv(8×8,s4) → Conv(4×4,s2) → Conv(3×3,s1) →
  flatten (1024) → LSTM(128). (This Nature-style downsampling stack is the
  only architectural change from the MiniGrid agent's 7×7 conv stack;
  everything downstream is identical.)
- **Text:** Embedding(256, 32) → GRU(128), over `mission` (baseline) or
  `mission [SEP] subgoal` (LGRL).
- **Fusion:** concat (256) → actor head + critic head (FFN 64).

≈ 771K parameters.

## Verification

`crafter/verify_pipeline.py` runs a consolidated correctness check (no real
training): validates all stage plans, runs env + parser + planner + tracker
over real rollouts (asserting the planner is forward-only and the tracker
fires on real/injected achievement transitions), does a model forward pass on
a real observation batch, and runs one full `reshape_reward` step against a
live env.

```bash
python crafter/verify_pipeline.py
```

Each module also has a `__main__` self-test (e.g. `python crafter/crafter_tasks.py`
prints and validates every stage plan).

## Evaluation plan

For each target task, compare **Base** (`train_baseline.py`),
**LGRL** (`train_lgrl_rule.py` or the LLM planner), and the **curriculum /
transfer** agent (`train_lgrl_curriculum.py`) on:

- **success rate** — target achievement unlocked within `T_max`;
- **average steps** to success.

The Crafter score (geometric mean over the 22 achievements) can be added as a
recognizable secondary metric.

## Notes & caveats

- **Compute.** `64×64 × 16 envs` over the long horizons the deeper targets
  need is heavy; the deepest tasks (`make_iron_pickaxe`, `collect_diamond`)
  are realistic only with a GPU and the curriculum carrying the agent in.
- **Survival noise.** Hunger/health/monsters can kill the agent mid-chain —
  noise the MiniGrid tasks never had. Short `T_max` and starting from shallow
  targets mitigate it.
- The verification was run CPU-only for wiring correctness; the smoke tests in
  development used 2-update runs. Full training requires a GPU.
