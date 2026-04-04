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
│   └── train_lgrl_rule.py      # LGRL rule oracle (standalone script; own training loop)
├── utils/
│   ├── __init__.py
│   ├── env_parser.py           # MiniGrid 7x7 observation -> JSON for the LLM
│   ├── llm_planner.py          # LLM subgoal generation (Ollama / Qwen 2.5 7B)
│   ├── rule_based_planner.py   # Oracle ablation baseline (deterministic subgoals)
│   └── subgoal_tracker.py      # Subgoal completion verification
├── checkpoints/                # Saved model weights (git-ignored)
├── logs/                       # Metrics CSV & plots (git-ignored)
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
```

| Artifact | Path |
|----------|------|
| Metrics CSV | `logs/lgrl_llm_metrics.csv` |
| Checkpoint | `checkpoints/lgrl_llm.pt` |
| Plot | `logs/plots/lgrl_llm_training_curves.png` |
| Subgoal log (with `--subgoal-log`) | `logs/lgrl_llm_subgoal_log.jsonl` |

### LGRL with rule-based oracle (ablation)

A deterministic planner that produces the same subgoal string format as the LLM using hand-coded heuristics. Isolates LLM quality from hierarchical reward and text conditioning. Same PPO loop and `LGRLAgent` as the LLM run.

```bash
python scripts/train_lgrl_rule.py
```

Rule training uses this script only (it does not import `train_lgrl.py`). For LLM training use `train_lgrl.py` (optionally `--planner rule_based` still works there if you prefer one file for both).

| Artifact | Path |
|----------|------|
| Metrics CSV | `logs/lgrl_rule_metrics.csv` |
| Checkpoint | `checkpoints/lgrl_rule.pt` |
| Plot | `logs/plots/lgrl_rule_training_curves.png` |
| Subgoal log (with `--subgoal-log`) | `logs/lgrl_rule_subgoal_log.jsonl` |

## Architecture

The agent operates in a bi-level hierarchy:

- **High level:** A planner (LLM or rule-based oracle) observes the environment state as a JSON description and generates the next subgoal (e.g. "pickup the yellow key").
- **Low level:** A recurrent PPO agent receives the concatenated text `"mission [SEP] subgoal"` alongside the visual observation and selects low-level actions.

### Reward Scaffolding

| Symbol | Formula | Value |
|--------|---------|-------|
| Mission reward | `r_m = R_m * (1 - 0.5 * T_used / T_max)` | R_m = 0.5 |
| Subgoal reward | `r_i = R_t * (1 - 0.5 * T_used / T_i)` | R_t = 0.5 |
| Subgoal budget | `T_i = (i / n) * T_max` | n = 5, T_max = 100 |
| Episode total  | `r = r_m + (1/n) * sum(r_i)` | |

## Tech Stack

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.6.0 (CUDA 12.4) | Deep learning framework |
| torch-ac | 1.4.0 | Actor-critic RL algorithms (PPO) |
| MiniGrid | latest | Grid-world environments |
| Gymnasium | latest | Environment interface |
| Ollama + Qwen 2.5 7B | q4_K_M | Local LLM inference |
