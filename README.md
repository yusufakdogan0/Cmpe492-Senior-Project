# An Experimental Evaluation of LLM-Assisted Hierarchical Reinforcement Learning

This repository contains the codebase for our CMPE 492 Senior Project at Boğaziçi University. This project focuses on implementing, adapting, and evaluating LLM-assisted hierarchical reinforcement learning architectures. 

We explore how Large Language Models can effectively decompose high-level missions into manageable subgoals to guide a PPO agent's exploration and decision-making in interactive environments.

**Team:** Onur Küçük & Yusuf Akdoğan  
**Advisor:** Emre Uğur  

## Project Documentation
All official project documentation, including our timeline, milestones, and methodology, is maintained in the [Repository Wiki](https://github.com/yusufakdogan0/Cmpe492-Senior-Project/wiki).

## Project Structure

```
492proj/
├── models/                     # Neural network architectures
│   ├── __init__.py
│   └── baseline_agent.py       # Recurrent actor-critic (ConvNet+LSTM / Embedding+GRU)
├── scripts/                    # Training & evaluation entry points
│   └── train_baseline.py       # PPO training loop with CSV logging & plotting
├── utils/                      # Shared utilities for LLM integration
│   ├── __init__.py
│   ├── env_parser.py           # MiniGrid observation → JSON for the LLM
│   └── llm_planner.py          # LLM-based subgoal generation via Ollama
├── checkpoints/                # Saved model weights (git-ignored)
├── logs/                       # Training metrics & plots (git-ignored)
├── requirements.txt            # Python dependencies
└── .gitignore
```

## Environment Setup

**Prerequisites:** Python 3.13, NVIDIA GPU with CUDA support, [Ollama](https://ollama.com/download) (for LLM inference).

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# 2. Install PyTorch with CUDA support
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Pull the LLM model (Qwen 2.5 7B, 4-bit quantized)
ollama pull qwen2.5:7b
```

## Tech Stack

| Package | Version | Purpose |
|---|---|---|
| PyTorch | 2.6.0 (CUDA 12.4) | Deep learning framework |
| torch-ac | 1.4.0 | Actor-critic RL algorithms (PPO) |
| MiniGrid | latest | Grid-world environments |
| Gymnasium | latest | Environment interface |
| NumPy | latest | Numerical computing |
| Matplotlib | latest | Training curve visualization |
| Requests | latest | HTTP client for Ollama API |
| Ollama | 0.17+ | Local LLM inference server |
| Qwen 2.5 7B | q4_K_M | LLM for subgoal generation |
