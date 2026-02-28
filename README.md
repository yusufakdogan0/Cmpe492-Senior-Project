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
├── models/              # Neural network architectures
│   └── __init__.py
├── checkpoints/         # Saved model weights (git-ignored)
├── requirements.txt     # Python dependencies
└── .gitignore
```

## Environment Setup

**Prerequisites:** Python 3.9, NVIDIA GPU with CUDA support.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# 2. Install PyTorch with CUDA support
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt
```

## Tech Stack

| Package | Version | Purpose |
|---|---|---|
| PyTorch | 2.5.1 (CUDA) | Deep learning framework |
| torch-ac | 1.4.0 | Actor-critic RL algorithms (PPO) |
| MiniGrid | latest | Grid-world environments |
| Gymnasium | latest | Environment interface |
| NumPy | latest | Numerical computing |

