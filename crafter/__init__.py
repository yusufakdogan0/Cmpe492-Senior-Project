"""
crafter/ — Crafter port of the LGRL (LLM-Guided Reinforcement Learning)
framework.

Self-contained: every module imports from siblings in this folder, so the
package does not depend on the MiniGrid project's models/ or utils/.

Public modules:
  crafter_tasks         task registry, recipe graph, stage-plan builder
  crafter_env           Gymnasium wrapper (target-achievement episodes)
  crafter_parser        symbolic state -> JSON description
  crafter_planner       rule-based tech-tree oracle planner
  crafter_llm_planner   LLM (Ollama/Qwen) planner — full-framework test
  crafter_tracker       subgoal completion verification
  crafter_agent         64x64 recurrent actor-critic + Vocabulary
  crafter_logger        per-env subgoal JSONL logging
  seq_env               single-process env stepper (torch-ac ParallelEnv shim)

Training entry points:
  train_baseline.py         mission-only PPO (control)
  train_lgrl_rule.py        LGRL single-task (rule oracle)
  train_lgrl_curriculum.py  LGRL transfer-learning curriculum (easy->hard)
"""
