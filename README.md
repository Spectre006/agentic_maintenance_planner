# Agentic Maintenance Planner with Reinforcement Learning (RL)

Via OpenEnv we create execution environment for AI Agent Maintenance Planner learn to takes decision on Single Asset to do PM or Delay using Reinforcement Learning.
In case PM is selected, Asset health is maintained and AI Agent is given Reward as part of Reinforcement Learning.
In case Delay is selected, Asset health decreased and AI Agent is given Penalty.
Unsloth is used for fine tuning the AI Agent.

In the notebook which run on Google Colab with Runtime of T4 GPU is used for Training and Evaluation.

On execution we also prepare KPI Dashboard for Reinforcement Learning.

## Execution

!git clone https://github.com/Spectre006/agentic_maintenance_planner.git
%cd agentic_maintenance_planner

!pip install -r requirements.txt

!python training/train_unsloth_trl_singleasset.py


## OpenEnv Compliance

This environment implements the OpenEnv API standard:

- `reset()` – starts a new episode
- `step(action)` – executes an agent action
- `state()` – returns environment metadata (required by OpenEnv)

A Green Agent wrapper is provided to allow base or inexperienced agents
to interact with the environment without environment-specific logic.