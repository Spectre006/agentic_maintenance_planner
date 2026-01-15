# Agentic Maintenance Planner with Reinforcement Learning (RL)

Via OpenEnv we create execution environment for AI Agent Maintenance Planner leanr to takes decision on Single Asset to do PM or Delay using Reinforcement Learning.
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

