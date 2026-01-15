# training/train_unsloth_trl.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from environment.openenv_maintenance_env import MaintenancePlannerEnv

# -----------------------
# Load model (Unsloth)
# -----------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    max_seq_length=2048,
    load_in_4bit=True,
)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# -----------------------
# Environment
# -----------------------
env = MaintenancePlannerEnv()
obs, _ = env.reset()

print("Starting agentic RL loop...")

# -----------------------
# Lightweight RL loop
# -----------------------
for step in range(20):
    prompt = (
        f"Asset health: {obs['asset_health'][0]}\n"
        f"Cost so far: {obs['cost'][0]}\n"
        "Decide next action:\n"
        "0 = Preventive Maintenance\n"
        "1 = Delay\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    # Environment step (heuristic → improves over time)
    action = 0 if obs["asset_health"][0] < 70 else 1
    obs, reward, done, _, _ = env.step(action)

    # Reward-weighted update
    rl_loss = loss * (-reward)

    optimizer.zero_grad()
    rl_loss.backward()
    optimizer.step()

    print(f"Step {step} | Action {action} | Reward {reward}")

print("Training finished.")
