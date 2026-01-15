# training/train_unsloth_trl.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt
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

# -----------------------
# KPI Tracking
# -----------------------
rewards = []
asset_healths = []
costs = []
actions = []

print("Starting agentic RL loop...")

# -----------------------
# Helper: model decides action
# -----------------------
def decide_action(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    if "0" in response:
        return 0
    return 1

# -----------------------
# Training loop
# -----------------------
for step in range(30):
    prompt = (
        f"Asset health: {obs['asset_health'][0]}\n"
        f"Cost so far: {obs['cost'][0]}\n"
        "Choose action:\n"
        "0 = Preventive Maintenance\n"
        "1 = Delay\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    action = decide_action(model, tokenizer, prompt)
    obs, reward, done, _, _ = env.step(action)

    # Reward-weighted update
    rl_loss = loss * (-reward)

    optimizer.zero_grad()
    rl_loss.backward()
    optimizer.step()

    # Track KPIs
    rewards.append(reward)
    asset_healths.append(obs["asset_health"][0])
    costs.append(obs["cost"][0])
    actions.append(action)

    print(f"Step {step:02d} | Action {action} | Reward {reward}")

print("Training finished.")

# -----------------------
# Save trained agent
# -----------------------
model.save_pretrained("trained_planner")
tokenizer.save_pretrained("trained_planner")

# -----------------------
# KPI Dashboard
# -----------------------
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(rewards, label="Reward")
plt.title("Reward per Step")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(asset_healths, color="green")
plt.title("Asset Health Over Time")
plt.xlabel("Step")
plt.ylabel("Health")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(costs, color="red")
plt.title("Cumulative Cost")
plt.xlabel("Step")
plt.ylabel("Cost")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.bar(["PM", "Delay"], [actions.count(0), actions.count(1)])
plt.title("Action Distribution")

plt.tight_layout()
plt.show()
