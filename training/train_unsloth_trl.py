# training/train_unsloth_trl.py
# --------------------------------------------------
# Agentic Maintenance Planner
# OpenEnv-style Gym environment + Unsloth RL training
# --------------------------------------------------

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from environment.openenv_maintenance_env import MaintenancePlannerEnv

# --------------------------------------------------
# 1. Load model using Unsloth (GPU)
# --------------------------------------------------
print("\n🔹 Loading language model with Unsloth...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 🔴 Important stability fixes
model.gradient_checkpointing_disable()
model.config.use_cache = False

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

print("✅ Model loaded successfully on GPU.\n")

# --------------------------------------------------
# 2. Initialize Maintenance Environment
# --------------------------------------------------
print("🔹 Initializing simulated maintenance world...")

env = MaintenancePlannerEnv()
obs, _ = env.reset()

print(
    f"   ▶ Initial asset health: {obs['asset_health'][0]}\n"
    f"   ▶ Initial maintenance cost: {obs['cost'][0]}\n"
)

# --------------------------------------------------
# 3. KPI Tracking (Business Metrics)
# --------------------------------------------------
rewards = []
asset_healths = []
costs = []
actions = []

# --------------------------------------------------
# 4. Helper: Let the MODEL decide the action
# --------------------------------------------------
def decide_action(model, tokenizer, prompt):
    """
    The language model reads the situation
    and decides whether to perform maintenance or delay.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Simple text-to-action mapping
    if "0" in response:
        return 0
    return 1


# --------------------------------------------------
# 5. Training Loop (Agentic RL)
# --------------------------------------------------
print("🚀 Starting agentic reinforcement learning loop...\n")

for step in range(30):
    prompt = (
        f"Asset health is {obs['asset_health'][0]}.\n"
        f"Total maintenance cost so far is {obs['cost'][0]}.\n\n"
        "You are a maintenance planner.\n"
        "Choose the best next action:\n"
        "0 = Perform Preventive Maintenance\n"
        "1 = Delay Maintenance\n"
    )

    # Language model forward pass (for learning signal)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    # Agent decides action
    action = decide_action(model, tokenizer, prompt)

    # Environment responds
    obs, reward, done, _, _ = env.step(action)

    # Reward-weighted policy update
    rl_loss = loss * (-reward)

    optimizer.zero_grad()
    rl_loss.backward()
    optimizer.step()

    # Track KPIs
    rewards.append(reward)
    asset_healths.append(obs["asset_health"][0])
    costs.append(obs["cost"][0])
    actions.append(action)

    # --------------------------------------------------
    # Human-friendly logging
    # --------------------------------------------------
    action_text = "Preventive Maintenance" if action == 0 else "Delay Maintenance"
    outcome_text = (
        "👍 Good decision (asset stabilized)"
        if reward > 0
        else "⚠ Risky decision (asset degradation)"
    )

    print(
        f"Step {step + 1:02d} | "
        f"Decision: {action_text} | "
        f"Reward: {reward} | "
        f"Asset Health: {obs['asset_health'][0]} | "
        f"Cost: {obs['cost'][0]} | "
        f"{outcome_text}"
    )

print("\n✅ Training completed.\n")

# --------------------------------------------------
# 6. Save trained agent
# --------------------------------------------------
print("💾 Saving trained maintenance planner model...")
model.save_pretrained("trained_planner")
tokenizer.save_pretrained("trained_planner")
print("✅ Model saved successfully.\n")

# --------------------------------------------------
# 7. KPI Dashboard (Visualization)
# --------------------------------------------------
print("📊 Generating KPI dashboard...\n")

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(rewards, label="Reward per Step")
plt.title("Learning Signal (Reward)")
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
plt.title("Cumulative Maintenance Cost")
plt.xlabel("Step")
plt.ylabel("Cost")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.bar(
    ["Preventive Maintenance", "Delay"],
    [actions.count(0), actions.count(1)],
)
plt.title("Planner Decision Distribution")

plt.tight_layout()
plt.show()

print("🏁 KPI dashboard displayed. Agentic maintenance training complete.")
