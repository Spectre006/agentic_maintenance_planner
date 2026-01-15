# training/train_unsloth_trl.py
# ==========================================================
# Agentic Maintenance Planner
# Gym-style OpenEnv-compatible environment
# Unsloth-based reward-driven agentic RL
# KPI dashboard saved for analysis & blog
# ==========================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from environment.openenv_maintenance_env import MaintenancePlannerEnv

# ==========================================================
# 1. Load model using Unsloth
# ==========================================================
print("\n🔹 Loading language model with Unsloth (GPU required)...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    max_seq_length=2048,
    load_in_4bit=True,
)

model.gradient_checkpointing_disable()
model.config.use_cache = False

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

print("✅ Model loaded successfully.\n")

# ==========================================================
# 2. Initialize environment
# ==========================================================
print("🔹 Initializing simulated maintenance world...")

env = MaintenancePlannerEnv()
obs, _ = env.reset()

print(
    f"   ▶ Initial asset health : {obs['asset_health'][0]}\n"
    f"   ▶ Initial total cost   : {obs['cost'][0]}\n"
)

# ==========================================================
# 3. KPI tracking
# ==========================================================
rewards = []
asset_healths = []
costs = []
actions = []

# ==========================================================
# 4. Action decision helper (STRICT parsing)
# ==========================================================
def decide_action(model, tokenizer, prompt):
    """
    Model decides next action.
    We parse ONLY the last token to avoid bias.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # ✅ Strict parsing
    if response.endswith("0"):
        return 0
    if response.endswith("1"):
        return 1

    # Exploration fallback
    return torch.randint(0, 2, (1,)).item()

# ==========================================================
# 5. Agentic RL loop
# ==========================================================
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

    # Forward pass for learning signal
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    # Model decides
    action = decide_action(model, tokenizer, prompt)

    # Environment reacts
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

    # Human-readable logging
    action_text = "Preventive Maintenance" if action == 0 else "Delay Maintenance"
    outcome_text = (
        "👍 Positive outcome (asset stabilized)"
        if reward > 0
        else "⚠ Negative outcome (risk increased)"
    )

    print(
        f"Step {step + 1:02d} | "
        f"Decision: {action_text} | "
        f"Reward: {reward:+d} | "
        f"Asset Health: {obs['asset_health'][0]} | "
        f"Cost: {obs['cost'][0]} | "
        f"{outcome_text}"
    )

print("\n✅ Training completed.\n")

# ==========================================================
# 6. Save trained agent
# ==========================================================
print("💾 Saving trained maintenance planner...")
model.save_pretrained("trained_planner")
tokenizer.save_pretrained("trained_planner")
print("✅ Trained model saved.\n")

# ==========================================================
# 7. KPI Dashboard (saved as image)
# ==========================================================
print("📊 Generating KPI dashboard...")

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(rewards)
plt.title("Reward per Step (Learning Signal)")
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
plt.title("Decision Distribution")

plt.tight_layout()
plt.savefig("kpi_dashboard.png")

print("📊 KPI dashboard saved as kpi_dashboard.png")
print("🏁 Agentic maintenance training complete.")

# ==========================================================
# 8. Evaluation Episode (NO LEARNING)
# ==========================================================
print("\n🧪 Starting evaluation episode (no learning)...\n")

model.eval()  # 🔒 Freeze model (important)

eval_env = MaintenancePlannerEnv()
eval_obs, _ = eval_env.reset()

eval_total_reward = 0
eval_actions = []
eval_asset_health = []
eval_costs = []

for step in range(20):
    eval_prompt = (
        f"Asset health is {eval_obs['asset_health'][0]}.\n"
        f"Total maintenance cost so far is {eval_obs['cost'][0]}.\n\n"
        "You are a maintenance planner.\n"
        "Choose the best next action:\n"
        "0 = Perform Preventive Maintenance\n"
        "1 = Delay Maintenance\n"
    )

    # Model decides (NO gradient, NO update)
    with torch.no_grad():
        action = decide_action(model, tokenizer, eval_prompt)

    eval_obs, reward, done, _, _ = eval_env.step(action)

    eval_total_reward += reward
    eval_actions.append(action)
    eval_asset_health.append(eval_obs["asset_health"][0])
    eval_costs.append(eval_obs["cost"][0])

    action_text = "Preventive Maintenance" if action == 0 else "Delay Maintenance"

    print(
        f"[EVAL] Step {step + 1:02d} | "
        f"Decision: {action_text} | "
        f"Reward: {reward:+d} | "
        f"Asset Health: {eval_obs['asset_health'][0]} | "
        f"Cost: {eval_obs['cost'][0]}"
    )

print("\n✅ Evaluation completed.\n")

# ----------------------------------------------------------
# Evaluation Summary
# ----------------------------------------------------------
print("📊 Evaluation Summary")
print("---------------------")
print(f"Total evaluation reward : {eval_total_reward}")
print(f"PM actions             : {eval_actions.count(0)}")
print(f"Delay actions          : {eval_actions.count(1)}")
print(f"Final asset health     : {eval_asset_health[-1]}")
print(f"Final total cost       : {eval_costs[-1]}")
