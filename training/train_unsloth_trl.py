# training/train_unsloth_trl.py

from unsloth import FastLanguageModel
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer
from environment.openenv_maintenance_env import MaintenancePlannerEnv
import torch

# ---------------------------
# Load model using Unsloth
# ---------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# ---------------------------
# PPO configuration
# ---------------------------
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=1,
)

trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)

# ---------------------------
# Environment
# ---------------------------
env = MaintenancePlannerEnv()
obs, _ = env.reset()

print("Starting PPO training...")

# ---------------------------
# Training loop
# ---------------------------
for step in range(20):
    prompt = (
        f"Asset health: {obs['asset_health']}\n"
        f"Cost so far: {obs['cost']}\n"
        "Action?\n"
        "0 = Perform Preventive Maintenance\n"
        "1 = Delay\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Simple heuristic → will be learned away by PPO
    action = 0 if obs["asset_health"] < 70 else 1

    obs, reward, done, _, _ = env.step(action)

    trainer.step(
        queries=[prompt],
        responses=[inputs["input_ids"]],
        rewards=[reward],
    )

    print(f"Step {step} | Action {action} | Reward {reward}")

print("Training completed.")
