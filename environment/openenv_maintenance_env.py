import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MaintenancePlannerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=30):
        super().__init__()

        self.max_steps = max_steps
        self.current_step = 0

        # Example internal state
        self.asset_health = 1.0
        self.total_cost = 0.0

        # Action: 0 = do nothing, 1 = preventive maintenance
        self.action_space = spaces.Discrete(2)

        # Observation: asset health
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

    # ----------------------------
    # REQUIRED BY OPENENV
    # ----------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.asset_health = 1.0
        self.total_cost = 0.0

        observation = np.array([self.asset_health], dtype=np.float32)
        info = {}

        return observation, info

    def step(self, action):
        self.current_step += 1

        reward = 0.0

        if action == 1:
            # Preventive maintenance
            self.asset_health = 1.0
            self.total_cost += 10
            reward -= 10
        else:
            # Natural degradation
            self.asset_health -= 0.05
            reward += 1

        terminated = self.asset_health <= 0
        truncated = self.current_step >= self.max_steps

        observation = np.array([self.asset_health], dtype=np.float32)

        info = {
            "cost": self.total_cost
        }

        return observation, reward, terminated, truncated, info

    # ----------------------------
    # 🔴 MISSING METHOD – NOW FIXED
    # ----------------------------
    def state(self):
        """
        OpenEnv-required method.
        Returns environment metadata (NOT the observation).
        """
        return {
            "step": self.current_step,
            "asset_health": self.asset_health,
            "total_cost": self.total_cost,
            "max_steps": self.max_steps,
            "done": self.asset_health <= 0 or self.current_step >= self.max_steps
        }
