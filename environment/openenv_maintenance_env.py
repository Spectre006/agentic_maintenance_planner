import gymnasium as gym
from gymnasium import spaces
import random

class MaintenancePlannerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(2)  # 0=PM, 1=Delay
        self.observation_space = spaces.Dict({
            "day": spaces.Discrete(100),
            "asset_health": spaces.Box(low=0, high=100, shape=(1,), dtype=int),
            "cost": spaces.Box(low=0, high=10000, shape=(1,), dtype=int),
        })

        self.max_steps = 50

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.asset_health = 100
        self.cost = 0
        return self._obs(), {}

    def step(self, action):
        self.day += 1
        reward = 0

        if action == 0:  # Preventive Maintenance
            self.asset_health += 5
            self.cost += 100
            reward += 3
        else:  # Delay
            self.asset_health -= random.randint(5, 15)
            reward -= 2

        if self.asset_health <= 0:
            reward -= 20
            self.asset_health = 50

        terminated = self.day >= self.max_steps
        truncated = False

        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        return {
            "day": self.day,
            "asset_health": [self.asset_health],
            "cost": [self.cost],
        }
