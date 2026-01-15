
from openenv import OpenEnv
import random

class MaintenancePlannerEnv(OpenEnv):
    def __init__(self):
        super().__init__()
        self.max_steps = 50

    def reset(self, seed=None):
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
        elif action == 1:  # Delay
            self.asset_health -= random.randint(5, 15)
            reward -= 2

        if self.asset_health <= 0:
            reward -= 20
            self.asset_health = 50

        terminated = self.day >= self.max_steps
        return self._obs(), reward, terminated, False, {}

    def _obs(self):
        return {
            "day": self.day,
            "asset_health": self.asset_health,
            "cost": self.cost
        }
