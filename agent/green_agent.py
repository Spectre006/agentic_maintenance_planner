
class GreenAgent:
    def __init__(self, policy_fn):
        self.policy_fn = policy_fn

    def act(self, observation):
        prompt = f"""
Asset health: {observation['asset_health']}
Cost so far: {observation['cost']}
Choose action:
0 = Perform Preventive Maintenance
1 = Delay
"""
        return self.policy_fn(prompt)
