
class GreenAgent:
    def __init__(self, policy_fn):
        self.policy_fn = policy_fn

    def act(self, observation):
        prompt = (
            f"Asset health: {observation['asset_health']}\n"
            f"Cost so far: {observation['cost']}\n"
            "Choose action:\n"
            "0 = Perform Preventive Maintenance\n"
            "1 = Delay"
        )
        return self.policy_fn(prompt)
