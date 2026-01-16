class GreenAgent:
    """
    OpenEnv Green Agent Wrapper
    A simple, non-expert agent used for baseline interaction.
    """

    def __init__(self, policy_fn):
        self.policy_fn = policy_fn

    def act(self, observation):
        return self.policy_fn(observation)
