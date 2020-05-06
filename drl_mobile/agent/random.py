class RandomAgent:
    """Agent that always selects a random action. Following the stable_baselines API."""
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        self.action_space.seed(seed)

    def predict(self, observation):
        """Choose a random action independent of the observation"""
        return self.action_space.sample()
