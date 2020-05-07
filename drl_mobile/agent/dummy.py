class RandomAgent:
    """Agent that always selects a random action. Following the stable_baselines API."""
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        self.action_space.seed(seed)

    def predict(self, observation, **kwargs):
        """Choose a random action independent of the observation and other args"""
        return self.action_space.sample(), None


class FixedAgent:
    """Agent that always selects a the same fixed action. Following the stable_baselines API."""
    def __init__(self, action):
        self.action = action

    def predict(self, observation, **kwargs):
        """Choose a same fixed action independent of the observation and other args"""
        return self.action, None
