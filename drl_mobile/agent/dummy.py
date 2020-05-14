class RandomAgent:
    """Agent that always selects a random action. Following the stable_baselines API."""
    def __init__(self, action_space, num_vec_envs=1, seed=None):
        self.action_space = action_space
        self.action_space.seed(seed)
        # number of envs inside the VecEnv determines the number of actions to make in each step; or None if no VecEnv
        self.num_vec_envs = num_vec_envs

    def predict(self, observation, **kwargs):
        """Choose a random action independent of the observation and other args"""
        # num_vec_envs=None means we don't use a VecEnv --> return action directly (not in array)
        if self.num_vec_envs is None:
            return self.action_space.sample(), None
        else:
            return [self.action_space.sample() for _ in range(self.num_vec_envs)], None


class FixedAgent:
    """Agent that always selects a the same fixed action. Following the stable_baselines API."""
    def __init__(self, action, num_vec_envs=1):
        self.action = action
        # number of envs inside the VecEnv determines the number of actions to make in each step; or None if no VecEnv
        self.num_vec_envs = num_vec_envs

    def predict(self, observation, **kwargs):
        """Choose a same fixed action independent of the observation and other args"""
        if self.num_vec_envs is None:
            return self.action, None
        else:
            return [self.action for _ in range(self.num_vec_envs)], None
