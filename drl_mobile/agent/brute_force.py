class BruteForceAgent:
    """
    Brute force approach, testing all possible actions and choosing the best one.
    Finds the optimal action per step but requires access to the env to test and evaluate each action.
    """
    def __init__(self, env):
        self.env = env

    def compute_action(self, observation):
        pass
