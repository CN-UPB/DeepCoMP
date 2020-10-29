import math


class BruteForceAgent:
    """
    Brute force approach, testing all possible actions and choosing the best one.
    Finds the optimal action per step but requires access to the env to test and evaluate each action.
    """
    def __init__(self, env):
        self.env = env

    @staticmethod
    def number_to_base(n, b, num_digits=None):
        """
        Convert any decimal integer to a new number with any base.
        Adjusted from: https://stackoverflow.com/a/28666223/2745116

        :param n: Decimal integer
        :param b: Base
        :param num_digits: Number of digits to return
        :return: List representing the new number. One list element per digit.
        """
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        result = digits[::-1]

        if num_digits is None:
            return result
        # pad with zeros to get the desired number of digits
        assert num_digits >= len(result), "Num digits too small to represent converted number."
        missing_digits = num_digits - len(result)
        return [0 for _ in range(missing_digits)] + result

    def get_ith_action(self, i):
        """Get the i-th action, when walking through the entire action space"""
        action_list = self.number_to_base(i, self.env.num_bs, num_digits=self.env.num_ue)
        action_dict = {self.env.ue_list[j]: action_list[j] for j in range(self.env.num_ue)}
        assert self.env.action_space.contains(action_dict)
        return action_dict

    def compute_action(self, observation):
        """Test all actions and return the best one"""
        best_action = None
        best_reward = -math.inf

        for i in range(self.env.num_ue ** self.env.num_bs):
            action = self.get_ith_action(i)
            rewards = self.env.test_action(action)
            print(rewards)
            # TODO: test
