import math


class BruteForceAgent:
    """
    Brute force approach, testing all possible actions and choosing the best one.
    Finds the optimal action per step but requires access to the env to test and evaluate each action.
    Optimal in terms of the reward function of the central agent, eg, sum of UE utilities per step.
    """
    def __init__(self):
        self.env = None

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
        # special case n=0
        if n == 0:
            if num_digits is None:
                return [0]
            else:
                return [0 for _ in range(num_digits)]

        # actual conversion
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
        result = [0 for _ in range(missing_digits)] + result
        assert len(result) == num_digits
        return result

    @staticmethod
    def highest_number(base, num_digits):
        """Return the highest number that can be represented as a number of given base and with given num digits"""
        # return sum([(base - 1) * base**d for d in range(num_digits)])
        # much simpler: take the next higher number that can no longer be included, ie, b^d, and subtract 1
        return base**num_digits - 1

    def get_ith_action(self, i):
        """Get the i-th action, when walking through the entire action space"""
        # convert to number with base num_bs + 1, ie, actions selecting 0 (=noop) or one of the BS
        action_list = self.number_to_base(i, self.env.num_bs + 1, num_digits=self.env.num_ue)
        assert self.env.action_space.contains(action_list)
        return action_list

    def compute_action(self, observation):
        """Test all actions and return the best one"""
        assert self.env is not None, "Set agent's env before computing actions."
        best_action = None
        best_reward = -math.inf

        # each UE has num_bs + 1 choices: noop or one of the BS --> (num_bs+1)^num_ue -1 options
        # TODO: any way to parallelize this? not if they are working on the same object..
        for i in range(self.highest_number(self.env.num_bs + 1, self.env.num_ue)):
            action_list = self.get_ith_action(i)
            # need to test the action in dict form
            action_dict = self.env.get_ue_actions(action_list)
            rewards = self.env.test_ue_actions(action_dict)
            reward = self.env.step_reward(rewards)
            if reward > best_reward:
                # need to return the action in list form
                best_action = action_list
                best_reward = reward

        return best_action
