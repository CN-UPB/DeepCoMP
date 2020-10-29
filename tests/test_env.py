"""Tests for the base env"""
import copy
import random
from unittest import TestCase

from drl_mobile.util.env_setup import create_env_config, get_env
from drl_mobile.env.multi_ue.multi_agent import MultiAgentMobileEnv


class TestEnv(TestCase):
    def setUp(self) -> None:
        map, ue_list, bs_list = get_env('small', 100, 1, 1, 1, 'resource-fair')
        env_config = {
            'episode_length': 50, 'map': map, 'bs_list': bs_list, 'ue_list': ue_list,
            'new_ue_interval': None, 'rand_episodes': False, 'log_metrics': False, 'seed': 42,
        }
        self.env = MultiAgentMobileEnv(env_config)

    def test_env_rewards(self):
        """Ensure that the reward function always returns the same value within a step"""
        while self.env.time < self.env.episode_length:
            rewards1 = self.env.update_ue_drs_rewards(penalties=None)
            rewards2 = self.env.update_ue_drs_rewards(penalties=None)
            self.assertEqual(rewards1, rewards2)

            # progress env with random action
            action = {ue: random.randint(0, 2) for ue in self.env.ue_list}
            self.env.step(action)

    def test_test_action(self):
        """
        Ensure that testing an action does not alter the environment
        by comparing the UE's rewards before and after testing.
        """
        while self.env.time < self.env.episode_length:
            # get current rewards
            original_rewards = self.env.update_ue_drs_rewards(penalties=None)

            # test a random action
            test_env = copy.deepcopy(self.env)
            action = {ue: random.randint(0, 2) for ue in self.env.ue_list}
            test_rewards = test_env.test_ue_actions(action)

            # rewards after reverting
            revert_rewards = test_env.update_ue_drs_rewards(penalties=None)
            self.assertEqual(original_rewards, revert_rewards)

            # progress the environment
            self.env.step(action)
