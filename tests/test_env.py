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
            'episode_length': 100, 'map': map, 'bs_list': bs_list, 'ue_list': ue_list,
            'new_ue_interval': None, 'rand_episodes': False, 'log_metrics': False, 'seed': 42,
        }
        self.env = MultiAgentMobileEnv(env_config)

        # take 3 random steps to initialize
        for i in range(3):
            rand_action = {ue: random.randint(0, 2) for ue in self.env.ue_list}
            self.env.step(rand_action)

    def test_test_action(self):
        # get current rewards
        original_rewards = self.env.update_ue_drs_rewards(penalties=None)
        original_env = copy.deepcopy(self.env)

        # test action
        action = {ue: 1 for ue in self.env.ue_list}
        test_rewards = self.env.test_ue_actions(action)

        # rewards after reverting
        revert_rewards = self.env.update_ue_drs_rewards(penalties=None)
        self.assertEqual(original_rewards, revert_rewards)
