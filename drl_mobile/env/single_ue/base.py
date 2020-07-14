"""Base mobile environment. Implemented and extended by sublcasses."""
import random
import logging

import gym
import gym.spaces
import structlog
import numpy as np
import matplotlib.pyplot as plt

from drl_mobile.util.logs import config_logging


class MobileEnv(gym.Env):
    """
    Base environment class with moving UEs and stationary BS on a map. RLlib and OpenAI Gym-compatible.
    No observation or action space implemented. This needs to be done in subclasses.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config):
        """
        Create a new environment object with an OpenAI Gym interface. Required fields in the env_config:

        * episode_length: Total number of simulation time steps in one episode
        * map: Map object representing the playground
        * bs_list: List of base station objects in the environment
        * ue_list: List of UE objects in the environment
        * seed: Seed for the RNG; for reproducibility. May be None.

        :param env_config: Dict containing all configuration options for the environment. Required by RLlib.
        """
        super(gym.Env, self).__init__()
        self.time = 0
        self.episode_length = env_config['episode_length']
        self.map = env_config['map']
        self.bs_list = env_config['bs_list']
        self.ue_list = env_config['ue_list']
        # seed the environment
        self.seed(env_config['seed'])

        # current observation
        self.obs = None
        # observation and action space are defined in the subclass --> different variants
        self.observation_space = None
        self.action_space = None

        # configure logging inside env to ensure it works in ray/rllib. https://github.com/ray-project/ray/issues/9030
        config_logging(round_digits=3)
        self.log = structlog.get_logger()
        self.log.info('Env init', env_config=env_config)

    @property
    def num_bs(self):
        return len(self.bs_list)

    @property
    def num_ue(self):
        return len(self.ue_list)

    def seed(self, seed=None):
        random.seed(seed)

    def set_log_level(self, log_dict):
        """
        Set a logging levels for a set of given logger. Needs to happen here, inside the env, for RLlib workers to work.
        :param dict log_dict: Dict with logger name --> logging level (eg, logging.INFO)
        """
        for logger_name, level in log_dict.items():
            logging.getLogger(logger_name).setLevel(level)

    def get_obs(self, ue):
        """Return the an observation of the current world for a given UE"""
        raise NotImplementedError('Implement in subclass')

    def calc_reward(self, ue, penalty):
        """
        Calculate and return reward for specific UE: The UE's utility (based on its data rate) + penalty
        """
        return ue.utility + penalty

    def reset(self):
        """Reset environment by resetting time and all UEs (pos & movement) and their connections"""
        self.time = 0
        for ue in self.ue_list:
            ue.reset()
        for bs in self.bs_list:
            bs.reset()
        # TODO: this just returns the observation for the 1st UE
        self.obs = self.get_obs(self.ue_list[0])
        return self.obs

    def apply_ue_actions(self, action):
        """
        Apply actions of UEs. In this base case, just of one UE (selected based on current time).
        In extended env version, apply actions of all UEs.

        :param: Action to be applied (here: for a single UE)
        :return: Dict of penalties for each UE based on unsuccessful connection attempts (-3)
        """
        assert self.action_space.contains(action), f"Action {action} does not fit action space {self.action_space}"
        penalties = {ue: 0 for ue in self.ue_list}
        # select active UE (to update in this step) using round robin
        ue = self.ue_list[self.time % self.num_ue]

        # apply action: try to connect to BS; or: 0 = no op
        if action > 0:
            bs = self.bs_list[action-1]
            # penalty of -3 for unsuccessful connection attempt
            penalties[ue] = -3 * (not ue.connect_to_bs(bs, disconnect=True))

        return penalties

    def update_ue_drs_rewards(self, penalties):
        """
        Update cached data rates of all UE-BS connections.
        Calculate and return corresponding rewards based on given penalties.

        :param penalties: Dict of penalties for all UEs. Used for calculating rewards.
        :return: Dict of rewards: UE --> reward (incl. penalty)
        """
        rewards = dict()
        for ue in self.ue_list:
            ue.update_curr_dr()
            rewards[ue] = self.calc_reward(ue, penalties[ue])
        return rewards

    def move_ues(self):
        """
        Move all UEs and return dict of penalties corresponding to number of lost connections.

        :return: Penalties for lost connections: UE --> -1 * num. lost connections
        """
        penalties = dict()
        for ue in self.ue_list:
            num_lost_conn = ue.move()
            # add penalty of -1 for each lost connection through movement (rather than actively disconnected)
            penalties[ue] = -num_lost_conn
        return penalties

    def next_obs(self):
        """
        Return next observation after a step.
        Here, the obs for the next UE. Overwritten by env vars as needed.

        :returns: Next observation
        """
        next_ue = self.ue_list[self.time % self.num_ue]
        return self.get_obs(next_ue)

    def step_reward(self, rewards):
        """
        Return the overall reward for the step (called at the end of a step). Overwritten by variants.

        :param rewards: Dict of avg rewards per UE (before and after movement)
        :returns: Reward for the step (depends on the env variant; here just for one UE)
        """
        # here: get reward for UE that applied the action (at time - 1)
        ue = self.ue_list[(self.time-1) % self.num_ue]
        return rewards[ue]

    def done(self):
        """
        Return whether the episode is done.

        :return: Whether the current episode is done or not
        """
        return self.time >= self.episode_length

    def info(self):
        """Return info dict that's returned after a step"""
        info_dict = {
            'time': self.time,
            'dr': {ue: ue.curr_dr for ue in self.ue_list},
            'utility': {ue: ue.utility for ue in self.ue_list}
        }
        # TODO: add info about unsuccessful conn. attempts and dropped connections
        return info_dict

    def step(self, action):
        """
        Environment step consisting of 1) Applying actions, 2) Updating data rates and rewards, 3) Moving UEs,
        4) Updating data rates and rewards again (avg with reward before), 5) Updating the observation

        In the base env here, only one UE applies actions per time step. This is overwritten in other env variants.

        :param action: Action to be applied. Here, for a single UE. In other env variants, for all UEs.
        :return: Tuple of next observation, reward, done, info
        """
        prev_obs = self.obs

        # perform step: apply action, move UEs, update data rates and rewards in between; increment time
        penalties = self.apply_ue_actions(action)
        rewards_before = self.update_ue_drs_rewards(penalties)
        penalties = self.move_ues()
        rewards_after = self.update_ue_drs_rewards(penalties)
        rewards = {ue: np.mean([rewards_before[ue], rewards_after[ue]]) for ue in self.ue_list}
        self.time += 1

        # get and return next obs, reward, done, info
        self.obs = self.next_obs()
        reward = self.step_reward(rewards)
        done = self.done()
        info = self.info()
        self.log.info("Step", time=self.time, prev_obs=prev_obs, action=action, reward=reward, next_obs=self.obs,
                      done=done)
        return self.obs, reward, done, info

    def render(self, mode='human'):
        """Plot and visualize the current status of the world. Return the patch of actors for animation."""
        # list of matplotlib "artists", which can be used to create animations
        patch = []

        # limit to map borders
        plt.xlim(0, self.map.width)
        plt.ylim(0, self.map.height)

        # users & connections
        for ue in self.ue_list:
            # plot connections to all BS
            for bs in ue.bs_dr.keys():
                patch.extend(plt.plot([ue.pos.x, bs.pos.x], [ue.pos.y, bs.pos.y], color='blue'))
            # plot UE
            patch.extend(ue.plot())

        # base stations
        for bs in self.bs_list:
            patch.extend(bs.plot())

        # title isn't redrawn in animation (out of box) --> static --> show time as text inside box, top-right corner
        patch.append(plt.title(type(self).__name__))
        # extra info: time step, curr data rate
        patch.append(plt.text(0.9*self.map.width, 0.9*self.map.height, f"t={self.time}"))

        # legend doesn't change --> only draw once at the beginning
        # if self.time == 0:
        #     plt.legend(loc='upper left')
        return patch
