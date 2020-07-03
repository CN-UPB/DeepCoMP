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

        self.log = structlog.get_logger()
        # configure logging inside env to ensure it works in ray/rllib. https://github.com/ray-project/ray/issues/9030
        config_logging(round_digits=3)

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
        Calculate and return reward for specific UE. Called before and after UE movement.
        High positive if connected with enough data rate, high negative if otherwise.
        Add penalty for undesired actions, eg, unsuccessful connection attempt; passed as arg.
        """
        reward = penalty
        # +10 if UE is connected such that its dr requirement is satisfied
        if ue.curr_dr >= ue.dr_req:
            reward += 10
        # -10 if not connected with sufficient data rate
        else:
            reward -= 10

        return reward

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

    def step(self, action: int):
        """
        Do 1 time step: Apply action and update UE position. Return new state, reward.
        Only update one UE at a time. With multiple UEs, select active UE using round robin.
        """
        penalty = 0
        # select active UE (to update in this step) using round robin
        ue = self.ue_list[self.time % self.num_ue]
        prev_obs = self.obs

        # apply action; 0 = no op
        if action > 0:
            bs = self.bs_list[action-1]
            # penalty of -3 for unsuccessful connection attempt
            penalty = -3 * (not ue.connect_to_bs(bs, disconnect=True))

        # check connections and reward before and after moving
        # TODO: usually before & after are the same anyways; so I can drop this if the simulator becomes too slow
        reward_before = self.calc_reward(ue, penalty)
        num_lost_conn = ue.move()
        # add penalty of -1 for each lost connection through movement (rather than actively disconnected)
        penalty -= num_lost_conn
        self.time += 1
        reward_after = self.calc_reward(ue, penalty)

        # return next observation, reward, done, info
        # get obs of next UE
        next_ue = self.ue_list[self.time % self.num_ue]
        self.obs = self.get_obs(next_ue)
        # average reward
        reward = np.mean([reward_before, reward_after])
        done = self.time >= self.episode_length
        info = {'time': self.time}
        self.log.info("Step", time=self.time, ue=ue, prev_obs=prev_obs, action=action, reward_before=reward_before,
                      reward_after=reward_after, reward=reward, next_obs=self.obs, next_ue=next_ue, done=done)
        # print(f"{self.time=}, {ue=}, {prev_obs=}, {action=}, {reward=}, {self.obs=}")
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
            for bs in ue.conn_bs:
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
