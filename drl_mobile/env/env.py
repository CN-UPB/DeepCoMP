import random

import gym
import gym.spaces
import structlog
from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt


# TODO: if I want to experiment with different Gym interfaces with diff action/obs space,
#  make one general env and inherit with different versions of act/obs space
class MobileEnv(gym.Env):
    """OpenAI Gym environment with multiple moving UEs and stationary BS on a map"""
    metadata = {'render.modes': ['human']}

    def __init__(self, episode_length, width, height, bs_list, ue_list):
        """
        Create a new environment object with an OpenAI Gym interface
        :param episode_length: Total number of simulation time steps in one episode
        :param width: Width of the map
        :param height: Height of the map
        :param bs_list: List of basestation objects in the environment
        :param ue_list: List of UE objects in the environment
        """
        super(gym.Env, self).__init__()
        # construct the rectangular world map
        self.time = 0
        self.episode_length = episode_length
        self.width = width
        self.height = height
        self.map = Polygon([(0,0), (0, height), (width, height), (width, 0)])
        # save other attributes
        self.bs_list = bs_list
        # pass the env to all users (needed for movement; interference etc)
        self.ue_list = ue_list
        for ue in self.ue_list:
            ue.env = self
        assert len(self.ue_list) == 1, "Currently only support 1 UE"
        # current observation
        self.obs = None
        # observation and action space are defined in the subclass --> different variants
        self.observation_space = None
        self.action_space = None

        self.log = structlog.get_logger()

    @property
    def num_bs(self):
        return len(self.bs_list)

    @property
    def active_bs(self):
        return [bs for bs in self.bs_list if bs.active]

    def seed(self, seed=None):
        random.seed(seed)

    def get_obs(self, ue):
        """
        Return the an observation of the current world for a given UE
        """
        raise NotImplementedError('Implement in subclass')

    def calc_reward(self, action_success: bool):
        """Calculate and return reward"""
        raise NotImplementedError('Implement in subclass')

    def reset(self):
        """Reset environment by resetting time and all UEs (pos & movement) and their connections"""
        self.time = 0
        # TODO: randomize to avoid repeating always the same episode?
        for ue in self.ue_list:
            ue.reset()
        # TODO: this just returns the observation for the 1st UE
        self.obs = self.get_obs(self.ue_list[0])
        return self.obs

    def step(self, action: int):
        """Do 1 time step: Apply action and update UE position. Return new state, reward."""
        # TODO: simplyfing assumption for now: just 1 UE! all actions are applied to 1st UE only!
        ue = self.ue_list[0]
        prev_obs = self.obs

        # apply action; 0 = no op
        success = True
        if action > 0:
            bs = self.bs_list[action-1]
            success = ue.connect_to_bs(bs, disconnect=True)

        ue.move()
        self.time += 1

        # return next observation, reward, done, info
        self.obs = self.get_obs(ue)
        reward = self.calc_reward(success)
        done = self.time >= self.episode_length
        info = {}
        self.log.info("Step", ue=ue, time=self.time, prev_obs=prev_obs, action=action, reward=reward, next_obs=self.obs, done=done)
        return self.obs, reward, done, info

    def render(self, mode='human'):
        """Plot and visualize the current status of the world. Return the patch of actors for animation."""
        # list of matplotlib "artists", which can be used to create animations
        patch = []

        # map borders
        patch.extend(plt.plot(*self.map.exterior.xy, color='gray'))
        # users & connections
        for ue in self.ue_list:
            patch.append(plt.scatter(*ue.pos.xy, color='blue'))
            for bs in ue.conn_bs:
                patch.extend(plt.plot([ue.pos.x, bs.pos.x], [ue.pos.y, bs.pos.y], color='orange'))
        # base stations
        for bs in self.bs_list:
            patch.append(plt.scatter(*bs.pos.xy, marker='^', c='black'))
            patch.extend(plt.plot(*bs.coverage.exterior.xy, color='black'))

        # title isn't redrawn in animation (out of box) --> static --> show time as text inside box, top-right corner
        patch.append(plt.title(type(self).__name__))
        patch.append(plt.text(0.9*self.width, 0.9*self.height, f"t={self.time}"))
        return patch


class BinaryMobileEnv(MobileEnv):
    """Subclass of the general Mobile Env that uses binary observations to indicate which BS are & can be connected"""
    def __init__(self, episode_length, width, height, bs_list, ue_list):
        super().__init__(episode_length, width, height, bs_list, ue_list)
        # observations: binary vector of BS availability (in range and dr >= req_dr) + already connected BS
        self.observation_space = gym.spaces.MultiBinary(2 * self.num_bs)
        # actions: select a BS to be connected to/disconnect from or noop
        self.action_space = gym.spaces.Discrete(self.num_bs + 1)

    def get_obs(self, ue):
        """
        Return the an observation of the current world for a given UE
        It consists of 2 binary vectors: BS availability and already connected BS
        """
        bs_availability = [int(ue.can_connect(bs)) for bs in self.bs_list]
        connected_bs = [int(bs in ue.conn_bs) for bs in self.bs_list]
        return bs_availability + connected_bs

    def calc_reward(self, action_success: bool):
        """Calculate and return reward"""
        # TODO: -1 for losing connection?
        reward = 0
        # +10 for every UE that's connected to at least one BS; -10 for each that isn't
        for ue in self.ue_list:
            if len(ue.conn_bs) >= 1:
                reward += 10
            else:
                reward -= 10
        # -1 if action wasn't successful
        if not action_success:
            reward -= 3
        return reward


class JustConnectedObsMobileEnv(BinaryMobileEnv):
    """Dummy observations just contain binary info about which BS are connected. Nothing about availablility"""
    def __init__(self, episode_length, width, height, bs_list, ue_list):
        super().__init__(episode_length, width, height, bs_list, ue_list)
        # observations: just binary vector of already connected BS
        self.observation_space = gym.spaces.MultiBinary(self.num_bs)
        # same action space as binary env: select a BS to be connected to/disconnect from or noop

    def get_obs(self, ue):
        """Observation: Currently connected BS"""
        connected_bs = [int(bs in ue.conn_bs) for bs in self.bs_list]
        return connected_bs


class DatarateMobileEnv(BinaryMobileEnv):
    """Subclass of the binary MobileEnv that uses the achievable data rate as observations"""
    def __init__(self, episode_length, width, height, bs_list, ue_list, dr_cutoff=200, sub_req_dr=False):
        """
        Env where the achievable data rate is passed as observations
        :param dr_cutoff: Any data rate above this value will be cut off --> help have obs in same range
        :param sub_req_dr: If true, subtract a UE's required data rate from the achievable dr --> neg obs if too little
        """
        super().__init__(episode_length, width, height, bs_list, ue_list)
        self.dr_cutoff = dr_cutoff
        self.sub_req_dr = sub_req_dr
        # observations: binary vector of BS availability (in range & free cap) + already connected BS
        # 1. Achievable data rate for given UE for all BS --> Box;
        # cut off dr at given dr level. here, dr is below 200 anyways --> default doesn't cut off
        max_dr_req = max([ue.dr_req for ue in self.ue_list])
        self.log.info('Max dr req', max_dr_req=max_dr_req, sub_req_dr=self.sub_req_dr)
        assert max_dr_req < dr_cutoff, "Data rate cut off should be higher than max required data rate by UEs"
        # if we subtract the required data rate, observations may become negative
        if self.sub_req_dr:
            dr_low = np.full(shape=(self.num_bs,), fill_value=-max_dr_req)
        else:
            dr_low = np.zeros(self.num_bs)
        dr_high = np.full(shape=(self.num_bs,), fill_value=self.dr_cutoff)
        # 2. Connected BS --> MultiBinary
        conn_low = np.zeros(self.num_bs)
        conn_high = np.ones(self.num_bs)
        # Dict space would be most suitable but not supported by stable baselines 2 --> Box
        # TODO: normalize by dividing with req dr? doesn't change anything as long as all UEs require 1mbit
        self.observation_space = gym.spaces.Box(low=np.concatenate([dr_low, conn_low]),
                                                high=np.concatenate([dr_high, conn_high]))
        # same action space as binary env: select a BS to be connected to/disconnect from or noop

    def get_obs(self, ue):
        """Observation: Achievable data rate per BS + currently connected BS"""
        if self.sub_req_dr:
            bs_dr = [min(bs.data_rate(ue.pos, self.active_bs) - ue.dr_req, self.dr_cutoff) for bs in self.bs_list]
        else:
            bs_dr = [min(bs.data_rate(ue.pos, self.active_bs), self.dr_cutoff) for bs in self.bs_list]
        connected_bs = [int(bs in ue.conn_bs) for bs in self.bs_list]
        return bs_dr + connected_bs
