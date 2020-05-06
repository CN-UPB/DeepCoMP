import gym
import gym.spaces
import structlog
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


class MobileEnv(gym.Env):
    """OpenAI Gym environment with multiple moving UEs and stationary BS on a map"""
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, bs_list, ue_list):
        super().__init__()
        # construct the rectangular world map
        self.width = width
        self.height = height
        self.map = Polygon([(0,0), (0, height), (width, height), (width, 0)])
        # save other attributes
        self.bs_list = bs_list
        # pass the map to all users (needed for movement)
        self.ue_list = ue_list
        for ue in self.ue_list:
            ue.map = self.map
        assert len(self.ue_list) == 1, "Currently only support 1 UE"
        # set observation and action space
        # observations: binary vector of BS availability (in range & free cap) + already connected BS
        self.observation_space = gym.spaces.MultiBinary(2 * self.num_bs)
        # actions: select a BS to be connected to/disconnect from or noop
        self.action_space = gym.spaces.Discrete(self.num_bs + 1)

        self.log = structlog.get_logger(width=width, height=height, bs_list=self.bs_list, ue_list=self.ue_list,
                                        obs_space=self.observation_space, act_space=self.action_space)

    @property
    def num_bs(self):
        return len(self.bs_list)

    def seed(self, seed=None):
        raise NotImplementedError()

    def get_obs(self, ue):
        """
        Return the an observation of the current world for a given UE
        It consists of 2 binary vectors: BS availability and already connected BS
        """
        bs_availability = [int(ue.can_connect(bs)) for bs in self.bs_list]
        connected_bs = [int(bs in ue.assigned_bs) for bs in self.bs_list]
        return bs_availability + connected_bs

    def calc_reward(self, action_success: bool):
        """Calculate and return reward"""
        reward = 0
        # +10 for every UE that's connected to at least one BS; -10 for each that isn't
        for ue in self.ue_list:
            if len(ue.assigned_bs) >= 1:
                reward += 10
            else:
                reward -= 10
        # -1 if action wasn't successful
        if not action_success:
            reward -= 1
        return reward

    def reset(self):
        """Reset environment by resetting all UEs (pos & movement) and their connections"""
        # TODO: randomize to avoid repeating always the same episode?
        for ue in self.ue_list:
            ue.reset()
        # TODO: this just returns the observation for the 1st UE
        return self.get_obs(self.ue_list[0])

    def step(self, action: int):
        """Do 1 time step: Apply action and update UE position. Return new state, reward."""
        # TODO: simplyfing assumption for now: just 1 UE! all actions are applied to 1st UE only!
        ue = self.ue_list[0]
        self.log.info("Step", ue=ue, action=action)

        # apply action; 0 = no op
        if action > 0:
            bs = self.bs_list[action-1]
            success = ue.connect_to_bs(bs, disconnect=True)

        ue.move()

        # return next observation, reward, done, info
        obs = self.get_obs(ue)
        reward = self.calc_reward(success)
        # TODO: the env needs to know the number of steps to set this!
        done = False
        info = {}
        return obs, reward, done, info

    def render(self, mode='human', title=None):
        """Plot and visualize the current status of the world"""
        # square figure and equal aspect ratio to avoid distortions
        plt.figure(figsize=(5, 5))
        plt.gca().set_aspect('equal')

        # map borders
        plt.plot(*self.map.exterior.xy)
        # users & connections
        for ue in self.ue_list:
            plt.scatter(*ue.pos.xy)
            for bs in ue.assigned_bs:
                plt.plot([ue.pos.x, bs.pos.x], [ue.pos.y, bs.pos.y], color='orange')
        # base stations
        for bs in self.bs_list:
            plt.scatter(*bs.pos.xy, marker='^', c='black')
            plt.plot(*bs.coverage.exterior.xy, color='black')

        plt.title(title)
        plt.show()
