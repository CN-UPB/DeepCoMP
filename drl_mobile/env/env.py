import gym
import gym.spaces
import structlog
from shapely.geometry import Polygon
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
        super().__init__()
        # construct the rectangular world map
        self.time = 0
        self.episode_length = episode_length
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
        # current observation
        self.obs = None
        # observations: binary vector of BS availability (in range & free cap) + already connected BS
        self.observation_space = gym.spaces.MultiBinary(2 * self.num_bs)
        # actions: select a BS to be connected to/disconnect from or noop
        self.action_space = gym.spaces.Discrete(self.num_bs + 1)

        self.log = structlog.get_logger()

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
        # TODO: -1 for losing connection?
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
        # square figure and equal aspect ratio to avoid distortions
        fig = plt.figure(figsize=(5, 5))
        plt.gca().set_aspect('equal')
        # list of matplotlib "actors", which can be used to create animations
        patch = []

        # map borders
        patch.extend(plt.plot(*self.map.exterior.xy))
        # users & connections
        for ue in self.ue_list:
            patch.append(plt.scatter(*ue.pos.xy))
            for bs in ue.assigned_bs:
                patch.extend(plt.plot([ue.pos.x, bs.pos.x], [ue.pos.y, bs.pos.y], color='orange'))
        # base stations
        for bs in self.bs_list:
            patch.append(plt.scatter(*bs.pos.xy, marker='^', c='black'))
            patch.extend(plt.plot(*bs.coverage.exterior.xy, color='black'))

        plt.title(f"t={self.time}")
        plt.show()
        return patch
