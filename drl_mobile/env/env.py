import gym
import gym.spaces
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


class MobileEnv(gym.Env):
    """OpenAI Gym environment with multiple moving UEs and stationary BS on a map"""
    # FIXME: rethink Gym design (who calls what when?) and fix/adjust implementation; not complete like this
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

        # set observation and action space
        # observations: binary vector of BS availability (in range & free cap)
        self.observation_space = gym.spaces.MultiBinary(self.num_bs)
        # actions: select a single BS to be connected to
        self.action_space = gym.spaces.Discrete(self.num_bs)

    @property
    def num_bs(self):
        return len(self.bs_list)

    def step(self, action):
        """Do 1 time step and update UE position. Apply action. Return new state, reward."""
        for ue in self.ue_list:
            ue.move()
            # test: always try to connect to same BS
            # ue.connect_to_bs(self.bs_list[1])
            print(action)

    def plot(self, title=None):
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


