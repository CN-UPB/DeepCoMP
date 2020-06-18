# working dummy envs for testing with RLlib
import gym
import gym.spaces
from shapely.geometry import Polygon
import structlog


# example for a custom env: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
# start left & go left +1, go right until end of tunnel: +100; anything else: 0
# rather than just optimizing short-term reward (going left), the agent should learn to opitmize long-term rward (right)
class TunnelEnv(gym.Env):
    # must take a singe arg "env_config" that's a dict
    def __init__(self, env_config):
        self.len_tunnel = env_config['len_tunnel']
        self.pos = 0
        self.len_episode = env_config['len_episode']
        self.time = 0
        # actions: left & right
        self.action_space = gym.spaces.Discrete(2)
        # observation: pos in the tunnel (starting left at pos 0)
        self.observation_space = gym.spaces.Discrete(self.len_tunnel)

        # self.log = structlog.get_logger()
        self.log = structlog.get_logger(test='works')

    def reset(self):
        self.pos = 0
        self.time = 0
        return self.pos

    def step(self, action):
        self.time += 1
        reward = 0
        # left
        if action == 0:
            # at starting pos: +1 reward, but don't move
            if self.pos == 0:
                reward = 1
            else:
                self.pos -= 1
        # right
        else:
            self.pos += 1
            # reached the right end: +100 & reset to start
            if self.pos >= self.len_tunnel - 1:
                reward = 100
                self.pos = 0

        # return obs, reward, done, info as usual
        done = self.time >= self.len_episode
        print(f"{self.time=}, {self.pos=}, {reward=}, {done=}")
        # logging with structlog works!
        self.log.info('ok')
        return self.pos, reward, done, {}


class ChildTunnelEnv(TunnelEnv):
    def __init__(self, env_config):
        super().__init__(env_config)


class DummyMobileEnv(gym.Env):
    """Drastically simplified, dummy mobile env. Incrementally extended to debug the problem."""
    def __init__(self, env_config):
        # construct the rectangular world map
        self.time = 0
        self.len_tunnel = env_config['len_tunnel']
        self.episode_length = env_config['len_episode']

        # self.width = 150
        # self.height = 100
        # self.map = Polygon([(0,0), (0, self.height), (self.width, self.height), (self.width, 0)])
        # self.disable_interference = True
        # # disable interference for all BS (or not)
        # self.bs_list = []
        # for bs in self.bs_list:
        #     bs.disable_interference = self.disable_interference
        # # pass the env to all users (needed for movement; interference etc)
        # self.ue_list = []
        # for ue in self.ue_list:
        #     ue.env = self
        # # current observation
        # self.obs = None
        # dummy observations and actions
        # observation and action space are defined in the subclass --> different variants
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        """Reset environment by resetting time and all UEs (pos & movement) and their connections"""
        self.time = 0
        # TODO: this just returns the observation for the 1st UE
        # self.obs = self.get_obs(self.ue_list[0])
        return 1

    def step(self, action):
        # dummy
        print(f"{action=}")
        return 1, 0, False, {}
