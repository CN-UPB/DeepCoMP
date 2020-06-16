# minimal working env with RLlib. To be extended to what I have/want
import gym
import gym.spaces


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
        return self.pos, reward, done, {}
