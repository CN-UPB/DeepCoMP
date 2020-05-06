import logging

import structlog
from structlog.stdlib import LoggerFactory
from shapely.geometry import Point

from drl_mobile.env.env import MobileEnv
from drl_mobile.env.user import User
from drl_mobile.env.station import Basestation
from drl_mobile.agent.random import RandomAgent


log = structlog.get_logger()


class Simulation:
    """Simulation class"""
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run(self, render=False):
        """Run simulation loop"""
        done = False
        obs = self.env.reset()
        while not done:
            action = self.agent.predict(obs)
            obs, reward, done, info = self.env.step(action)
            if render:
                self.env.render()


if __name__ == "__main__":
    # configure logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    structlog.configure(logger_factory=LoggerFactory())

    # create the environment
    ue1 = User('ue1', start_pos=Point(2,5), move_x=1)
    # ue2 = User('ue2', start_pos=Point(3,4), move_x=-1)
    bs1 = Basestation('bs1', pos=Point(3,6), cap=1, radius=3)
    bs2 = Basestation('bs2', pos=Point(7,6), cap=1, radius=3)
    env = MobileEnv(episode_length=5, width=10, height=10, bs_list=[bs1, bs2], ue_list=[ue1])

    # setup and run the simulation
    agent = RandomAgent(env.action_space, seed=1234)
    sim = Simulation(env, agent)

    sim.run(render=True)
