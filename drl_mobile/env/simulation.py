import logging
import os

import structlog
from structlog.stdlib import LoggerFactory
from shapely.geometry import Point
from stable_baselines import results_plotter, PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

from drl_mobile.env.env import MobileEnv
from drl_mobile.env.user import User
from drl_mobile.env.station import Basestation
from drl_mobile.agent.random import RandomAgent


log = structlog.get_logger()
# for stable baselines logs
training_dir = '../../training'


class Simulation:
    """Simulation class"""
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, train_steps, plot=False):
        """Train agent for specified training steps"""
        log.info('Start training', train_steps=train_steps, env=self.env, agent=self.agent)
        agent.learn(train_steps)
        agent.save(f'../../{training_dir}/ppo2_{train_steps}')
        if plot:
            results_plotter.plot_results([training_dir], train_steps, results_plotter.X_TIMESTEPS, 'Learning Curve')
            plt.savefig(f'{training_dir}/ppo2_{train_steps}.png')
            plt.show()

    def run(self, render=False):
        """Run one simulation episode. Return episode reward."""
        reward = 0
        done = False
        obs = self.env.reset()
        while not done:
            # deterministic=True is important: https://github.com/hill-a/stable-baselines/issues/832
            action, _states = self.agent.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            if render:
                self.env.render()
        return reward


if __name__ == "__main__":
    # configure logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    structlog.configure(logger_factory=LoggerFactory())

    # create the environment
    ue1 = User('ue1', start_pos=Point(2,5), move_x=1)
    # ue2 = User('ue2', start_pos=Point(3,4), move_x=-1)
    bs1 = Basestation('bs1', pos=Point(3,6), cap=1, radius=3)
    bs2 = Basestation('bs2', pos=Point(7,6), cap=1, radius=3)
    env = MobileEnv(episode_length=10, width=10, height=10, bs_list=[bs1, bs2], ue_list=[ue1])

    # create agent
    # agent = RandomAgent(env.action_space, seed=1234)
    # agent = PPO2(MlpPolicy, Monitor(env, filename=training_dir))
    agent = PPO2.load(f'{training_dir}/ppo2_10000.zip')

    # run the simulation
    sim = Simulation(env, agent)
    # sim.train(train_steps=10000, plot=True)
    reward = sim.run(render=True)
    log.info('Testing complete', reward=reward)
