import logging
import os

import structlog
from structlog.stdlib import LoggerFactory
from shapely.geometry import Point
# disable tf printed warning: https://github.com/tensorflow/tensorflow/issues/27045#issuecomment-480691244
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
from stable_baselines import results_plotter, PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import matplotlib.animation


from drl_mobile.env.env import MobileEnv
from drl_mobile.env.user import User
from drl_mobile.env.station import Basestation
from drl_mobile.agent.dummy import RandomAgent, FixedAgent


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
        log.info('Start training', train_steps=train_steps)
        agent.learn(train_steps)
        agent.save(f'{training_dir}/ppo2_{train_steps}')
        if plot:
            results_plotter.plot_results([training_dir], train_steps, results_plotter.X_TIMESTEPS, 'Learning Curve')
            plt.savefig(f'{training_dir}/ppo2_{train_steps}.png')
            plt.show()

    def run(self, render=False):
        """Run one simulation episode. Return episode reward."""
        patches = []
        episode_reward = 0
        done = False
        obs = self.env.reset()
        while not done:
            if render:
                patches.append(self.env.render())
            # deterministic=True is important: https://github.com/hill-a/stable-baselines/issues/832
            action, _states = self.agent.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            episode_reward += reward
        if render:
            patches.append(self.env.render())
            # FIXME: save animation as html5 video
            # fig = plt.figure(figsize=(5, 5))
            # anim = matplotlib.animation.ArtistAnimation(fig, patches, repeat=False)
            # html = anim.to_html5_video()
            # with open('replay.html', 'w') as f:
            #     f.write(html)
        return episode_reward


if __name__ == "__main__":
    # configure logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('drl_mobile').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    structlog.configure(logger_factory=LoggerFactory())

    # create the environment
    ue1 = User('ue1', start_pos=Point(20,40), move_x=5)
    # ue2 = User('ue2', start_pos=Point(3,3), move_x=-1)
    bs1 = Basestation('bs1', pos=Point(70,50))
    bs2 = Basestation('bs2', pos=Point(130,50))
    env = MobileEnv(episode_length=30, width=200, height=100, bs_list=[bs1, bs2], ue_list=[ue1])

    # create agent
    # agent = RandomAgent(env.action_space, seed=1234)
    # agent = FixedAgent(action=1)
    # agent = PPO2(MlpPolicy, Monitor(env, filename=training_dir))
    agent = PPO2.load(f'{training_dir}/ppo2_10000.zip')

    # run the simulation
    sim = Simulation(env, agent)
    # sim.train(train_steps=10000, plot=True)
    logging.getLogger('drl_mobile').setLevel(logging.INFO)
    reward = sim.run(render=True)
    log.info('Testing complete', episode_reward=reward)
