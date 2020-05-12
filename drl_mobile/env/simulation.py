import logging
import os

import gym
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


from drl_mobile.env.env import BinaryMobileEnv, DatarateMobileEnv
from drl_mobile.env.user import User
from drl_mobile.env.station import Basestation
from drl_mobile.agent.dummy import RandomAgent, FixedAgent


log = structlog.get_logger()


class Simulation:
    """Simulation class"""
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, train_steps, save_dir, plot=False):
        """Train agent for specified training steps"""
        log.info('Start training', train_steps=train_steps)
        agent.learn(train_steps)
        agent.save(f'{save_dir}/ppo2_{train_steps}')
        if plot:
            results_plotter.plot_results([save_dir], train_steps, results_plotter.X_TIMESTEPS, 'Learning Curve')
            plt.savefig(f'{save_dir}/ppo2_{train_steps}.png')
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
    gym.logger.set_level(gym.logger.ERROR)
    structlog.configure(logger_factory=LoggerFactory())

    # create the environment
    # ue1 = User('ue1', pos_x='random', pos_y=40, move_x='slow')
    ue1 = User('ue1', pos_x=20, pos_y=40, move_x=5)
    # ue2 = User('ue2', start_pos=Point(3,3), move_x=-1)
    bs1 = Basestation('bs1', pos=Point(70,50))
    bs2 = Basestation('bs2', pos=Point(130,50))
    env = DatarateMobileEnv(episode_length=20, width=200, height=100, bs_list=[bs1, bs2], ue_list=[ue1])
    env_name = type(env).__name__
    # env.seed(42)

    # create dummy agent
    # agent = RandomAgent(env.action_space, seed=1234)
    # agent = FixedAgent(action=1)
    # or create RL agent
    # for stable baselines logs
    training_dir = f'../../training/{env_name}'
    train_steps = 5000
    os.makedirs(training_dir, exist_ok=True)
    # agent = PPO2(MlpPolicy, Monitor(env, filename=f'{training_dir}'))
    # or load RL agent
    agent = PPO2.load(f'{training_dir}/ppo2_{train_steps}.zip')

    # run the simulation
    sim = Simulation(env, agent)
    # sim.train(train_steps=train_steps, save_dir=training_dir, plot=True)
    logging.getLogger('drl_mobile').setLevel(logging.INFO)
    reward = sim.run(render=True)
    log.info('Testing complete', episode_reward=reward)

    # evaluate learned policy
    logging.getLogger('drl_mobile').setLevel(logging.WARNING)
    mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=10)
    log.info("Eval. episode reward", mean_reward=mean_reward, std_reward=std_reward)
