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


from drl_mobile.env.env import BinaryMobileEnv, DatarateMobileEnv, JustConnectedObsMobileEnv
from drl_mobile.env.user import User
from drl_mobile.env.station import Basestation
from drl_mobile.agent.dummy import RandomAgent, FixedAgent
from drl_mobile.util.logs import FloatRounder


log = structlog.get_logger()


class Simulation:
    """Simulation class"""
    def __init__(self, env, agent):
        self.env = env
        self.env_name = type(env).__name__
        self.agent = agent

    def train(self, train_steps, save_dir, plot=False):
        """Train agent for specified training steps"""
        log.info('Start training', train_steps=train_steps)
        agent.learn(train_steps)
        agent.save(f'{save_dir}/ppo2_{train_steps}')
        if plot:
            results_plotter.plot_results([save_dir], train_steps, results_plotter.X_TIMESTEPS,
                                         f'Learning Curve for {self.env_name}')
            plt.savefig(f'{save_dir}/ppo2_{train_steps}.png')
            plt.show()

    def run(self, render=None, save_dir=None):
        """Run one simulation episode. Render situation at beginning of each time step. Return episode reward."""
        if render is not None:
            # square figure and equal aspect ratio to avoid distortions
            fig = plt.figure(figsize=(5, 5))
            plt.gca().set_aspect('equal')

        patches = []
        episode_reward = 0
        done = False
        obs = self.env.reset()
        while not done:
            if render is not None:
                patches.append(self.env.render())
                if render == 'plot':
                    plt.show()
            # deterministic=True is important: https://github.com/hill-a/stable-baselines/issues/832
            action, _states = self.agent.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            episode_reward += reward
        if render is not None:
            patches.append(self.env.render())
            # it's either plt.show or saving the video; both doesn't work (would prob. work with plt.draw)
            if render == 'plot':
                plt.show()
            if render == 'video':
                assert save_dir is not None, 'You must specify save_dir if rendering video'
                anim = matplotlib.animation.ArtistAnimation(fig, patches, repeat=False)
                html = anim.to_html5_video()
                with open(f'{save_dir}/replay.html', 'w') as f:
                    f.write(html)
                log.info('Video saved', path=f'{save_dir}/replay.html')
        return episode_reward


if __name__ == "__main__":
    # configure logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('drl_mobile').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    gym.logger.set_level(gym.logger.ERROR)
    # structlog.configure(logger_factory=LoggerFactory())
    structlog.configure(logger_factory=LoggerFactory(),
                        processors=[
                            structlog.stdlib.filter_by_level,
                            # structlog.stdlib.add_logger_name,
                            # structlog.stdlib.add_log_level,
                            # structlog.stdlib.PositionalArgumentsFormatter(),
                            # structlog.processors.StackInfoRenderer(),
                            # structlog.processors.format_exc_info,
                            # structlog.processors.UnicodeDecoder(),
                            # structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                            # structlog_pretty.NumericRounder(digits=3),
                            FloatRounder(digits=3),
                            structlog.dev.ConsoleRenderer()
                            # structlog.stdlib.render_to_log_kwargs,
                            # structlog.processors.JSONRenderer()
                        ])

    # create the environment
    ue1 = User('ue1', pos_x='random', pos_y=40, move_x=0)
    # ue1 = User('ue1', pos_x=20, pos_y=40, move_x=5)
    # ue2 = User('ue2', start_pos=Point(3,3), move_x=-1)
    bs1 = Basestation('bs1', pos=Point(50,50))
    bs2 = Basestation('bs2', pos=Point(100,50))
    eps_length = 10
    env = DatarateMobileEnv(episode_length=eps_length, width=150, height=100, bs_list=[bs1, bs2], ue_list=[ue1])
    env.seed(42)

    # dir for saving logs, plots, replay video
    training_dir = f'../../training/{type(env).__name__}'
    os.makedirs(training_dir, exist_ok=True)
    train_steps = 10000

    # create dummy agent
    # agent = RandomAgent(env.action_space, seed=1234)
    # agent = FixedAgent(action=1)
    # or create RL agent
    # agent = PPO2(MlpPolicy, Monitor(env, filename=f'{training_dir}'))
    # or load RL agent
    agent = PPO2.load(f'{training_dir}/ppo2_{train_steps}.zip')

    # run the simulation
    sim = Simulation(env, agent)
    # sim.train(train_steps=train_steps, save_dir=training_dir, plot=True)
    logging.getLogger('drl_mobile').setLevel(logging.DEBUG)
    reward = sim.run(render='video', save_dir=training_dir)
    log.info('Simulation complete', episode_reward=reward)

    # evaluate learned policy
    logging.getLogger('drl_mobile').setLevel(logging.WARNING)
    mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=10)
    log.info("Policy evaluation", mean_eps_reward=mean_reward, std_eps_reward=std_reward,
             mean_step_reward=mean_reward/eps_length)
