"""Main execution script used for experimentation"""

import logging
import os

import gym
import structlog
from structlog.stdlib import LoggerFactory
from shapely.geometry import Point
# disable tf printed warning: https://github.com/tensorflow/tensorflow/issues/27045#issuecomment-480691244
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.bench import Monitor

from drl_mobile.env.env import BinaryMobileEnv, DatarateMobileEnv, JustConnectedObsMobileEnv
from drl_mobile.env.user import User
from drl_mobile.env.station import Basestation
from drl_mobile.env.simulation import Simulation
from drl_mobile.agent.dummy import RandomAgent, FixedAgent
from drl_mobile.util.logs import FloatRounder


log = structlog.get_logger()


def config_logging(round_digits):
    """Configure logging using structlog, stdlib logging, and custom FloatRounder to round to spec numb digits"""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('drl_mobile').setLevel(logging.WARNING)
    logging.getLogger('drl_mobile.env.simulation').setLevel(logging.INFO)
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
                            FloatRounder(digits=round_digits, not_fields=['sinr', 'signal', 'interference']),
                            structlog.dev.ConsoleRenderer()
                            # structlog.stdlib.render_to_log_kwargs,
                            # structlog.processors.JSONRenderer()
                        ])


def create_env(eps_length):
    """Create and return the environment with specific episode length"""
    # ue1 = User('ue1', pos_x='random', pos_y=40, move_x=0)
    ue1 = User('ue1', pos_x=20, pos_y=40, move_x=5)
    # ue2 = User('ue2', start_pos=Point(3,3), move_x=-1)
    bs1 = Basestation('bs1', pos=Point(50,50))
    bs2 = Basestation('bs2', pos=Point(100,50))
    env = DatarateMobileEnv(episode_length=eps_length, width=150, height=100, bs_list=[bs1, bs2], ue_list=[ue1],
                             dr_cutoff=200, sub_req_dr=True, disable_interference=True)
    check_env(env)
    return env


def create_agent(agent_name, train=True):
    """Create and return agent based on specified name/string"""
    # dummy agents
    if agent_name == 'random':
        return RandomAgent(env.action_space, seed=1234)
    if agent_name == 'fixed':
        return FixedAgent(action=1)
    # PPO RL agent
    if agent_name == 'ppo':
        if train:
            return PPO2(MlpPolicy, Monitor(env, filename=f'{training_dir}'))
        else:
            # load trained agent
            return PPO2.load(f'{training_dir}/ppo2_{train_steps}.zip')
    return None


if __name__ == "__main__":
    config_logging(round_digits=3)
    env = create_env(eps_length=10)
    env.seed(42)

    # FIXME: try to normalize observations automatically
    # env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200, clip_reward=200)

    # dir for saving logs, plots, replay video
    training_dir = f'../training/{type(env).__name__}'
    os.makedirs(training_dir, exist_ok=True)
    train_steps = 2000

    train = False
    agent = create_agent('fixed', train=train)
    sim = Simulation(env, agent)

    # train
    if train:
        sim.train(train_steps=train_steps, save_dir=training_dir, plot=True)

    # simulate one run
    logging.getLogger('drl_mobile').setLevel(logging.DEBUG)
    sim.run(render='video', save_dir=training_dir)

    # evaluate
    logging.getLogger('drl_mobile').setLevel(logging.WARNING)
    # sim.evaluate(eval_eps=10)
