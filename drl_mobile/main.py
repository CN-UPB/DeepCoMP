"""Main execution script used for experimentation"""

import logging
import os

import gym
import structlog
from structlog.stdlib import LoggerFactory
from shapely.geometry import Point
# disable tf printed warning: https://github.com/tensorflow/tensorflow/issues/27045#issuecomment-480691244
import tensorflow as tf
if type(tf.contrib) != type(tf):
    tf.contrib._warning = None
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


def create_env(eps_length, normalize, train):
    """
    Create and return the environment with specific episode length
    :param eps_length: Number of time steps per episode before the env resets
    :param normalize: Whether to normalize (and clip?) observations (and rewards?)
    :param train: Only relevant if normalize=true. If train, record new normalize stats, else load saved stats.
    :return: The created env and the path to the training dir, based on the env name
    """
    ue1 = User('ue1', color='blue', pos_x='random', pos_y=40, move_x=5)
    ue2 = User('ue2', color='red', pos_x='random', pos_y=30, move_x=5)
    bs1 = Basestation('bs1', pos=Point(50, 50))
    bs2 = Basestation('bs2', pos=Point(100, 50))
    env = DatarateMobileEnv(episode_length=eps_length, width=150, height=100, bs_list=[bs1, bs2], ue_list=[ue1, ue2],
                            dr_cutoff='auto', sub_req_dr=True, disable_interference=True)
    check_env(env)

    # dir for saving logs, plots, replay video
    training_dir = f'../training/{type(env).__name__}'
    os.makedirs(training_dir, exist_ok=True)

    env = Monitor(env, filename=f'{training_dir}')
    env = DummyVecEnv([lambda: env])
    # normalize using running avg
    if normalize:
        if train:
            # clipping is only done if normalizing (after normalization)
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200, clip_reward=200)
        else:
            # load saved normalization stats (running avg etc)
            env = VecNormalize.load(f'{training_dir}/vec_norm.pkl', env)
            # disable any updates to stats during testing
            # https://stable-baselines.readthedocs.io/en/master/guide/examples.html#pybullet-normalizing-input-features
            env.training = False
            env.norm_reward = False
    return env, training_dir


def create_agent(agent_name, env, seed=None, train=True):
    """Create and return agent based on specified name/string"""
    # dummy agents
    if agent_name == 'random':
        return RandomAgent(env.action_space, seed=seed)
    if agent_name == 'fixed':
        return FixedAgent(action=1)
    # PPO RL agent
    if agent_name == 'ppo':
        if train:
            return PPO2(MlpPolicy, env, seed=seed)
        else:
            # load trained agent
            return PPO2.load(f'{training_dir}/ppo2_{train_steps}.zip')
    return None


if __name__ == "__main__":
    config_logging(round_digits=3)
    # settings
    train_steps = 5000
    eps_length = 10
    # train or load trained agent (& env norm stats); only set train=True for ppo agent!
    train = False
    # normalize obs (& clip? & reward?); better: use custom env normalization with dr_cutoff='auto'
    normalize = False
    # seed for agent & env
    seed = 42

    # create env
    env, training_dir = create_env(eps_length=eps_length, normalize=normalize, train=train)
    env.seed(seed)

    agent = create_agent('fixed', env, seed=seed, train=train)
    sim = Simulation(env, agent, normalize=normalize)

    # train
    if train:
        sim.train(train_steps=train_steps, save_dir=training_dir, plot=True)

    # simulate one run
    logging.getLogger('drl_mobile').setLevel(logging.DEBUG)
    sim.run(render='video', save_dir=training_dir)

    # evaluate
    logging.getLogger('drl_mobile').setLevel(logging.WARNING)
    # sim.evaluate(eval_eps=10)
