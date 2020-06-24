"""Main execution script used for experimentation"""

import os
import logging

import gym
import structlog
from shapely.geometry import Point
import ray
import ray.rllib.agents.ppo as ppo
# import ray.tune
import ray.rllib.agents.ppo as rllib_ppo
from ray.tune.logger import pretty_print
# DO NOT import tensorflow before ray! https://github.com/ray-project/ray/issues/8993
# disable tf printed warning: https://github.com/tensorflow/tensorflow/issues/27045#issuecomment-480691244
# import tensorflow as tf
# if hasattr(tf, 'contrib') and type(tf.contrib) != type(tf):
#     tf.contrib._warning = None
# sb imports don't work with tf2
# from stable_baselines import PPO2
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.env_checker import check_env
# from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines.bench import Monitor


from drl_mobile.env.env import BinaryMobileEnv, DatarateMobileEnv, CentralMultiUserEnv, RLlibEnv
from drl_mobile.env.simulation import Simulation
from drl_mobile.agent.dummy import RandomAgent, FixedAgent
from drl_mobile.util.logs import config_logging
from drl_mobile.rllib.env import TunnelEnv
from drl_mobile.env.user import User
from drl_mobile.env.station import Basestation
from drl_mobile.env.map import Map


log = structlog.get_logger()


# def create_env(eps_length, normalize, train, seed=None):
#     """
#     Create and return the environment with specific episode length
#     :param eps_length: Number of time steps per episode before the env resets
#     :param normalize: Whether to normalize (and clip?) observations (and rewards?)
#     :param train: Only relevant if normalize=true. If train, record new normalize stats, else load saved stats.
#     :return: The created env and the path to the training dir, based on the env name
#     """
#     ue1 = User('ue1', color='blue', pos_x='random', pos_y=40, move_x='slow')
#     # ue2 = User('ue2', color='red', pos_x='random', pos_y=30, move_x='fast')
#     bs1 = Basestation('bs1', pos=Point(50, 50))
#     bs2 = Basestation('bs2', pos=Point(100, 50))
#     # env = DatarateMobileEnv(episode_length=eps_length, width=150, height=100, bs_list=[bs1, bs2], ue_list=[ue1],
#     #                         dr_cutoff='auto', sub_req_dr=True, disable_interference=True)
#     # env = CentralMultiUserEnv(episode_length=eps_length, width=150, height=100, bs_list=[bs1, bs2], ue_list=[ue1, ue2],
#     #                           disable_interference=True)
#     # env.seed(seed)
#
#     # create env_config for RLlib instead
#     env_config = {'episode_length': eps_length, 'width': 150, 'height': 100, 'bs_list': [bs1, bs2], 'ue_list': [ue1],
#                   'dr_cutoff': 'auto', 'sub_req_dr': True, 'disable_interference': True, 'seed': seed}
#     env = env_config
#
#     # check_env(env)
#
#     # dir for saving logs, plots, replay video
#     training_dir = f'../training/{type(env).__name__}'
#     os.makedirs(training_dir, exist_ok=True)
#
#     # TODO: implement for RLlib
#     # env = Monitor(env, filename=f'{training_dir}')
#     # env = DummyVecEnv([lambda: env])
#     # normalize using running avg
#     if normalize:
#         if train:
#             # clipping is only done if normalizing (after normalization)
#             env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200, clip_reward=200)
#         else:
#             # load saved normalization stats (running avg etc)
#             env = VecNormalize.load(f'{training_dir}/vec_norm.pkl', env)
#             # disable any updates to stats during testing
#             # https://stable-baselines.readthedocs.io/en/master/guide/examples.html#pybullet-normalizing-input-features
#             env.training = False
#             env.norm_reward = False
#     return env, training_dir
#
#
# def create_agent(agent_name, env, seed=None, train=True):
#     """Create and return agent based on specified name/string"""
#     # dummy agents
#     if agent_name == 'random':
#         return RandomAgent(env.action_space, seed=seed)
#     if agent_name == 'fixed':
#         return FixedAgent(action=1, noop_interval=4)
#     # stable_baselines PPO RL agent
#     if agent_name == 'sb_ppo':
#         if train:
#             return PPO2(MlpPolicy, env, seed=seed)
#         else:
#             # load trained agent
#             return PPO2.load(f'{training_dir}/ppo2_{train_steps}.zip')
#     # RLlib PPO RL agent
#     if agent_name == 'rllib_ppo':
#         if train:
#             config = rllib_ppo.DEFAULT_CONFIG.copy()
#             config['num_workers'] = 1
#             # config['log_level'] = 'INFO'    # default: warning
#             # in case of RLlib env is the env_config
#             config['env_config'] = env
#             # FIXME: rllib tries to do a deepcopy which fails when copying some structlog code
#             return rllib_ppo.PPOTrainer(config=config, env=RLlibEnv)
#         else:   # TODO: rllib testing
#             raise NotImplementedError('Still have to implement testing with RLlib')
#     return None

def get_config(seed=None, monitor=False, train_batch_size=4000, env=RLlibEnv):
    """
    Create environment and config. Return config
    :param seed: Seed for reproducible results
    :param monitor: Whether or not to monitor and log stats
    :param train_batch_size: Number of sampled env steps in a single training iteration (default 4000)
    :param env: Environment class (not object) to use
    :return: The complete config for an RLlib agent, including the env & env_config
    """
    # create the environment
    map = Map(width=150, height=100)
    ue1 = User('ue1', map, color='blue', pos_x='random', pos_y=40, move_x='slow')
    # ue2 = User('ue2', color='red', pos_x='random', pos_y=30, move_x='fast')
    bs1 = Basestation('bs1', pos=Point(50, 50))
    bs2 = Basestation('bs2', pos=Point(100, 50))

    # create and return the config
    # TODO: for now, hard-code ppo config; make it configurable if necessary
    config = ppo.DEFAULT_CONFIG.copy()
    # 0 = no workers/actors at all --> low overhead for short debugging
    config['num_workers'] = 0
    config['seed'] = seed
    # write training stats to file under ~/ray_results (default: False)
    config['monitor'] = monitor
    config['train_batch_size'] = train_batch_size        # default: 4000; default in stable_baselines: 128
    # config['log_level'] = 'INFO'    # ray logging default: warning
    config['env'] = env
    env_config = {
        'episode_length': 10, 'map': map, 'bs_list': [bs1, bs2], 'ue_list': [ue1],
        'dr_cutoff': 'auto', 'sub_req_dr': True, 'disable_interference': True, 'seed': seed
    }
    config['env_config'] = env_config

    return config


if __name__ == "__main__":
    ray.init()
    config_logging(round_digits=3)

    # TODO: use stop dir for tune.run?
    # settings
    train_iter = 1
    # env steps per train_iter
    train_batch_size = 200
    eps_length = 10
    # train or load trained agent (& env norm stats); only set train=True for ppo agent!
    train = False
    # normalize obs (& clip? & reward?); better: use custom env normalization with dr_cutoff='auto'
    normalize = False
    # seed for agent & env
    seed = 42

    # create env
    config = get_config(seed=seed, monitor=True, train_batch_size=train_batch_size, env=RLlibEnv)

    # simulator doesn't need RLlib's env_config (contained in agent anyways)
    sim = Simulation(config=config, agent_type='ppo', normalize=normalize)

    # train
    if train:
        analysis = sim.train(train_iter)

    # simulate one run
    sim.run(config, render='video', log_steps=True)

    # evaluate
    sim.run(config, num_episodes=10, log_steps=False)
