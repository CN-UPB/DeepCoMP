"""Utility module for setting up different envs"""
from shapely.geometry import Point
from ray.rllib.agents.ppo import DEFAULT_CONFIG
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from drl_mobile.env.single_ue.variants import BinaryMobileEnv, DatarateMobileEnv
from drl_mobile.env.multi_ue.central import CentralMultiUserEnv
from drl_mobile.env.multi_ue.multi_agent import MultiAgentMobileEnv
from drl_mobile.env.entities.user import User
from drl_mobile.env.entities.station import Basestation
from drl_mobile.env.entities.map import Map


def create_small_env():
    """Create and return small env with 2 UEs and 2 BS"""
    map = Map(width=150, height=100)
    ue1 = User(1, map, pos_x='random', pos_y=40, move_x='slow')
    ue2 = User(2, map, pos_x='random', pos_y=30, move_x='fast')
    ue_list = [ue1, ue2]
    bs1 = Basestation(1, pos=Point(50, 50))
    bs2 = Basestation(2, pos=Point(100, 50))
    bs_list = [bs1, bs2]

    env_config = {
        'map': map, 'bs_list': bs_list, 'ue_list': ue_list, 'dr_cutoff': 'auto', 'sub_req_dr': True,
        'curr_dr_obs': True, 'ues_at_bs_obs': False
    }
    return env_config


def create_large_env():
    """Create and return larger env with 5 UEs and 3 BS"""
    map = Map(width=205, height=165)
    # ue1 = User(1, map, pos_x='random', pos)
    # TODO:


def create_env_config(eps_length, num_workers=1, train_batch_size=1000, seed=None, agents_share_nn=True):
    """
    Create environment and RLlib config. Return config.

    :param eps_length: Number of time steps per episode (parameter of the environment)
    :param num_workers: Number of RLlib workers for training. For longer training, num_workers = cpu_cores-1 makes sense
    :param train_batch_size: Number of sampled env steps in a single training iteration
    :param seed: Seed for reproducible results
    :param agents_share_nn: Whether all agents in a multi-agent env should share the same NN or have separate copies
    :return: The complete config for an RLlib agent, including the env & env_config
    """
    env_class = MultiAgentMobileEnv

    env_config = create_small_env()
    env_config['episode_length'] = eps_length
    env_config['seed'] = seed

    # create and return the config
    config = DEFAULT_CONFIG.copy()
    # 0 = no workers/actors at all --> low overhead for short debugging; 2+ workers to accelerate long training
    config['num_workers'] = num_workers
    config['seed'] = seed
    # write training stats to file under ~/ray_results (default: False)
    config['monitor'] = True
    config['train_batch_size'] = train_batch_size        # default: 4000; default in stable_baselines: 128
    # configure the size of the neural network's hidden layers
    # config['model']['fcnet_hiddens'] = [100, 100]
    # config['log_level'] = 'INFO'    # ray logging default: warning
    config['env'] = env_class
    config['env_config'] = env_config

    # for multi-agent env: https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical
    if MultiAgentEnv in env_class.__mro__:
        # instantiate env to access obs and action space
        env = env_class(env_config)

        # all UEs use the same policy and NN
        if agents_share_nn:
            config['multiagent'] = {
                'policies': {'ue': (None, env.observation_space, env.action_space, {})},
                'policy_mapping_fn': lambda agent_id: 'ue'
            }
        # or: use separate policies (and NNs) for each agent
        else:
            config['multiagent'] = {
                'policies': {ue.id: (None, env.observation_space, env.action_space, {}) for ue in env_config['ue_list']},
                'policy_mapping_fn': lambda agent_id: agent_id
            }

    return config

