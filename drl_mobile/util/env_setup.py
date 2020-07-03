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


def get_env_class(env_type):
    """Return the env class corresponding to the string type (from CLI)"""
    allowed_types = ('single', 'central', 'multi')
    assert env_type in allowed_types, f"Environment type was {env_type} but has to be one of {allowed_types}."

    if env_type == 'single':
        return DatarateMobileEnv
    if env_type == 'central':
        return CentralMultiUserEnv
    if env_type == 'multi':
        return MultiAgentMobileEnv


def create_small_env():
    """
    Create small env with 2 UEs and 2 BS

    :returns: tuple (map, ue_list, bs_list)
    """
    map = Map(width=150, height=100)
    ue1 = User(1, map, pos_x='random', pos_y=40, move_x='slow')
    ue2 = User(2, map, pos_x='random', pos_y=30, move_x='fast')
    ue_list = [ue1, ue2]
    bs1 = Basestation(1, pos=Point(50, 50))
    bs2 = Basestation(2, pos=Point(100, 50))
    bs_list = [bs1, bs2]
    return map, ue_list, bs_list


def create_medium_env():
    """
    Same as large env, but with just 3 UEs and map restricted to areas with coverage.
    Thus, optimal episode reward should be close to num_ues * eps_length * 10 (ie, all UEs are always connected)
    """
    map = Map(width=205, height=85)
    ue1 = User(1, map, pos_x='random', pos_y=25, move_x='slow')
    ue2 = User(2, map, pos_x='random', pos_y=45, move_x='fast')
    ue3 = User(3, map, pos_x=130, pos_y='random', move_y='slow')
    ue_list = [ue1, ue2, ue3]

    bs1 = Basestation(1, pos=Point(45, 35))
    bs2 = Basestation(2, pos=Point(160, 35))
    bs3 = Basestation(3, pos=Point(100, 85))
    bs_list = [bs1, bs2, bs3]
    return map, ue_list, bs_list


def create_large_env():
    """
    Create larger env with 5 UEs and 3 BS and return UE & BS list

    :returns: Tuple(map, ue_list, bs_list)
    """
    map = Map(width=205, height=165)
    ue_list = [
        User(1, map, pos_x='random', pos_y=60, move_x='fast'),
        User(2, map, pos_x='random', pos_y=50, move_x='fast'),
        User(3, map, pos_x=130, pos_y='random', move_y='slow'),
        User(4, map, pos_x=60, pos_y='random', move_y='slow')
        # User(5, map, pos_x='random', pos_y='random', move_x='fast', move_y='fast')
    ]

    bs1 = Basestation(1, pos=Point(45, 70))
    bs2 = Basestation(2, pos=Point(160, 70))
    bs3 = Basestation(3, pos=Point(100, 120))
    bs_list = [bs1, bs2, bs3]
    return map, ue_list, bs_list


# TODO: at some point probably more convenient to specify num UEs and num BS and generate an env randomly
def get_env(env_str):
    """Create and return the environment corresponding to the given env_str"""
    allowed_envs = ('small', 'medium', 'large')
    assert env_str in allowed_envs, f"Environment {env_str} is not one of {allowed_envs}."

    if env_str == 'small':
        return create_small_env()
    if env_str == 'medium':
        return create_medium_env()
    if env_str == 'large':
        return create_large_env()


def create_env_config(agent, env, eps_length, num_workers=1, train_batch_size=1000, seed=None, agents_share_nn=True):
    """
    Create environment and RLlib config. Return config.

    :param agent: String indicating which environment version to use based on the agent type
    :param env: Size of the environment (as string)
    :param eps_length: Number of time steps per episode (parameter of the environment)
    :param num_workers: Number of RLlib workers for training. For longer training, num_workers = cpu_cores-1 makes sense
    :param train_batch_size: Number of sampled env steps in a single training iteration
    :param seed: Seed for reproducible results
    :param agents_share_nn: Whether all agents in a multi-agent env should share the same NN or have separate copies
    :return: The complete config for an RLlib agent, including the env & env_config
    """
    env_class = get_env_class(agent)
    map, ue_list, bs_list = get_env(env)

    env_config = {
        'episode_length': eps_length, 'seed': seed,
        'map': map, 'bs_list': bs_list, 'ue_list': ue_list, 'dr_cutoff': 'auto', 'sub_req_dr': True,
        'curr_dr_obs': True, 'ues_at_bs_obs': False
    }

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
                'policies': {ue.id: (None, env.observation_space, env.action_space, {}) for ue in ue_list},
                'policy_mapping_fn': lambda agent_id: agent_id
            }

    return config

