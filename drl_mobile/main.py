"""Main execution script used for experimentation"""

import structlog
from shapely.geometry import Point
import ray
import ray.rllib.agents.ppo as ppo

from drl_mobile.env.env import BinaryMobileEnv, DatarateMobileEnv, CentralMultiUserEnv
from drl_mobile.env.simulation import Simulation
from drl_mobile.agent.dummy import RandomAgent, FixedAgent
from drl_mobile.util.logs import config_logging
from drl_mobile.env.user import User
from drl_mobile.env.station import Basestation
from drl_mobile.env.map import Map


log = structlog.get_logger()


def create_env_config(env, eps_length, train_batch_size=1000, seed=None):
    """
    Create environment and RLlib config. Return config.
    :param env: Environment class (not object) to use
    :param eps_length: Number of time steps per episode (parameter of the environment)
    :param train_batch_size: Number of sampled env steps in a single training iteration
    :param seed: Seed for reproducible results
    :return: The complete config for an RLlib agent, including the env & env_config
    """
    # create the environment and env_config
    map = Map(width=150, height=100)
    ue1 = User('ue1', map, color='blue', pos_x='random', pos_y=40, move_x='slow')
    ue2 = User('ue2', map, color='red', pos_x='random', pos_y=30, move_x='fast')
    bs1 = Basestation('bs1', pos=Point(50, 50))
    bs2 = Basestation('bs2', pos=Point(100, 50))

    env_config = {
        'episode_length': eps_length, 'map': map, 'bs_list': [bs1, bs2], 'ue_list': [ue1, ue2],
        'dr_cutoff': 'auto', 'sub_req_dr': True, 'disable_interference': True, 'seed': seed
    }

    # create and return the config
    config = ppo.DEFAULT_CONFIG.copy()
    # 0 = no workers/actors at all --> low overhead for short debugging
    config['num_workers'] = 1
    config['seed'] = seed
    # write training stats to file under ~/ray_results (default: False)
    config['monitor'] = True
    config['train_batch_size'] = train_batch_size        # default: 4000; default in stable_baselines: 128
    # config['log_level'] = 'INFO'    # ray logging default: warning
    config['env'] = env
    config['env_config'] = env_config

    return config


if __name__ == "__main__":
    config_logging(round_digits=3)

    # settings
    # stop training when any of the criteria is met
    stop_criteria = {
        'training_iteration': 20,
        # 'episode_reward_mean': 250
    }
    # train or load trained agent; only set train=True for ppo agent
    train = False
    agent_name = 'ppo'
    # name of the RLlib dir to load the agent from for testing
    agent_path = '../training/PPO/PPO_CentralMultiUserEnv_0_2020-06-24_16-42-21tp6f0w12/checkpoint_20/checkpoint-20'
    # seed for agent & env
    seed = 42

    # create env and RLlib config
    config = create_env_config(CentralMultiUserEnv, eps_length=30, train_batch_size=1000, seed=seed)

    # simulator doesn't need RLlib's env_config (contained in agent anyways)
    sim = Simulation(config=config, agent_name=agent_name)

    # train
    if train:
        analysis = sim.train(stop_criteria)
    # test
    # TODO: currently I need to get the path of the trained agent manually and load it before testing.
    #  it should be possible to train and directly test
    else:
        sim.load_agent(path=agent_path, seed=seed)
        # simulate one run
        sim.run(render='video', log_steps=True)
        # evaluate
        sim.run(num_episodes=10, log_steps=False)
