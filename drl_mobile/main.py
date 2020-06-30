"""Main execution script used for experimentation"""
import structlog
from shapely.geometry import Point
from ray.rllib.agents.ppo import DEFAULT_CONFIG

from drl_mobile.env.single_ue.variants import BinaryMobileEnv, DatarateMobileEnv
from drl_mobile.env.multi_ue.central import CentralMultiUserEnv, CentralRemainingDrEnv
from drl_mobile.env.multi_ue.multi_agent import MultiAgentMobileEnv
from drl_mobile.util.simulation import Simulation
from drl_mobile.util.logs import config_logging
from drl_mobile.env.entities.user import User
from drl_mobile.env.entities.station import Basestation
from drl_mobile.env.entities.map import Map


log = structlog.get_logger()


def create_env_config(eps_length, num_workers=1, train_batch_size=1000, seed=None):
    """
    Create environment and RLlib config. Return config.
    :param eps_length: Number of time steps per episode (parameter of the environment)
    :param num_workers: Number of RLlib workers for training. For longer training, num_workers = cpu_cores-1 makes sense
    :param train_batch_size: Number of sampled env steps in a single training iteration
    :param seed: Seed for reproducible results
    :return: The complete config for an RLlib agent, including the env & env_config
    """
    # create the environment and env_config
    map = Map(width=150, height=100)
    ue1 = User('ue1', map, color='blue', pos_x='random', pos_y=40, move_x='slow')
    ue2 = User('ue2', map, color='red', pos_x='random', pos_y=30, move_x='fast')
    ue_list = [ue1, ue2]
    bs1 = Basestation('bs1', pos=Point(50, 50))
    bs2 = Basestation('bs2', pos=Point(100, 50))
    bs_list = [bs1, bs2]
    env_class = MultiAgentMobileEnv

    env_config = {
        'episode_length': eps_length, 'map': map, 'bs_list': bs_list, 'ue_list': ue_list, 'dr_cutoff': 'auto',
        'sub_req_dr': True, 'seed': seed
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
    # TODO: necessary to disable for single-agent envs?
    # for now, all UEs use the same policy (and NN?)
    # TODO: does this mean they all use the same NN or different NNs with the same policy? I guess the same one
    # instantiate env to access obs and action space
    env = env_class(env_config)
    config['multiagent'] = {
        'policies': {
            'ue': (
                None,
                env.observation_space,
                env.action_space,
                {}
            )
        },
        'policy_mapping_fn': lambda agent_id: 'ue'
    }

    return config


if __name__ == "__main__":
    config_logging(round_digits=3)

    # settings
    # stop training when any of the criteria is met
    stop_criteria = {
        'training_iteration': 1,
        # 'episode_reward_mean': 250
    }
    # train or load trained agent; only set train=True for ppo agent
    train = True
    agent_name = 'ppo'
    # name of the RLlib dir to load the agent from for testing
    agent_path = '../training/PPO/PPO_CentralRemainingDrEnv_0_2020-06-26_15-32-50raq5e_od/checkpoint_30/checkpoint-30'
    # seed for agent & env
    seed = 42

    # create RLlib config (with env inside) & simulator
    config = create_env_config(eps_length=10, num_workers=1, train_batch_size=200, seed=seed)
    sim = Simulation(config=config, agent_name=agent_name, debug=False)

    # train
    if train:
        agent_path, analysis = sim.train(stop_criteria)

    # load & test agent
    sim.load_agent(rllib_path=agent_path, rand_seed=seed, fixed_action=[1, 1])
    # simulate one episode and render
    sim.run(render='gif', log_steps=True)
    # evaluate over multiple episodes
    # sim.run(num_episodes=30, log_steps=False)
