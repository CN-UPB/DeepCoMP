import ray.rllib.agents.ppo as ppo
from shapely.geometry import Point

from drl_mobile.env.user import User
from drl_mobile.env.station import Basestation
from drl_mobile.env.map import Map
from drl_mobile.rllib.env import TunnelEnv, DummyMobileEnv, ChildTunnelEnv
from drl_mobile.env.env import RLlibEnv


# dummy function with hard-coded env to get sth simple working
def create_rllib_agent(seed=None, train=True):
    if train:
        config = ppo.DEFAULT_CONFIG.copy()
        config['num_workers'] = 1
        config['seed'] = seed
        # shorter training for faster debugging
        config['train_batch_size'] = 200
        # config['log_level'] = 'INFO'    # default: warning

        # TODO: avoid hard-coding env here
        # for real env
        map = Map(width=150, height=100)
        ue1 = User('ue1', map, color='blue', pos_x='random', pos_y=40, move_x='slow')
        # ue2 = User('ue2', color='red', pos_x='random', pos_y=30, move_x='fast')
        bs1 = Basestation('bs1', pos=Point(50, 50))
        bs2 = Basestation('bs2', pos=Point(100, 50))

        # create env_config for RLlib instead
        env_config = {
            # extra params for dummy envs
            'len_tunnel': 5, 'len_episode': 10,
            # real config
            'episode_length': 10, 'map': map, 'bs_list': [bs1, bs2], 'ue_list': [ue1],
            'dr_cutoff': 'auto', 'sub_req_dr': True, 'disable_interference': True, 'seed': seed
        }
        config['env_config'] = env_config
        return ppo.PPOTrainer(config=config, env=RLlibEnv)
    else:   # TODO: rllib testing
        raise NotImplementedError('Still have to implement testing with RLlib')