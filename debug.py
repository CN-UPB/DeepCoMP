# script for debugging
from copy import copy, deepcopy

import structlog
from shapely.geometry import Point
import ray.rllib.agents.ppo as ppo

from drl_mobile.env.map import Map
from drl_mobile.env.user import User
from drl_mobile.env.station import Basestation
from drl_mobile.env.env import DatarateMobileEnv


class Test:
    def __init__(self, id, works=True):
        self.id = id
        self.works = works
        self.log = structlog.get_logger()

test = Test(1)

# for real env
map = Map(width=150, height=100)
ue1 = User('ue1', map, color='blue', pos_x='random', pos_y=40, move_x='slow')
# ue2 = User('ue2', color='red', pos_x='random', pos_y=30, move_x='fast')
bs1 = Basestation('bs1', pos=Point(50, 50))
bs2 = Basestation('bs2', pos=Point(100, 50))

# create env_config for RLlib instead
# env_config = {
#     # extra params for dummy envs
#     'len_tunnel': 5, 'len_episode': 10,
#     # real config
#     'episode_length': 10, 'width': 150, 'height': 100, 'bs_list': [bs1, bs2], 'ue_list': [ue1],
#     'dr_cutoff': 'auto', 'sub_req_dr': True, 'seed': 1234
# }

# config = ppo.DEFAULT_CONFIG.copy()
# config['num_workers'] = 1
# config['seed'] = 1234
# # shorter training for faster debugging
# config['train_batch_size'] = 200
# # config['log_level'] = 'INFO'    # default: warning
# config['env_config'] = env_config

# does this work?
copy_conf = deepcopy(ue1)
print(copy_conf)

# still works
# config['env'] = DatarateMobileEnv
# copy_conf2 = deepcopy(config)
# print(copy_conf2)
