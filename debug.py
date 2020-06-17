# script for debugging
from copy import copy, deepcopy

from shapely.geometry import Point

from drl_mobile.env.user import User
from drl_mobile.env.station import Basestation
from drl_mobile.rllib.env import ChildTunnelEnv, DummyMobileEnv, TunnelEnv
from drl_mobile.env.env import RLlibEnv


# for real env
ue1 = User('ue1', color='blue', pos_x='random', pos_y=40, move_x='slow')
# ue2 = User('ue2', color='red', pos_x='random', pos_y=30, move_x='fast')
bs1 = Basestation('bs1', pos=Point(50, 50))
bs2 = Basestation('bs2', pos=Point(100, 50))

# create env_config for RLlib instead
env_config = {
    # extra params for dummy envs
    'len_tunnel': 5, 'len_episode': 10,
    # real config
    'episode_length': 10, 'width': 150, 'height': 100, 'bs_list': [bs1, bs2], 'ue_list': [ue1],
    'dr_cutoff': 'auto', 'sub_req_dr': True, 'disable_interference': True, 'seed': 1234
}

env = RLlibEnv(env_config)
print(env)

# copy seems to work for (child)tunnelenv, but not deepcopy
copy_env = copy(env)
print(copy_env)

# deepcopy works for DummyMobileEnv, which doesn't use structlog at all
deepcopy_env = deepcopy(env)
print(deepcopy_env)
