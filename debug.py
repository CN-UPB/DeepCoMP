# script for debugging
from copy import copy, deepcopy

import structlog
from shapely.geometry import Point
import ray.rllib.agents.ppo as ppo

from drl_mobile.env.entities.map import Map
from drl_mobile.env.entities.user import User
from drl_mobile.env.util.movement import UniformMovement
from drl_mobile.util.env_setup import create_small_map


map, bs_list = create_small_map()
ue = User(1, map, pos_x='random', pos_y='random', movement=UniformMovement(map))

print(ue.priority)

ue2 = deepcopy(ue)
print(ue2.priority)

