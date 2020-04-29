from shapely.geometry import Point

from drl_mobile.env.world import World
from drl_mobile.env.user import User
from drl_mobile.env.station import Basestation


class Simulation:
    """Simulation class"""
    def __init__(self, world, sim_time):
        self.world = world
        self.sim_time = sim_time

    def simulate(self):
        """Run simulation loop"""
        pass


if __name__ == "__main__":
    ue = User(start_pos=Point(5,5), move_direction=(0,0))

