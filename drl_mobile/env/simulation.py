import structlog
from shapely.geometry import Point

from drl_mobile.env.world import World
from drl_mobile.env.user import User
from drl_mobile.env.station import Basestation


log = structlog.get_logger()


class Simulation:
    """Simulation class"""
    def __init__(self, world, sim_time):
        self.world = world
        self.sim_time = sim_time

    def run(self):
        """Run simulation loop"""
        self.world.plot()
        # for t in range(self.sim_time):
        #     pass


if __name__ == "__main__":
    ue1 = User(start_pos=Point(5,5), move_x=1)
    bs1 = Basestation(pos=Point(3,6), cap=1, radius=3)
    bs2 = Basestation(pos=Point(7,6), cap=1, radius=3)
    world = World(width=10, height=10, bs_list=[bs1, bs2], ue_list=[ue1])
    sim = Simulation(world, sim_time=10)

    sim.run()
