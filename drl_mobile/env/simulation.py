import logging

import structlog
from structlog.stdlib import LoggerFactory
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
        for t in range(self.sim_time):
            self.world.plot(title=f"{t=}")
            self.world.step()
        self.world.plot(title='Final')


if __name__ == "__main__":
    # configure logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    structlog.configure(logger_factory=LoggerFactory())

    # create world and simulate
    ue1 = User('ue1', start_pos=Point(5,5), move_x=1)
    bs1 = Basestation('bs1', pos=Point(3,6), cap=1, radius=3)
    bs2 = Basestation('bs2', pos=Point(7,6), cap=1, radius=3)
    world = World(width=10, height=10, bs_list=[bs1, bs2], ue_list=[ue1])
    sim = Simulation(world, sim_time=3)

    sim.run()
