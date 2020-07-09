import random

import structlog
from shapely.geometry import Point
import matplotlib.pyplot as plt

from drl_mobile.env.util.utility import step_utility, log_utility


class User:
    """
    A user/UE moving around in the world and requesting mobile services
    Connection to BS are checked before connecting and after every move to check if connection is lost or still stable
    """
    def __init__(self, id, map, pos_x, pos_y, movement, dr_req=1):
        """
        Create new UE object
        :param id: Unique ID of UE (string)
        :param map: Map object representing the playground/world
        :param pos_x: x-coord of starting position or 'random'
        :param pos_y: y-coord of starting position or 'random'
        :param movement: Movement utility object implementing the movement of the UE
        :param dr_req: Data rate requirement by UE for successful service
        """
        self.id = id
        self.map = map
        self.movement = movement
        self.dr_req = dr_req
        self.conn_bs = []

        self.init_pos_x = pos_x
        self.init_pos_y = pos_y
        self.pos = None
        self.reset_pos()
        self.movement.reset()

        self.log = structlog.get_logger(id=self.id, pos=str(self.pos), move=self.movement,
                                        conn_bs=self.conn_bs, dr_req=self.dr_req)
        self.log.info('UE init')

    def __repr__(self):
        return str(self.id)

    @property
    def curr_dr(self):
        """Current data rate the UE gets through all its BS connections"""
        dr = 0
        for bs in self.conn_bs:
            dr += bs.data_rate(self)
        self.log.debug("Current data rate", curr_dr=dr)
        return dr

    @property
    def dr_req_satisfied(self):
        """Whether or not the UE's data rate requirement is satisfied by its current total data rate"""
        return self.curr_dr >= self.dr_req

    @property
    def utility(self):
        """Utility based on the current data rate and utility function"""
        # return step_utility(self.curr_dr, self.dr_req)
        return log_utility(self.curr_dr)

    def plot(self, radius=3):
        """
        Plot the UE as filled circle with a given radius and the ID. Green if demand satisfied, else orange.
        :param radius: Radius of the circle
        :return: A list of created matplotlib artists
        """
        curr_dr = self.curr_dr
        color = 'orange'
        if curr_dr >= self.dr_req:
            color = 'green'

        artists = plt.plot(*self.pos.buffer(radius).exterior.xy, color=color)
        artists.extend(plt.fill(*self.pos.buffer(radius).exterior.xy, color=color))
        artists.append(plt.annotate(self.id, xy=(self.pos.x, self.pos.y), ha='center', va='center'))

        # show curr data rate and utility below the UE
        artists.append(plt.annotate(f'dr: {curr_dr:.2f}', xy=(self.pos.x, self.pos.y -radius -2),
                                    ha='center', va='center'))
        artists.append(plt.annotate(f'util: {self.utility:.2f}', xy=(self.pos.x, self.pos.y -radius -6),
                                    ha='center', va='center'))
        return artists

    def reset_pos(self):
        """(Re)set position based on initial position x and y as Point. Resolve 'random'."""
        # set pos_x
        pos_x = self.init_pos_x
        if pos_x == 'random':
            pos_x = random.randint(0, self.map.width)
        # set pos_y
        pos_y = self.init_pos_y
        if pos_y == 'random':
            pos_y = random.randint(0, self.map.height)
        # set pos as Point
        self.pos = Point(pos_x, pos_y)

    def reset(self):
        """Reset UE to initial position and movement. Disconnect from all BS."""
        self.reset_pos()
        self.movement.reset()
        self.conn_bs = []

    def move(self):
        """
        Do one step according to movement object and update position

        :return: Number of connections lost through movement
        """
        self.pos = self.movement.step(self.pos)

        num_lost_connections = self.check_bs_connection()
        self.log = self.log.bind(pos=str(self.pos), move=self.movement)
        self.log.debug("User move", lost_connections=num_lost_connections)
        return num_lost_connections

    def check_bs_connection(self):
        """
        Check if assigned BS connections are still stable (after move), else remove.
        :return: Number of removed/lost connections
        """
        remove_bs = []
        for bs in self.conn_bs:
            if not bs.can_connect(self.pos):
                self.log.info("Losing connection to BS", bs=bs)
                remove_bs.append(bs)
        # remove/disconnect bs
        for bs in remove_bs:
            self.disconnect_from_bs(bs)
        return len(remove_bs)

    def connect_to_bs(self, bs, disconnect=False):
        """
        Try to connect to specified basestation. Return if successful.
        :param bs: Basestation to connect to
        :param disconnect: If True, disconnect from BS if it was previously connected.
        :return: True if (dis-)connected successfully. False if out of range.
        """
        log = self.log.bind(bs=bs, disconnect=disconnect, conn_bs=self.conn_bs)
        # already connected
        if bs in self.conn_bs:
            if disconnect:
                self.disconnect_from_bs(bs)
                log.info("Disconnected")
            else:
                log.info("Staying connected")
            return True
        # not yet connected
        if bs.can_connect(self.pos):
            self.conn_bs.append(bs)
            bs.conn_ues.append(self)
            log.info("Connected", conn_bs=self.conn_bs)
            self.log = self.log.bind(conn_bs=self.conn_bs)
            return True
        else:
            # log.info("Cannot connect")
            return False

    def disconnect_from_bs(self, bs):
        """Disconnect from given BS. Assume BS is currently connected."""
        assert bs in self.conn_bs, "Not connected to BS --> Cannot disconnect"
        self.conn_bs.remove(bs)
        bs.conn_ues.remove(self)
        self.log = self.log.bind(conn_bs=self.conn_bs)
