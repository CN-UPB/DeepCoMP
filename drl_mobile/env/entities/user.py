import random

import structlog
from shapely.geometry import Point
import matplotlib.pyplot as plt


class User:
    """
    A user/UE moving around in the world and requesting mobile services
    Connection to BS are checked before connecting and after every move to check if connection is lost or still stable
    """
    def __init__(self, id, map, pos_x, pos_y, move_x=0, move_y=0, dr_req=1, color='blue'):
        """
        Create new UE object
        :param id: Unique ID of UE (string)
        :param map: Map object representing the playground/world
        :param pos_x: x-coord of starting position or 'random'
        :param pos_y: y-coord of starting position or 'random'
        :param move_x: Movement per step along x-axis. Number or 'slow' -> randint(1,5) or 'fast' -> randint(10,20).
        :param move_y: Movement per step along y-axis. Number or 'slow' -> randint(1,5) or 'fast' -> randint(10,20).
        :param dr_req: Data rate requirement by UE for successful service
        :param color: Color for rendering. Default: blue
        """
        self.id = id
        self.map = map
        self.dr_req = dr_req
        self.conn_bs = []
        self.color = color

        self.init_pos_x = pos_x
        self.init_pos_y = pos_y
        self.pos = None
        self.reset_pos()

        self.init_move_x = move_x
        self.init_move_y = move_y
        self.move_x = None
        self.move_y = None
        self.reset_movement()

        self.log = structlog.get_logger(id=self.id, pos=str(self.pos), move=(self.move_x, self.move_y),
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

        # show curr data rate below the UE
        artists.append(plt.annotate(f'dr: {curr_dr:.2f}', xy=(self.pos.x, self.pos.y -radius -2),
                                    ha='center', va='center', fontweight='light'))
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

    def reset_movement(self):
        """(Re)set movement based on provided init movement. Resolve 'slow' or 'fast'."""
        if self.init_move_x == 'slow':
            self.move_x = random.randint(1, 5)
        elif self.init_move_x == 'fast':
            self.move_x = random.randint(10, 20)
        else:
            # assume init_move_x was a specific number for how to move
            self.move_x = self.init_move_x
        # same for move_y
        if self.init_move_y == 'slow':
            self.move_y = random.randint(1, 5)
        elif self.init_move_y == 'fast':
            self.move_y = random.randint(10, 20)
        else:
            # assume init_move_y was a specific number for how to move
            self.move_y = self.init_move_y

    def reset(self):
        """Reset UE to initial position and movement. Disconnect from all BS."""
        self.reset_pos()
        self.reset_movement()
        self.conn_bs = []

    def move(self):
        """
        Do one step in movement direction and update position
        Reverse movement direction to avoid moving out of the map
        :return: Number of connections lost through movement
        """
        # seems like points are immutable --> replace by new point
        new_pos = Point(self.pos.x + self.move_x, self.pos.y + self.move_y)
        # reverse movement if otherwise moving out of map
        if not new_pos.within(self.map.shape):
            self.move_x = -self.move_x
            self.move_y = -self.move_y
            new_pos = Point(self.pos.x + self.move_x, self.pos.y + self.move_y)
        self.pos = new_pos

        num_lost_connections = self.check_bs_connection()
        self.log = self.log.bind(pos=str(new_pos), move=(self.move_x, self.move_y))
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
