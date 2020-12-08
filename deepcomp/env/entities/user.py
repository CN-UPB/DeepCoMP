import random

import structlog
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib import cm

from deepcomp.env.util.utility import step_utility, log_utility
from deepcomp.util.constants import FAIR_WEIGHT_ALPHA, FAIR_WEIGHT_BETA, EPSILON


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
        # dict of connected BS: BS (only connected BS are keys!) --> data rate of connection
        self.bs_dr = dict()

        # own RNG for reproducibility; global random shares state that's manipulated by RL during training
        self.rng = random.Random()
        self.init_pos_x = pos_x
        self.init_pos_y = pos_y
        self.pos = None
        self.reset_pos()
        self.movement.reset()

        # exponentially weighted moving average data rate
        self.ewma_dr = 0

        self.log = structlog.get_logger(id=self.id, pos=str(self.pos), ewma_dr=self.ewma_dr,
                                        conn_bs=list(self.bs_dr.keys()), dr_req=self.dr_req)
        self.log.info('UE init')

    def __repr__(self):
        return str(self.id)

    # compare and hash UEs based on their ID only
    def __eq__(self, other):
        if type(other) is type(self):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    @property
    def curr_dr(self):
        """Current data rate the UE gets through all its BS connections"""
        dr = sum([dr for dr in self.bs_dr.values()])
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

    def seed(self, seed=None):
        self.rng.seed(seed)
        self.movement.seed(seed)

    def reset_pos(self):
        """(Re)set position based on initial position x and y as Point. Resolve 'random'."""
        # set pos_x
        pos_x = self.init_pos_x
        if pos_x == 'random':
            pos_x = self.rng.randint(0, int(self.map.width))
        # set pos_y
        pos_y = self.init_pos_y
        if pos_y == 'random':
            pos_y = self.rng.randint(0, int(self.map.height))
        # set pos as Point
        self.pos = Point(pos_x, pos_y)

    def reset(self):
        """Reset UE position, movement, and connections."""
        self.reset_pos()
        self.movement.reset()
        self.bs_dr = dict()
        self.ewma_dr = 0

    def plot(self, radius=2):
        """
        Plot the UE as filled circle with a given radius and the ID. Color from red to green indicating the utility.
        :param radius: Radius of the circle
        :return: A list of created matplotlib artists
        """
        # show utility as red to yellow to green. use color map for [0,1) --> normalize utility first
        colormap = cm.get_cmap('RdYlGn')
        norm = plt.Normalize(-20, 20)
        color = colormap(norm(self.utility))

        artists = plt.plot(*self.pos.buffer(radius).exterior.xy, color=color)
        artists.extend(plt.fill(*self.pos.buffer(radius).exterior.xy, color=color))
        artists.append(plt.annotate(self.id, xy=(self.pos.x, self.pos.y), ha='center', va='center'))

        # show curr data rate and utility below the UE
        artists.append(plt.annotate(f'dr: {self.curr_dr:.2f}', xy=(self.pos.x, self.pos.y -radius -2),
                                    ha='center', va='center'))
        artists.append(plt.annotate(f'util: {self.utility:.2f}', xy=(self.pos.x, self.pos.y -radius -6),
                                    ha='center', va='center'))
        return artists

    def update_curr_dr(self):
        """Update the current data rate of all BS connections according to the current situation (pos & assignment)"""
        for bs in self.bs_dr.keys():
            self.bs_dr[bs] = bs.data_rate(self)

    def update_ewma_dr(self, weight=0.9):
        """
        Update the exp. weighted moving avg. of this UE's current data rate:
        `EWMA(t) = weight * dr + (1-weight) * EWMA(t-1)`
        Used as historic avg. rate for proportional-fair sharing. Called after movement.

        :param weight: Weight for EWMA in [0, 1]. The higher, the more focus on new/current dr and less on previous.
        """
        self.ewma_dr = weight * self.curr_dr + (1 - weight) * self.ewma_dr
        self.log = self.log.bind(ewma_dr=self.ewma_dr)

    def move(self):
        """
        Do one step: Move according to own movement pattern. Check for lost connections. Update EWMA data rate.

        :return: Number of connections lost through movement
        """
        self.pos = self.movement.step(self.pos)

        num_lost_connections = self.check_bs_connection()
        self.log = self.log.bind(pos=str(self.pos))

        self.update_ewma_dr()

        self.log.debug("User move", lost_connections=num_lost_connections)
        return num_lost_connections

    def check_bs_connection(self):
        """
        Check if assigned BS connections are still stable (after move), else remove.
        :return: Number of removed/lost connections
        """
        remove_bs = []
        for bs in self.bs_dr.keys():
            if not bs.can_connect(self.pos):
                self.log.info("Losing connection to BS", bs=bs)
                remove_bs.append(bs)
        # remove/disconnect bs
        for bs in remove_bs:
            self.disconnect_from_bs(bs)
        return len(remove_bs)

    def connect_to_bs(self, bs, disconnect=False, return_connected=False):
        """
        Try to connect to specified basestation. Return if successful.

        :param bs: Basestation to connect to
        :param disconnect: If True, disconnect from BS if it was previously connected.
        :param return_connected: If True, return whether the UE is now connected to the BS or not.
        Else, return if the (dis-)connect was successful.
        :return: True if (dis-)connected successfully. False if out of range. If return_connected, return if connected.

        """
        log = self.log.bind(bs=bs, disconnect=disconnect, conn_bs=list(self.bs_dr.keys()))
        # already connected
        if bs in self.bs_dr.keys():
            if disconnect:
                self.disconnect_from_bs(bs)
                log.info("Disconnected")
                if return_connected:
                    return False
            else:
                log.info("Staying connected")
            return True
        # not yet connected
        if bs.can_connect(self.pos):
            # add BS to connections; important: initialize with data rate
            # also important: initialize before adding connection to bs.conn_ues; affects how data rate is calc
            self.bs_dr[bs] = bs.data_rate(self)
            bs.conn_ues.append(self)
            self.log = self.log.bind(conn_bs=list(self.bs_dr.keys()))
            log.info("Connected")
            return True
        else:
            # log.info("Cannot connect")
            return False

    def disconnect_from_bs(self, bs):
        """Disconnect from given BS. Assume BS is currently connected."""
        assert bs in self.bs_dr.keys(), "Not connected to BS --> Cannot disconnect"
        del self.bs_dr[bs]
        bs.conn_ues.remove(self)
        self.log = self.log.bind(conn_bs=list(self.bs_dr.keys()))

    def ues_at_same_bs(self):
        """Return set of UEs that are currently connected to any of the BS that this UE is connected to"""
        ue_set = set()
        for bs in self.bs_dr.keys():
            ue_set.update(set(bs.conn_ues))
        self.log.debug('UEs at same BS', ue_set=ue_set)
        return ue_set
