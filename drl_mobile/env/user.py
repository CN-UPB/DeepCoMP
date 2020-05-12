import structlog
from shapely.geometry import Point


class User:
    """
    A user/UE moving around in the world and requesting mobile services
    Connection to BS are checked before connecting and after every move to check if connection is lost or still stable
    """
    def __init__(self, id, start_pos, move_x=0, move_y=0, dr_req=1):
        """
        Create new UE object
        :param id: Unique ID of UE (string)
        :param start_pos: Starting position of the UE (Point)
        :param move_x: Movement per step along x-axis
        :param move_y: Movement per step along y-axis
        :param dr_req: Data rate requirement by UE for successful service
        """
        self.id = id
        self.pos = start_pos
        self.move_x = move_x
        self.move_y = move_y
        self.dr_req = dr_req
        self.env = None
        self.conn_bs = []
        self.log = structlog.get_logger(id=self.id, pos=str(self.pos), move=(self.move_x, self.move_y),
                                        conn_bs=self.conn_bs, dr_req=self.dr_req)
        # keep initial pos and movement for resetting
        self._init_pos = start_pos
        self._init_move_x = move_x
        self._init_move_y = move_y

    def __repr__(self):
        return self.id

    def reset(self):
        """Reset UE to initial position and movement. Disconnect from all BS."""
        self.pos = self._init_pos
        self.move_x = self._init_move_x
        self.move_y = self._init_move_y
        self.conn_bs = []

    def move(self):
        """
        Do one step in movement direction and update position
        Reverse movement direction to avoid moving out of the map
        """
        # seems like points are immutable --> replace by new point
        new_pos = Point(self.pos.x + self.move_x, self.pos.y + self.move_y)
        # reverse movement if otherwise moving out of map
        if not new_pos.within(self.env.map):
            self.move_x = -self.move_x
            self.move_y = -self.move_y
            new_pos = Point(self.pos.x + self.move_x, self.pos.y + self.move_y)
        self.pos = new_pos

        self.log = self.log.bind(pos=str(new_pos), move=(self.move_x, self.move_y))
        self.log.debug("User move")
        self.check_bs_connection()

    def check_bs_connection(self):
        """Check if assigned BS connections are still stable (after move), else remove."""
        remove_bs = []
        for bs in self.conn_bs:
            if not self.can_connect(bs):
                self.log.info("Losing connection to BS", bs=bs)
                remove_bs.append(bs)
        # remove/disconnect bs
        for bs in remove_bs:
            self.disconnect_from_bs(bs)

    def can_connect(self, bs):
        """Return whether or not the UE can connect to the BS (based achievable data rate at current pos)"""
        dr = bs.data_rate(self.pos, self.env.active_bs)
        return dr >= self.dr_req

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
        if self.can_connect(bs):
            self.conn_bs.append(bs)
            bs.conn_ue.append(self)
            log.info("Connected", conn_bs=self.conn_bs)
            self.log = self.log.bind(conn_bs=self.conn_bs)
            return True
        else:
            log.info("Cannot connect")
            return False

    def disconnect_from_bs(self, bs):
        """Disconnect from given BS. Assume BS is currently connected."""
        assert bs in self.conn_bs, "Not connected to BS --> Cannot disconnect"
        self.conn_bs.remove(bs)
        bs.conn_ue.remove(self)
        self.log = self.log.bind(conn_bs=self.conn_bs)
