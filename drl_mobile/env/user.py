import structlog
from shapely.geometry import Point


class User:
    """A user/UE moving around in the world and requesting mobile services"""
    def __init__(self, id, start_pos, move_x=0, move_y=0):
        self.id = id
        self.pos = start_pos
        self.move_x = move_x
        self.move_y = move_y
        self.map = None
        self.assigned_bs = []
        self.log = structlog.get_logger(id=self.id, pos=str(self.pos), move=(self.move_x, self.move_y),
                                        assigned_bs=self.assigned_bs)

    def __repr__(self):
        return self.id

    def move(self):
        """
        Do one step in movement direction and update position
        Reverse movement direction to avoid moving out of the map
        """
        # seems like points are immutable --> replace by new point
        new_pos = Point(self.pos.x + self.move_x, self.pos.y + self.move_y)
        # reverse movement if otherwise moving out of map
        if not new_pos.within(self.map):
            self.move_x = -self.move_x
            self.move_y = -self.move_y
            new_pos = Point(self.pos.x + self.move_x, self.pos.y + self.move_y)
        self.pos = new_pos

        self.log = self.log.bind(pos=str(new_pos), move=(self.move_x, self.move_y))
        self.log.debug("User move")
        self.check_bs_connection()

    def check_bs_connection(self):
        """Check if assigned BS connections are still stable (after move), else remove."""
        remove_bs_idx = []
        for i, bs in enumerate(self.assigned_bs):
            if not self.pos.within(bs.coverage):
                self.log.info("Losing connection to BS", bs=bs)
                remove_bs_idx.append(i)
        # remove bs
        for i in remove_bs_idx:
            del self.assigned_bs[i]

    def connect_to_bs(self, bs):
        """
        Try to connect to specified basestation. Return if successful.
        :param bs: Basestation to connect to
        :return: True if connected successfully (even if was connected before). False if out of range.
        """
        if bs in self.assigned_bs:
            log = self.log.bind(bs=bs)
            log.info("Already connected to BS", bs=bs)
            return True
        if self.pos.within(bs.coverage):
            self.assigned_bs.append(bs)
            self.log.info("Connected to BS", bs=bs)
            return True
        else:
            self.log.info("Cannot connect to BS", bs=bs)
            return False
