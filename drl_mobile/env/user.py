import structlog
from shapely.geometry import Point


log = structlog.get_logger()


class User:
    """A user/UE moving around in the world and requesting mobile services"""
    def __init__(self, id, start_pos, move_x=0, move_y=0):
        self.id = id
        self.pos = start_pos
        self.move_x = move_x
        self.move_y = move_y
        self.map = None
        self.assigned_bs = []

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

    def connect_to_bs(self, bs):
        """
        Try to connect to specified basestation. Return if successful.
        :param bs: Basestation to connect to
        :return: True if connected successfully (even if was connected before). False if out of range.
        """
        if bs in self.assigned_bs:
            # TODO: properly configure/use structlog to bind all ue info once
            log.msg("Already connected to BS")
            return True
        # TODO: check if BS is in range; if yes --> connect & true, else false
