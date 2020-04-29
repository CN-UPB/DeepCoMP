from shapely.geometry import Point


class User:
    """A user/UE moving around in the world and requesting mobile services"""
    def __init__(self, start_pos, move_x=0, move_y=0):
        self.pos = start_pos
        self.move_x = move_x
        self.move_y = move_y
        self.map = None

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
