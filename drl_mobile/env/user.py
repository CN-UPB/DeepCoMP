class User:
    """A user/UE moving around in the world and requesting mobile services"""
    def __init__(self, start_pos, move_x=0, move_y=0):
        self.pos = start_pos
        self.move_x = move_x
        self.move_y = move_y
