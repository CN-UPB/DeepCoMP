

class Basestation:
    """A basestation: Currently simply with a capacity and coverage radius"""
    # TODO: Next, add a real wireless model (see SVN)
    def __init__(self, pos, cap, radius):
        self.pos = pos
        self.cap = cap
        self.radius = radius
        self.coverage = pos.buffer(radius)
