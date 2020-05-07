

class Basestation:
    """A basestation: Currently simply with a capacity and coverage radius"""
    # TODO: Next, add a real wireless model (see SVN)
    def __init__(self, id, pos, cap, radius):
        self.id = id
        self.pos = pos
        self.cap = cap
        self.radius = radius
        self.coverage = pos.buffer(radius)

    def __repr__(self):
        return self.id
