import random

import numpy as np
from shapely.geometry import Point, Polygon


class Map:
    """
    Map/Playground of the environment: Rectangular space with given width and height.
    Separate class rather than just using Polygon to keep width and height accessible.
    """
    def __init__(self, width, height):
        """
        Create new rectangular world map/playground with the given width and height
        :param width: Width of the map
        :param height: Height of the map
        """
        self.width = int(width)
        self.height = int(height)
        self.diagonal = np.sqrt(self.width**2 + self.height**2)
        self.shape = Polygon([(0,0), (0, self.height), (self.width, self.height), (self.width, 0)])
        # own RNG for reproducibility; global random shares state that's manipulated by RL during training
        self.rng = random.Random()

    def __repr__(self):
        return f'{self.width}x{self.height}map'

    @property
    def figsize(self, target_height=7):
        """Scale figsize to target height while keeping the aspect ratio"""
        scaling_factor = self.height / target_height
        width = self.width / scaling_factor
        return int(width), int(target_height)

    def seed(self, seed=None):
        self.rng.seed(seed)

    def rand_border_point(self):
        """Return a random point on the border of the map"""
        x = self.rng.randint(0, self.width)
        y = self.rng.randint(0, self.height)
        # pin to one of the four borders randomly uniformly
        border = self.rng.choice(['left', 'right', 'top', 'bottom'])
        if border == 'left':
            return Point(0, y)
        if border == 'right':
            return Point(self.width - 1, y)
        if border == 'top':
            return Point(x, self.height - 1)
        if border == 'bottom':
            return Point(x, 0)
