import random

import numpy as np
from shapely.geometry import Point, Polygon


class Map:
    """
    Map/Playground of the environment: Rectangular space with given width and height.
    Separate class rather than just using Polygon to keep width and height accessible.
    """
    def __init__(self, width, height, min_x=0, min_y=0):
        """
        Create new rectangular world map/playground with the given width and height
        :param width: Width of the map
        :param height: Height of the map
        :param min_x: Origin x-coord
        :param min_y: Origin y-coord
        """
        self.width = int(width)
        self.height = int(height)
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = self.min_x + self.width
        self.max_y = self.min_y + self.height
        self.diagonal = np.sqrt(self.width**2 + self.height**2)
        self.shape = Polygon([(self.min_x, self.min_y), (self.min_x, self.max_y),
                              (self.max_x, self.max_y), (self.max_x, self.min_y)])
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
        x = self.rng.randint(self.min_x, self.max_x)
        y = self.rng.randint(self.min_y, self.max_y)
        # pin to one of the four borders randomly uniformly
        border = self.rng.choice(['left', 'right', 'top', 'bottom'])
        if border == 'left':
            return Point(self.min_x, y)
        if border == 'right':
            return Point(self.max_x - 1, y)
        if border == 'top':
            return Point(x, self.max_y - 1)
        if border == 'bottom':
            return Point(x, self.min_y)
