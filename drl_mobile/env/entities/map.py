import numpy as np
from shapely.geometry import Polygon


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
        self.width = width
        self.height = height
        self.diagonal = np.sqrt(self.width**2 + self.height**2)
        self.shape = Polygon([(0,0), (0, height), (width, height), (width, 0)])

    def __repr__(self):
        return f'{self.width}x{self.height}map'

    @property
    def figsize(self, target_height=7):
        """Scale figsize to target height while keeping the aspect ratio"""
        scaling_factor = self.height / target_height
        width = self.width / scaling_factor
        return int(width), int(target_height)
