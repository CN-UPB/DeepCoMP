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
        self.shape = Polygon([(0,0), (0, height), (width, height), (width, 0)])

    def __repr__(self):
        return f'{self.width}x{self.height}map'
