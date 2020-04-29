from shapely.geometry import Polygon


class World:
    """The playground/map/world of the simulation"""
    def __init__(self, width, height, bs_list, ue_list):
        # construct the rectangular world map
        self.width = width
        self.height = height
        self.map = Polygon([(0,0), (0, height), (width, height), (width, 0)])
        # save other attributes
        self.bs_list = bs_list
        self.ue_list = ue_list
