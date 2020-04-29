from shapely.geometry import Polygon
import matplotlib.pyplot as plt


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

    def plot(self, title=None):
        """Plot and visualize the current status of the world"""
        # square figure and equal aspect ratio to avoid distortions
        plt.figure(figsize=(5, 5))
        plt.gca().set_aspect('equal')

        # map borders
        plt.plot(*self.map.exterior.xy)
        # users
        for ue in self.ue_list:
            plt.scatter(*ue.pos.xy)
        # base stations
        for bs in self.bs_list:
            plt.scatter(*bs.pos.xy, marker='^', c='black')
            plt.plot(*bs.coverage.exterior.xy, color='black')

        plt.title(title)
        plt.show()

    def step(self):
        """Do 1 time step and update UE position"""
        for ue in self.ue_list:
            ue.move()
            # TODO: avoid moving out of map
