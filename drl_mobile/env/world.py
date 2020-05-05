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
        # pass the map to all users (needed for movement)
        self.ue_list = ue_list
        for ue in self.ue_list:
            ue.map = self.map

    def plot(self, title=None):
        """Plot and visualize the current status of the world"""
        # square figure and equal aspect ratio to avoid distortions
        plt.figure(figsize=(5, 5))
        plt.gca().set_aspect('equal')

        # map borders
        plt.plot(*self.map.exterior.xy)
        # users & connections
        for ue in self.ue_list:
            plt.scatter(*ue.pos.xy)
            for bs in ue.assigned_bs:
                plt.plot([ue.pos.x, bs.pos.x], [ue.pos.y, bs.pos.y], color='orange')
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
            # test: always try to connect to same BS
            ue.connect_to_bs(self.bs_list[1])
