"""Utility functions for UE movement"""
import random

import structlog
import numpy as np
from shapely.geometry import Point


class RandomWaypoint:
    def __init__(self, map, pause_duration=1):
        """
        Create random waypoint movement utility object.

        :param map: Map representing the area of movement that must not be left
        :param pause_duration: Duration [in env steps] to pause after reaching each waypoint
        """
        self.map = map
        self.pause_duration = pause_duration
        self.waypoint = None
        self.log = structlog.get_logger()

    def set_random_waypoint(self):
        """Set a new random waypoint inside the map"""
        x = random.randint(0, self.map.width)
        y = random.randint(0, self.map.height)
        self.waypoint = Point(x, y)
        assert self.waypoint.within(self.map), f"Waypoint {str(self.waypoint)} is outside the map!"

    def step_towards_waypoint(self, curr_pos, max_dist):
        """
        Take one step (of given max_dist) from the current position towards the current waypoint.

        :param Point curr_pos: Current position
        :param max_dist: (Max) distance to move within one step. Less if the waypoint is reached earlier.
        :return: New position after moving
        """
        self.log = self.log.bind(prev_pos=str(curr_pos), waypoint=str(self.waypoint))

        # if already close enough to waypoint, move directly onto waypoint (not passed it)
        if curr_pos.distance(self.waypoint) <= max_dist:
            self.log.info('Waypoint reached')
            return self.waypoint

        # else move by max_dist towards waypoint: https://math.stackexchange.com/a/175906/234077
        # convert points to np arrays/vectors for calculation
        np_curr = np.array([curr_pos.x, curr_pos.y])
        np_waypoint = np.array([self.waypoint.x, self.waypoint.y])
        v = np_waypoint - np_curr
        norm_v = v / np.linalg.norm(v)
        np_new_pos = np_curr + max_dist * norm_v
        new_pos = Point(np_new_pos)

        self.log.debug('Step to waypoint', new_pos=str(new_pos))
        return new_pos
