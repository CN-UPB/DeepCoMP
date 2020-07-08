"""Utility functions for UE movement. Must """
import random

import structlog
import numpy as np
from shapely.geometry import Point


class Movement:
    """Abstract movement class that all subclasses must inherit from"""
    def __init__(self, map):
        self.map = map

    def reset(self):
        raise NotImplementedError("This function must be implemented in the subclass")

    def step(self, curr_pos):
        raise NotImplementedError("This function must be implemented in the subclass")


class UniformMovement(Movement):
    pass
    # TODO: move existing, old functionality here


class RandomWaypoint(Movement):
    """
    Create random waypoints, move towards them, pause, and move towards new random waypoint
    with new random velocity (within given range)
    """
    # TODO: test
    def __init__(self, map, min_velocity, max_velocity, pause_duration=1):
        """
        Create random waypoint movement utility object.

        :param map: Map representing the area of movement that must not be left
        :param min_velocity: Lower bound on distance to move within one step. Less if the waypoint is reached earlier.
        :param max_velocity: Upper bound on distance to move within one step. Less if the waypoint is reached earlier.
        :param pause_duration: Duration [in env steps] to pause after reaching each waypoint
        """
        super().__init__(map)
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.velocity = None
        self.pause_duration = pause_duration
        self.waypoint = None
        self.log = structlog.get_logger()

        # reset to set velocity and waypoint
        self.reset()

    def reset(self):
        """Reset velocity and waypoint to new random values"""
        self.velocity = random.randint(self.min_velocity, self.max_velocity)
        self.waypoint = self.random_waypoint()

    def random_waypoint(self):
        """Return a new random waypoint inside the map"""
        x = random.randint(0, self.map.width)
        y = random.randint(0, self.map.height)
        new_waypoint = Point(x, y)
        assert new_waypoint.within(self.map), f"Waypoint {str(new_waypoint)} is outside the map!"
        return new_waypoint

    def step_towards_waypoint(self, curr_pos):
        """
        Take one step from the current position towards the current waypoint.

        :param Point curr_pos: Current position
        :return: New position after moving
        """
        self.log = self.log.bind(prev_pos=str(curr_pos), waypoint=str(self.waypoint))

        # if already close enough to waypoint, move directly onto waypoint (not passed it)
        if curr_pos.distance(self.waypoint) <= self.velocity:
            self.log.debug('Waypoint reached')
            return self.waypoint

        # else move by self.velocity towards waypoint: https://math.stackexchange.com/a/175906/234077
        # convert points to np arrays/vectors for calculation
        np_curr = np.array([curr_pos.x, curr_pos.y])
        np_waypoint = np.array([self.waypoint.x, self.waypoint.y])
        v = np_waypoint - np_curr
        norm_v = v / np.linalg.norm(v)
        np_new_pos = np_curr + self.velocity * norm_v
        new_pos = Point(np_new_pos)

        self.log.debug('Step to waypoint', new_pos=str(new_pos))
        return new_pos

    def step(self, curr_pos):
        """
        Perform one movement step: Move towards waypoint, pause, select new waypoint & repeat.
        :param Point curr_pos: Current position of the moving entity (eg, UE)
        :return: New position after one step
        """
        assert curr_pos.within(self.map), f"Current position {str(curr_pos)} is outside the map!"

        # pick new waypoint and velocity when a waypoint is reached --> reset()
        if curr_pos == self.waypoint:
            self.reset()

        # TODO: keep track of pausing: am I currently pausing? how long? --> pause further or stop
        new_pos = self.step_towards_waypoint(curr_pos)
        return new_pos
