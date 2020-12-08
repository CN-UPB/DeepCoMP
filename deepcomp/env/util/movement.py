"""Utility functions for UE movement. Must inherit from abstract Movement class."""
import random

import structlog
import numpy as np
from shapely.geometry import Point


class Movement:
    """Abstract movement class that all subclasses must inherit from"""
    def __init__(self, map):
        self.map = map
        # own RNG for reproducibility; global random shares state that's manipulated by RL during training
        self.rng = random.Random()

    def seed(self, seed=None):
        self.rng.seed(seed)

    def reset(self):
        raise NotImplementedError("This function must be implemented in the subclass")

    def step(self, curr_pos):
        raise NotImplementedError("This function must be implemented in the subclass")


class UniformMovement(Movement):
    """
    Uniformly move with same speed in one direction. When hitting the borders of the map, "bounce off" like a ball
    """
    def __init__(self, map, move_x=0, move_y=0):
        """
        Create object for uniform movement into given direction. Instantiate new movement object for each UE!

        :param map: Map representing the area of movement
        :param move_x: How far to move in x direction per step. Number or 'slow'/'fast'.
        :param move_y: How far to move in y direction per step. Number or 'slow'/'fast'.
        """
        super().__init__(map)
        self.init_move_x = move_x
        self.init_move_y = move_y
        self.move_x = None
        self.move_y = None

    def __str__(self):
        return f"UniformMovement({self.move_x}, {self.move_y})"

    def reset(self):
        """Reset to original movement direction (may change when hitting a map border)"""
        if self.init_move_x == 'slow':
            self.move_x = self.rng.randint(1, 5)
        elif self.init_move_x == 'fast':
            self.move_x = self.rng.randint(10, 20)
        else:
            # assume init_move_x was a specific number for how to move
            self.move_x = self.init_move_x
        # same for move_y
        if self.init_move_y == 'slow':
            self.move_y = self.rng.randint(1, 5)
        elif self.init_move_y == 'fast':
            self.move_y = self.rng.randint(10, 20)
        else:
            # assume init_move_y was a specific number for how to move
            self.move_y = self.init_move_y

    def step(self, curr_pos):
        """
        Make a step from the current position into the given direction. Bounce off at map borders.

        :param Point curr_pos: Current position
        :returns Point: New position after taking one step
        """
        # seems like points are immutable --> replace by new point
        new_pos = Point(curr_pos.x + self.move_x, curr_pos.y + self.move_y)
        # reverse movement if otherwise moving out of map
        if not new_pos.within(self.map.shape):
            self.move_x = -self.move_x
            self.move_y = -self.move_y
            new_pos = Point(curr_pos.x + self.move_x, curr_pos.y + self.move_y)
        return new_pos


class RandomWaypoint(Movement):
    """
    Create random waypoints, move towards them, pause, and move towards new random waypoint
    with new random velocity (within given range)
    """
    def __init__(self, map, velocity, pause_duration=2, border_buffer=10):
        """
        Create random waypoint movement utility object. Instantiate new movement object for each UE!

        :param map: Map representing the area of movement that must not be left
        :param velocity: Distance to move within one step. Number or 'slow'/'fast'.
        :param pause_duration: Duration [in env steps] to pause after reaching each waypoint
        :param border_buffer: Buffer to the map border in which no waypoints are placed
        """
        super().__init__(map)
        self.init_velocity = velocity
        self.velocity = None
        self.waypoint = None
        self.pause_duration = pause_duration
        self.pausing = False
        self.curr_pause = 0
        assert border_buffer > 0, "Border Buffer must be >0 to avoid placing waypoints on or outside map borders."
        self.border_buffer = border_buffer
        self.log = structlog.get_logger()

    def __str__(self):
        return f"RandomWaypoint({self.init_velocity})"

    def reset(self):
        """Reset velocity and waypoint to new random values. Reset current pause."""
        if self.init_velocity == 'slow':
            self.velocity = self.rng.randint(1, 3)
        elif self.init_velocity == 'fast':
            self.velocity = self.rng.randint(5, 10)
        else:
            self.velocity = self.init_velocity

        self.waypoint = self.random_waypoint()
        self.pausing = False
        self.curr_pause = 0
        self.log.debug('Reset movement', new_velocity=self.velocity, new_waypoint=str(self.waypoint))

    def random_waypoint(self):
        """Return a new random waypoint inside the map"""
        x = self.rng.randint(self.border_buffer, int(self.map.width - self.border_buffer))
        y = self.rng.randint(self.border_buffer, int(self.map.height - self.border_buffer))
        new_waypoint = Point(x, y)
        assert new_waypoint.within(self.map.shape), f"Waypoint {str(new_waypoint)} is outside the map!"
        return new_waypoint

    def step_towards_waypoint(self, curr_pos):
        """
        Take one step from the current position towards the current waypoint.

        :param Point curr_pos: Current position
        :return: New position after moving
        """
        self.log = self.log.bind(prev_pos=str(curr_pos), waypoint=str(self.waypoint))

        # if already close enough to waypoint, move directly onto waypoint (not past it)
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
        assert curr_pos.within(self.map.shape) or curr_pos.touches(self.map.shape), \
            f"Current position {str(curr_pos)} is outside the map!"

        # start pausing when reaching the waypoint
        if curr_pos == self.waypoint:
            self.pausing = True

        if self.pausing:
            # continue pausing if pause duration is not yet reached
            if self.curr_pause < self.pause_duration:
                self.curr_pause += 1
                return curr_pos
            # else stop pausing and choose a new waypoint --> reset()
            else:
                self.reset()

        # move towards (new) waypoint
        new_pos = self.step_towards_waypoint(curr_pos)
        return new_pos
