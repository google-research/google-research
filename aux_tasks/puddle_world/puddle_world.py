# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A PuddleWorld that allows arbitrary geometry.

This has a slightly different take on PuddleWorld, since we do not use
puddles that impose a negative reward for stepping through them. Instead
we use puddles that act as walls (WallPuddle), or are harder to move
through (SlowPuddle).
"""
import abc
import dataclasses
import enum
from typing import Generator, Sequence, Union

import numpy as np
from shapely import geometry



def _clip(x, min_val, max_val):
  return min(max(x, min_val), max_val)


def _iterate_segments(
    points
    ):
  for p1, p2 in zip(points, points[1:]):
    yield geometry.LineString((p1, p2))


def circle(center,
           radius,
           num_points = 100):
  theta = np.linspace(0.0, 2 * np.pi, num=num_points, endpoint=False)
  xs = center.x + radius * np.cos(theta)
  ys = center.y + radius * np.sin(theta)
  return geometry.Polygon(list(zip(xs, ys)))


class Puddle(abc.ABC):
  """Abstract base class for puddles."""

  def __init__(self, shape):
    self.shape = shape

  @abc.abstractmethod
  def update_transition_end_point(
      self,
      start_point,
      end_point):
    """Calculates the new end point of a transition given this puddle.

    Args:
      start_point: The start point of the transition.
      end_point: The end point of the transition if no puddle were found.

    Returns:
      A newly calculated end point that takes into account whether the
        transition passes through the puddle.
    """

  @abc.abstractmethod
  def is_valid_position(self, position):
    """Returns whether the specified position is valid.

    Some puddles may not allow the agent to enter certain parts of it.
    If the supplied position is one of these points, this function should
    return False.

    Args:
      position: The position to test.

    Returns:
      True if the position is valid, False if it is invalid.
    """


class SlowPuddle(Puddle):
  """A puddle that slows movement through it."""

  def __init__(self,
               shape,
               multiplier = 0.5):
    """SlowPuddle constructor.

    Args:
      shape: The shape of the puddle.
      multiplier: The speed multiplier for the puddle. A value of 0.0 makes
        the puddle act as a wall, while 1.0 means the puddle acts as if
        there were no puddle at all. Must be in [0, 1].
    """
    super().__init__(shape)

    # TODO: Better name than multiplier.
    if not 0 <= multiplier <= 1:
      raise ValueError(f'multiplier must be in [0, 1]. Got {multiplier}.')
    self.multiplier = multiplier

  def update_transition_end_point(
      self,
      start_point,
      end_point):
    """Calculates the new end point of a transition given this puddle."""
    full_segment = geometry.LineString((start_point, end_point))
    full_length = full_segment.length

    intersections = self.shape.intersection(full_segment)
    if not intersections:
      return end_point  # We aren't touching the puddle! 🎉

    # Turn the intersections into a series of points.
    intersection_points = list()
    if isinstance(intersections, geometry.MultiLineString):
      for segment in intersections:
        intersection_points.extend(
            [geometry.Point(c) for c in segment.coords])
    elif isinstance(intersections, geometry.MultiPoint):
      for point in intersections:
        intersection_points.append(point)
    else:
      # It's just one segment.
      intersection_points.extend(
          [geometry.Point(c) for c in intersections.coords])

    if start_point not in intersection_points:
      intersection_points.append(start_point)
    if end_point not in intersection_points:
      intersection_points.append(end_point)

    intersection_points = sorted(intersection_points,
                                 key=start_point.distance)

    expended_distance = 0.0
    for segment in _iterate_segments(intersection_points):
      moving_through_puddle = self.shape.contains(segment.centroid)
      multiplier = self.multiplier if  moving_through_puddle else 1.0

      if multiplier == 0.0:
        cost = float('inf')
      else:
        cost = segment.length / multiplier

      if expended_distance + cost >= full_length:
        remaining_distance = full_length - expended_distance

        if cost == float('inf'):
          fraction_moved = 0.0
        else:
          fraction_moved = remaining_distance / cost

        return segment.interpolate(fraction_moved, normalized=True)

      expended_distance += cost

    # If we get to the end and haven't run out of movement, return the
    # original end point.
    return end_point

  def is_valid_position(self, position):
    """Returns whether the specified position is valid."""
    if self.multiplier == 0.0:
      return self.shape.contains(position)
    return True


class WallPuddle(SlowPuddle):
  """A puddle that acts as a wall, blocking any movement."""

  def __init__(self, shape):
    super().__init__(shape, multiplier=0.0)


class Action(enum.IntEnum):
  UP: int = 0
  RIGHT: int = 1
  DOWN: int = 2
  LEFT: int = 3


@dataclasses.dataclass
class Transition:
  state: geometry.Point
  action: Action
  reward: float
  next_state: geometry.Point
  is_terminal: bool


class PuddleWorld:
  """Class defining a Puddle World."""

  def __init__(self,
               puddles,
               goal_position,
               noise = 0.01,
               thrust = 0.05,
               goal_radius = 0.05,
               step_cost = 0.0):
    """PuddleWorld constructor 💦.

    A PuddleWorld is a square arena, with x, y both in [0, 1].
    It has a continous state space and discrete action space.
    Within the arena are puddles, which in this implementation are arbitrary
    shapes that impose different movement rules on the agent.

    Note that puddles in PuddleWorld generally impose a negative reward for
    traversing through them. However, this implementation allows a broader
    definition of a puddle, including puddles that act as walls, or
    puddles that slow the movement of the agent.

    Args:
      puddles: A sequence of puddles. PuddleWorld accepts a mixture of
        puddle types. For example, this may contain a mixture of WallPuddles
        and SlowPuddles.
      goal_position: The position of the goal in the arena.
      noise: The standard deviation of the gaussian noise added to
        each transition.
      thrust: The size of each step.
      goal_radius: If within goal_radius of goal_position, it is a goal state.
      step_cost: The cost for each step in the environment.
    """
    self.puddles = puddles
    self.goal = goal_position
    self.noise = noise
    self.thrust = thrust
    self.goal_radius = goal_radius
    self.step_cost = step_cost

  def transition(self,
                 state,
                 action):
    """Samples a transition from the PuddleWorld.

    Args:
      state: The starting state.
      action: The action to take.

    Returns:
      A transition which includes the next observed state, the reward,
        and whether we have transitioned to a terminal state.
    """
    action = Action(action)

    noise = np.random.normal(scale=self.noise, size=(2,))
    delta_x = noise[0].item()
    delta_y = noise[1].item()

    if action == Action.UP:
      delta_y += self.thrust
    elif action == Action.RIGHT:
      delta_x += self.thrust
    elif action == Action.DOWN:
      delta_y -= self.thrust
    elif action == Action.LEFT:
      delta_x -= self.thrust
    else:
      raise ValueError(
          f'Unknown action {action}. Actions should be in {0, 1, 2, 3}.')

    new_x = _clip(state.x + delta_x, 0.0, 1.0)
    new_y = _clip(state.y + delta_y, 0.0, 1.0)
    next_state = geometry.Point((new_x, new_y))

    # Let the puddles do their work on the new state.
    for puddle in self.puddles:
      next_state = puddle.update_transition_end_point(state, next_state)

    # TODO: is_near_goal is slow, so have a separate function.
    # is_terminal = self._is_near_goal(next_state)
    is_terminal = False

    # TODO: Update the reward function, if we need it.
    reward = -self.step_cost
    if is_terminal:
      reward = 1.0

    return Transition(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        is_terminal=is_terminal)

  def is_valid_position(self, position):
    return all(p.is_valid_position(position) for p in self.puddles)

  def _is_near_goal(self,
                    state):
    return self.goal.distance(state) <= self.goal_radius
