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

"""Tests for puddle_world."""

from absl.testing import absltest
from absl.testing import parameterized
from shapely import geometry

from aux_tasks.puddle_world import puddle_world



def rectangle(p1, p2):
  p1, p2, p3, p4 = (p1.x, p1.y), (p2.x, p1.y), (p2.x, p2.y), (p1.x, p2.y)
  return geometry.Polygon([p1, p2, p3, p4])


class PuddleWorldTest(parameterized.TestCase):

  def assertPointsAlmostEqual(
      self, p1, p2):
    distance = p1.distance(p2)
    self.assertAlmostEqual(distance, 0.0)

  @parameterized.named_parameters(
      dict(testcase_name='stepping_over_wall_blocks',
           start_point=geometry.Point((0.0, 0.5)),
           end_point=geometry.Point((1.0, 0.5)),
           expected_end_point=geometry.Point((0.5, 0.5))),
      dict(testcase_name='stepping_into_wall_blocks',
           start_point=geometry.Point((0.0, 0.5)),
           end_point=geometry.Point((0.6, 0.5)),
           expected_end_point=geometry.Point((0.5, 0.5))),
      dict(testcase_name='from_boundary_away_doesnt_block',
           start_point=geometry.Point((0.5, 0.5)),
           end_point=geometry.Point((0.0, 0.5)),
           expected_end_point=geometry.Point((0.0, 0.5))),
      dict(testcase_name='from_boundary_towards_blocks',
           start_point=geometry.Point((0.5, 0.5)),
           end_point=geometry.Point((0.6, 0.8)),
           expected_end_point=geometry.Point((0.5, 0.5))),
      dict(testcase_name='moving_along_wall_doesnt_block',
           start_point=geometry.Point((0.5, 0.5)),
           end_point=geometry.Point((0.5, 0.8)),
           expected_end_point=geometry.Point((0.5, 0.8))),
      dict(testcase_name='stepping_out_of_wall_blocks',
           start_point=geometry.Point((0.6, 0.5)),
           end_point=geometry.Point((0.0, 0.0)),
           expected_end_point=geometry.Point((0.6, 0.5)),))
  def test_wall_blocks_movement_correctly(
      self,
      start_point,
      end_point,
      expected_end_point):
    shape = rectangle(geometry.Point((0.5, 0.0)),
                      geometry.Point((0.75, 1.0)))
    wall = puddle_world.WallPuddle(shape)

    new_end_point = wall.update_transition_end_point(start_point, end_point)
    self.assertPointsAlmostEqual(new_end_point, expected_end_point)

  def test_wall_blocks_movement_for_non_convex_shape(self):
    # This is a special case that can't be worked into the parameterized test.
    # The following shape is a square 'U' shape.
    shape = geometry.Polygon((
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.75, 1.0),
        (0.75, 0.25),
        (0.25, 0.25),
        (0.25, 1.0),
        (0.0, 1.0)))
    wall = puddle_world.WallPuddle(shape)

    # We are stepping across empty space, from one wall boundary to another.
    start_point = geometry.Point((0.25, 0.5))
    end_point = geometry.Point((0.75, 0.5))
    new_end_point = wall.update_transition_end_point(start_point, end_point)

    self.assertEqual(new_end_point, geometry.Point((0.75, 0.5)))

  @parameterized.named_parameters(
      dict(testcase_name='inside',
           start_point=geometry.Point((0.25, 0.5)),
           end_point=geometry.Point((0.75, 0.5)),
           expected_end_point=geometry.Point((0.5, 0.5))),
      dict(testcase_name='from_outside_in',
           start_point=geometry.Point((-0.5, 0.5)),
           end_point=geometry.Point((0.5, 0.5)),
           expected_end_point=geometry.Point((0.25, 0.5))),
      dict(testcase_name='from_inside_out',
           start_point=geometry.Point((0.5, 0.5)),
           end_point=geometry.Point((0.5, 2.0)),
           expected_end_point=geometry.Point((0.5, 1.5))),
      dict(testcase_name='from_inside_to_border',
           start_point=geometry.Point((0.5, 0.5)),
           end_point=geometry.Point((0.5, 1.5)),
           expected_end_point=geometry.Point((0.5, 1.0))),
      dict(testcase_name='from_border_to_border',
           start_point=geometry.Point((0.0, 1.0)),
           end_point=geometry.Point((1.0, 0.0)),
           expected_end_point=geometry.Point((0.5, 0.5))),
      dict(testcase_name='along_edge',
           start_point=geometry.Point((0.0, 0.0)),
           end_point=geometry.Point((0.0, 1.0)),
           expected_end_point=geometry.Point((0.0, 1.0))),
      dict(testcase_name='low_multiplier',
           start_point=geometry.Point((0.25, 0.5)),
           end_point=geometry.Point((0.75, 0.5)),
           expected_end_point=geometry.Point((0.3, 0.5)),
           multiplier=0.1),
      dict(testcase_name='high_multiplier',
           start_point=geometry.Point((0.25, 0.5)),
           end_point=geometry.Point((0.75, 0.5)),
           expected_end_point=geometry.Point((0.7, 0.5)),
           multiplier=0.9),)
  def test_slow_puddle_uses_movement_as_expected(
      self,
      start_point,
      end_point,
      expected_end_point,
      multiplier = 0.5):
    shape = rectangle(geometry.Point((0.0, 0.0)), geometry.Point((1.0, 1.0)))
    wall = puddle_world.SlowPuddle(shape, multiplier)

    new_end_point = wall.update_transition_end_point(start_point, end_point)

    self.assertAlmostEqual(new_end_point, expected_end_point)

  @parameterized.named_parameters(
      dict(testcase_name='up',
           start_position=geometry.Point((0.7, 0.7)),
           action=puddle_world.Action.UP,
           expected_end_position=geometry.Point((0.7, 0.75))),
      dict(testcase_name='right',
           start_position=geometry.Point((0.15, 0.5)),
           action=puddle_world.Action.RIGHT,
           expected_end_position=geometry.Point((0.2, 0.5))),
      dict(testcase_name='down',
           start_position=geometry.Point((0.9, 0.1)),
           action=puddle_world.Action.DOWN,
           expected_end_position=geometry.Point((0.9, 0.05))),
      dict(testcase_name='left',
           start_position=geometry.Point((0.3, 0.3)),
           action=puddle_world.Action.LEFT,
           expected_end_position=geometry.Point((0.25, 0.3)),))
  def test_puddle_world_calculates_correct_transition_with_no_puddles(
      self,
      start_position,
      action,
      expected_end_position):
    pw = puddle_world.PuddleWorld(puddles=(),
                                  goal_position=geometry.Point((0.5, 0.5)),
                                  noise=0.0)

    transition = pw.transition(start_position, action)

    self.assertEqual(transition.state, start_position)
    self.assertPointsAlmostEqual(transition.next_state, expected_end_position)

  def test_puddle_world_calculates_transition_with_multiple_slow_puddles(self):
    # This world has 2 circular ⚫️ slow puddles centered in the middle
    # of the arena. One had a radius of 0.25, and the other 0.1. We expect
    # that as we move through the larger one, we will be slowed to
    # 0.5 times base speed, and as we move through both we will be slowed to
    # 0.25 times base speed. 🐌
    pw = puddle_world.PuddleWorld(
        puddles=(
            puddle_world.SlowPuddle(
                shape=puddle_world.circle(geometry.Point((0.5, 0.5)), 0.25)),
            puddle_world.SlowPuddle(
                shape=puddle_world.circle(geometry.Point((0.5, 0.5)), 0.1))),
        goal_position=geometry.Point((0.5, 0.5)),
        noise=0.0,
        thrust=0.5)  # A large thrust to step over multiple circles.

    start_position = geometry.Point((0.2, 0.5))
    transition = pw.transition(start_position, puddle_world.Action.RIGHT)

    # 0.5 total movement 🏃
    # 0.05 spent getting to first circle's edge.
    # 0.3 spent getting to second circle's edge.
    # 0.15 remaining at 25% efficiency => 0.0375 into inner circle.
    expected_end_position = geometry.Point((0.4375, 0.5))

    self.assertEqual(transition.state, start_position)
    self.assertPointsAlmostEqual(transition.next_state, expected_end_position)

  def test_puddle_world_correctly_applies_wall_puddles(self):
    pw = puddle_world.PuddleWorld(
        puddles=(
            puddle_world.SlowPuddle(
                shape=puddle_world.circle(geometry.Point((0.5, 0.5)), 0.25)),
            puddle_world.WallPuddle(
                shape=puddle_world.circle(geometry.Point((0.5, 0.5)), 0.1))),
        goal_position=geometry.Point((0.5, 0.5)),
        noise=0.0,
        thrust=0.5)  # A large thrust to step over multiple circles.

    start_position = geometry.Point((0.2, 0.5))
    transition = pw.transition(start_position, puddle_world.Action.RIGHT)

    # We should stop at the inner wall puddle.
    expected_end_position = geometry.Point((0.4, 0.5))

    self.assertEqual(transition.state, start_position)
    self.assertPointsAlmostEqual(transition.next_state, expected_end_position)

  @parameterized.named_parameters(
      dict(testcase_name='up',
           start_position=geometry.Point((0.7, 0.99)),
           action=puddle_world.Action.UP),
      dict(testcase_name='right',
           start_position=geometry.Point((0.91, 0.4)),
           action=puddle_world.Action.RIGHT),
      dict(testcase_name='down',
           start_position=geometry.Point((0.0, 0.0)),
           action=puddle_world.Action.DOWN),
      dict(testcase_name='left',
           start_position=geometry.Point((0.1, 0.7)),
           action=puddle_world.Action.LEFT))
  def test_puddle_world_obeys_arena_boundaries(
      self,
      start_position,
      action):
    pw = puddle_world.PuddleWorld((),
                                  goal_position=geometry.Point((0.5, 0.5)),
                                  noise=0.0,
                                  thrust=0.1)

    transition = pw.transition(start_position, action)

    self.assertBetween(transition.next_state.x, 0.0, 1.0)
    self.assertBetween(transition.next_state.y, 0.0, 1.0)

  # TODO: Test for terminal calculation.
  # TODO: Test noise is applied correctly.

if __name__ == '__main__':
  absltest.main()
