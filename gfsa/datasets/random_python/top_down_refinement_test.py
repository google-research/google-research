# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Tests for gfsa.datasets.google.random_python.top_down_refinement."""

import collections
from absl.testing import absltest
import numpy as np
from gfsa.datasets.random_python import top_down_refinement


class TopDownRefinementTest(absltest.TestCase):

  def test_construct_simple(self):
    """Simple test that we can fill different hole types."""

    class FooTemplate(top_down_refinement.HoleFillerTemplate):
      fills_type = "foo"
      required_cost = 2

      def fill(self, hole, rng):
        return top_down_refinement.ThingWithHoles(
            1, [top_down_refinement.Hole("bar", None)], lambda bar: "foo" + bar)

    class BarTemplate(top_down_refinement.HoleFillerTemplate):
      fills_type = "bar"
      required_cost = 1

      def fill(self, hole, rng):
        return top_down_refinement.ThingWithHoles(1, [], lambda: "bar")

    result = top_down_refinement.top_down_construct(
        root_object=top_down_refinement.ThingWithHoles(
            0, [top_down_refinement.Hole("foo", None)], lambda foo: foo),
        target_cost=2,
        refinement_distribution=top_down_refinement.RefinementDistribution(
            weighted_templates=[
                top_down_refinement.WeightedTemplate(FooTemplate(), 1),
                top_down_refinement.WeightedTemplate(BarTemplate(), 1),
            ],
            hole_selection_weights={
                "foo": 1,
                "bar": 1
            }))

    self.assertEqual(result, "foobar")

  def test_cost_and_precedence(self):
    """Test that we use highest-precedence rules unless constrained by cost."""

    class BigTemplate(top_down_refinement.HoleFillerTemplate):
      fills_type = "thing"
      required_cost = 2

      def fill(self, hole, rng):
        return top_down_refinement.ThingWithHoles(
            1, [top_down_refinement.Hole("thing", None)], lambda t: "a" + t)

    class SmallTemplate(top_down_refinement.HoleFillerTemplate):
      fills_type = "thing"
      required_cost = 1

      def fill(self, hole, rng):
        return top_down_refinement.ThingWithHoles(1, [], lambda: "b")

    result = top_down_refinement.top_down_construct(
        root_object=top_down_refinement.ThingWithHoles(
            0, [top_down_refinement.Hole("thing", None)], lambda t: t),
        target_cost=10,
        refinement_distribution=top_down_refinement.RefinementDistribution(
            weighted_templates=[
                top_down_refinement.WeightedTemplate(
                    BigTemplate(), 1, precedence=1),
                top_down_refinement.WeightedTemplate(
                    SmallTemplate(), 10000, precedence=0),
            ],
            hole_selection_weights={"thing": 1}))

    self.assertEqual(result, "aaaaaaaaab")

  def test_random_sampling(self):
    """Test that holes and templates are chosen proportional to weights."""

    class A1Template(top_down_refinement.HoleFillerTemplate):
      fills_type = "a"
      required_cost = 2

      def fill(self, hole, rng):
        return top_down_refinement.ThingWithHoles(2, [], lambda: "a1")

    class A2Template(top_down_refinement.HoleFillerTemplate):
      fills_type = "a"
      required_cost = 2

      def fill(self, hole, rng):
        return top_down_refinement.ThingWithHoles(2, [], lambda: "a2")

    class AFallbackTemplate(top_down_refinement.HoleFillerTemplate):
      fills_type = "a"
      required_cost = 1

      def fill(self, hole, rng):
        return top_down_refinement.ThingWithHoles(2, [], lambda: "af")

    class B1Template(top_down_refinement.HoleFillerTemplate):
      fills_type = "b"
      required_cost = 2

      def fill(self, hole, rng):
        return top_down_refinement.ThingWithHoles(2, [], lambda: "b1")

    class BFallbackTemplate(top_down_refinement.HoleFillerTemplate):
      fills_type = "b"
      required_cost = 1

      def fill(self, hole, rng):
        return top_down_refinement.ThingWithHoles(1, [], lambda: "bf")

    counts = collections.Counter()
    rng = np.random.RandomState(1234)
    trials = 10000
    for _ in range(trials):
      result = top_down_refinement.top_down_construct(
          root_object=top_down_refinement.ThingWithHoles(
              0, [
                  top_down_refinement.Hole("a", None),
                  top_down_refinement.Hole("b", None)
              ], lambda a, b: (a, b)),
          target_cost=3,
          refinement_distribution=top_down_refinement.RefinementDistribution(
              weighted_templates=[
                  top_down_refinement.WeightedTemplate(A1Template(), 1),
                  top_down_refinement.WeightedTemplate(A2Template(), 2),
                  top_down_refinement.WeightedTemplate(
                      AFallbackTemplate(), 1, precedence=0),
                  top_down_refinement.WeightedTemplate(B1Template(), 1),
                  top_down_refinement.WeightedTemplate(
                      BFallbackTemplate(), 1, precedence=0),
              ],
              hole_selection_weights={
                  "a": 3,
                  "b": 1
              }),
          rng=rng)
      counts[result] += 1

    # Assert that counts are within one standard deviation of the mean (which is
    # sufficient for the fixed seed above).
    p_a1_bf = (3 / 4) * (1 / 3)
    np.testing.assert_allclose(
        counts["a1", "bf"],
        trials * p_a1_bf,
        atol=np.sqrt(trials * p_a1_bf * (1 - p_a1_bf)))

    p_a2_bf = (3 / 4) * (2 / 3)
    np.testing.assert_allclose(
        counts["a2", "bf"],
        trials * p_a2_bf,
        atol=np.sqrt(trials * p_a2_bf * (1 - p_a2_bf)))

    p_af_b1 = 1 / 4
    np.testing.assert_allclose(
        counts["af", "b1"],
        trials * p_af_b1,
        atol=np.sqrt(trials * p_af_b1 * (1 - p_af_b1)))

  def test_deterministic(self):

    class RandIntTemplate(top_down_refinement.HoleFillerTemplate):
      fills_type = "a"
      required_cost = 1

      def fill(self, hole, rng):
        v = rng.randint(10000000)
        return top_down_refinement.ThingWithHoles(1, [], lambda: v)

    values = []
    for _ in range(2):
      rng = np.random.RandomState(1234)
      values.append(
          top_down_refinement.top_down_construct(
              root_object=top_down_refinement.ThingWithHoles(
                  0, [
                      top_down_refinement.Hole("a", None),
                      top_down_refinement.Hole("a", None)
                  ], lambda a, b: (a, b)),
              target_cost=2,
              refinement_distribution=top_down_refinement
              .RefinementDistribution(
                  weighted_templates=[
                      top_down_refinement.WeightedTemplate(
                          RandIntTemplate(), 1)
                  ],
                  hole_selection_weights={"a": 1}),
              rng=rng))

    # Deterministic across seeds, but random across holes
    self.assertEqual(values[0], values[1])
    self.assertNotEqual(values[0][0], values[0][1])


if __name__ == "__main__":
  absltest.main()
