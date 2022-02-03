# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for graph_sampler."""

from absl.testing import absltest
from graph_sampler import graph_sampler


def is_simple(graph):
  if len(graph) == len(set(graph)):
    return 1.0
  else:
    return 0.0


class GraphSamplerTest(absltest.TestCase):

  def test_can_be_connected(self):
    self.assertTrue(graph_sampler.can_be_connected([1, 1]))
    self.assertTrue(graph_sampler.can_be_connected([4, 1, 1, 1, 1]))
    self.assertTrue(graph_sampler.can_be_connected([4, 4, 4, 3, 2, 1]))

    self.assertFalse(graph_sampler.can_be_connected([1, 1, 1, 1]))
    self.assertFalse(graph_sampler.can_be_connected([3, 3, 1, 1, 1, 1, 1, 1]))

  def test_unique_graph_0(self):
    graph, weight = graph_sampler.sample_graph([4, 4])
    self.assertEqual(weight, 1.0)
    self.assertEqual(graph, 4 * [(0, 1)])

    est, std = graph_sampler.estimate_number_of_graphs([4, 4])
    self.assertEqual(est, 1)
    self.assertEqual(std, 0)

  def test_unique_graph_1(self):
    graph, weight = graph_sampler.sample_graph([2, 2, 2])
    self.assertEqual(weight, 1.0)
    self.assertEqual(graph, [(0, 1), (0, 2), (1, 2)])

    est, std = graph_sampler.estimate_number_of_graphs([2, 2, 2])
    self.assertEqual(est, 1)
    self.assertEqual(std, 0)

  def test_estimate(self):
    # There at 6 graphs with this degree vector.
    degrees = [2, 2, 2, 2]
    est, std = graph_sampler.estimate_number_of_graphs(degrees, rng_seed=0)
    self.assertLess(abs(6.0 - est), 3 * std)

  def test_cubic_simples(self):

    # Number of simple cubic graphs on 2*n nodes for n between 0 and 12, from
    # https://oeis.org/A002829
    num_simple_cubics = [
        1, 0, 1, 70, 19355, 11180820, 11555272575, 19506631814670,
        50262958713792825, 187747837889699887800, 976273961160363172131825,
        6840300875426184026353242750, 62870315446244013091262178375075,
        741227949070136911068308523257857500
    ]

    for i in range(2, 14):
      degrees = [3] * (2 * i)
      est, std = graph_sampler.estimate_number_of_graphs(
          degrees, weight_func=is_simple, rng_seed=0)
      true_error = est - num_simple_cubics[i]
      self.assertLess(abs(true_error), 3 * std)

  def test_quartic_simples(self):

    # Number of simple 4-valent graphs on n nodes for n between 0 and 17,from
    # https://oeis.org/A005815
    num_simple_quartics = [
        1, 0, 0, 0, 0, 1, 15, 465, 19355, 1024380, 66462606, 5188453830,
        480413921130, 52113376310985, 6551246596501035, 945313907253606891,
        155243722248524067795, 28797220460586826422720
    ]

    for i in range(5, 18):
      degrees = [4] * i
      est, std = graph_sampler.estimate_number_of_graphs(
          degrees, weight_func=is_simple, rng_seed=0)
      true_error = est - num_simple_quartics[i]
      self.assertLess(abs(true_error), 3 * std)

  def test_zero_weight(self):
    # Make sure that we terminate eventually when given a weight func that is
    # all zero. This can occur in our real cases if we give weight 0 to
    # disconnected graphs and this degree vector can't make connected graphs.

    degrees = [1] * 4

    # The trivial case is that there are a fixed number of samples
    est, std = graph_sampler.estimate_number_of_graphs(
        degrees, weight_func=lambda _: 0.0, rng_seed=0)
    self.assertEqual(est, 0.0)
    self.assertEqual(std, 0.0)

    # Zero weight case with only relative precision will go forever, but if we
    # provide both relative and absolute, it should terminate.
    est, std = graph_sampler.estimate_number_of_graphs(
        degrees,
        weight_func=lambda _: 0.0,
        relative_precision=0.01,
        absolute_precision=0.5,
        rng_seed=0)
    self.assertEqual(est, 0.0)
    self.assertEqual(std, 0.0)

  # TODO(geraschenko): test estimate_expected_value


if __name__ == '__main__':
  absltest.main()
