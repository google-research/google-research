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

"""Tests for generate_data."""

from absl.testing import absltest
from absl.testing import parameterized

import hypothesis
from jax import random
import numpy as np
from squiggles import generate_data


class GenerateSineNetTest(absltest.TestCase):

  @hypothesis.settings(deadline=None, derandomize=True)
  @hypothesis.given(
      random_key=hypothesis.strategies.integers(
          min_value=0, max_value=2**32 - 1),
      hidden_size=hypothesis.strategies.integers(min_value=1, max_value=300),
      num_points=hypothesis.strategies.integers(min_value=1, max_value=100))
  def test_generate_sine_net(self, random_key, hidden_size, num_points):
    latents, coords = generate_data._sine_net_latent_and_coords(
        random.PRNGKey(random_key), hidden_size, num_points)

    expected_latents_shape = (hidden_size, 4)
    expected_coords_shape = (num_points, 2)

    self.assertEqual(latents.shape, expected_latents_shape)
    self.assertEqual(coords.shape, expected_coords_shape)
    self.assertFalse(
        np.all(np.equal(latents, np.zeros(expected_latents_shape))))
    self.assertFalse(
        np.all(np.equal(latents, np.ones(expected_latents_shape))))
    self.assertFalse(
        np.all(np.equal(coords, np.zeros(expected_coords_shape))))
    self.assertFalse(
        np.all(np.equal(coords, np.ones(expected_coords_shape))))


class GenerateTaylorTest(absltest.TestCase):

  @hypothesis.settings(deadline=None, derandomize=True)
  @hypothesis.given(
      random_key=hypothesis.strategies.integers(
          min_value=0, max_value=2**32 - 1),
      hidden_size=hypothesis.strategies.integers(min_value=1, max_value=300),
      num_points=hypothesis.strategies.integers(min_value=1, max_value=100))
  def test_generate_taylor(self, random_key, hidden_size, num_points):
    latents, coords = generate_data._taylor_latent_and_coords(
        random.PRNGKey(random_key), hidden_size, num_points)

    expected_latents_shape = (2, hidden_size)
    expected_coords_shape = (num_points, 2)

    self.assertEqual(latents.shape, expected_latents_shape)
    self.assertEqual(coords.shape, expected_coords_shape)
    self.assertFalse(
        np.all(np.equal(latents, np.zeros(expected_latents_shape))))
    self.assertFalse(
        np.all(np.equal(latents, np.ones(expected_latents_shape))))
    self.assertFalse(
        np.all(np.equal(coords, np.zeros(expected_coords_shape))))
    self.assertFalse(
        np.all(np.equal(coords, np.ones(expected_coords_shape))))


class GenerateDataSetTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('taylor', generate_data.LatentSpace.TAYLOR, 0, 100, 0x0, 7, 100),
      ('sine_net', generate_data.LatentSpace.SINE_NET, 0, 100, 0x0, 7, 100))
  def test_generate_dataset(self, latent_space,
                            start_seed, end_seed, dataset_code,
                            hidden_size, num_points):
    latents, coords, labels = generate_data.generate_dataset(
        latent_space,
        start_seed,
        end_seed,
        dataset_code,
        hidden_size,
        num_points)

    num_samples = end_seed - start_seed
    assert len(latents) == num_samples
    assert len(labels) == num_samples
    assert len(coords) == num_samples

    if latent_space == generate_data.LatentSpace.TAYLOR:
      expected_latent_shape = (2, hidden_size)
    elif latent_space == generate_data.LatentSpace.SINE_NET:
      expected_latent_shape = (hidden_size, 4)
    expected_coord_shape = (num_points, 2)

    for latent, coord, label in zip(latents, coords, labels):
      assert latent.shape == expected_latent_shape
      assert coord.shape == expected_coord_shape
      assert isinstance(label, bool)

      self.assertFalse(
          np.all(np.equal(latent, np.zeros(expected_latent_shape))))
      self.assertFalse(
          np.all(np.equal(latent, np.ones(expected_latent_shape))))
      self.assertFalse(
          np.all(np.equal(coord, np.zeros(expected_coord_shape))))
      self.assertFalse(
          np.all(np.equal(coord, np.ones(expected_coord_shape))))

if __name__ == '__main__':
  absltest.main()
