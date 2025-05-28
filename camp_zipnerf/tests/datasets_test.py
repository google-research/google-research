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

"""Tests for datasets."""

from absl.testing import absltest
from internal import camera_utils
from internal import configs
from internal import datasets
from jax import random
import numpy as np


class DummyDataset(datasets.Dataset):

  def _load_renderings(self, config):
    """Generates dummy image and pose data."""
    self._n_examples = 2
    self.height = 3
    self.width = 4
    self._resolution = self.height * self.width
    self.focal = 5.0
    self.pixtocams = np.linalg.inv(
        camera_utils.intrinsic_matrix(
            self.focal, self.focal, self.width * 0.5, self.height * 0.5
        )
    )

    rng = random.PRNGKey(0)

    key, rng = random.split(rng)
    images_shape = (self._n_examples, self.height, self.width, 3)
    self.images = random.uniform(key, images_shape)

    key, rng = random.split(rng)
    self.camtoworlds = np.stack(
        [
            camera_utils.viewmatrix(*random.normal(k, (3, 3)))
            for k in random.split(key, self._n_examples)
        ],
        axis=0,
    )


class DatasetsTest(absltest.TestCase):

  def test_dataset_batch_creation(self):
    np.random.seed(0)
    config = configs.Config(batch_size=8)

    # Check shapes are consistent across all ray attributes.
    for split in ['train', 'test']:
      dummy_dataset = DummyDataset(split, '', config)
      rays = datasets.RayBatcher(dummy_dataset).peek().rays
      sh_gt = rays.origins.shape[:-1]
      for z in rays.__dict__.values():
        if z is not None:
          self.assertEqual(z.shape[:-1], sh_gt)

    # Check test batch generation matches golden data.
    dummy_dataset = DummyDataset('test', '', config)
    batch = datasets.RayBatcher(dummy_dataset).peek()

    rgb = batch.rgb.ravel()
    rgb_gt = np.array([
        0.5289556,
        0.28869557,
        0.24527192,
        0.12083626,
        0.8904066,
        0.6259936,
        0.57573485,
        0.09355974,
        0.8017353,
        0.538651,
        0.4998169,
        0.42061496,
        0.5591258,
        0.00577283,
        0.6804651,
        0.9139203,
        0.00444758,
        0.96962905,
        0.52956843,
        0.38282406,
        0.28777933,
        0.6640035,
        0.39736128,
        0.99495006,
        0.13100398,
        0.7597165,
        0.8532667,
        0.67468107,
        0.6804743,
        0.26873016,
        0.60699487,
        0.5722265,
        0.44482303,
        0.6511061,
        0.54807067,
        0.09894073,
    ])
    np.testing.assert_allclose(rgb, rgb_gt, atol=1e-4, rtol=1e-4)

    ray_origins = batch.rays.origins.ravel()
    ray_origins_gt = np.array([
        -0.20050469,
        -0.6451472,
        -0.8818224,
        -0.20050469,
        -0.6451472,
        -0.8818224,
        -0.20050469,
        -0.6451472,
        -0.8818224,
        -0.20050469,
        -0.6451472,
        -0.8818224,
        -0.20050469,
        -0.6451472,
        -0.8818224,
        -0.20050469,
        -0.6451472,
        -0.8818224,
        -0.20050469,
        -0.6451472,
        -0.8818224,
        -0.20050469,
        -0.6451472,
        -0.8818224,
        -0.20050469,
        -0.6451472,
        -0.8818224,
        -0.20050469,
        -0.6451472,
        -0.8818224,
        -0.20050469,
        -0.6451472,
        -0.8818224,
        -0.20050469,
        -0.6451472,
        -0.8818224,
    ])
    np.testing.assert_allclose(
        ray_origins, ray_origins_gt, atol=1e-4, rtol=1e-4
    )

    ray_dirs = batch.rays.directions.ravel()
    ray_dirs_gt = np.array([
        0.24370372,
        0.89296186,
        -0.5227117,
        0.05601424,
        0.8468699,
        -0.57417226,
        -0.13167524,
        0.8007779,
        -0.62563276,
        -0.31936473,
        0.75468594,
        -0.67709327,
        0.17780769,
        0.96766925,
        -0.34928587,
        -0.0098818,
        0.9215773,
        -0.4007464,
        -0.19757128,
        0.87548524,
        -0.4522069,
        -0.38526076,
        0.82939327,
        -0.5036674,
        0.11191163,
        1.0423766,
        -0.17586003,
        -0.07577785,
        0.9962846,
        -0.22732055,
        -0.26346734,
        0.95019263,
        -0.2787811,
        -0.45115682,
        0.90410066,
        -0.3302416,
    ])
    np.testing.assert_allclose(ray_dirs, ray_dirs_gt, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
