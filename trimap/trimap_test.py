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

"""Tests for TriMap."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpy.testing as npt
from pynndescent import NNDescent

from trimap import trimap

python_version = 'PY3'


class TestTrimap(absltest.TestCase):

  def test_sliced_distances(self):
    num_points = 100
    dim = 50
    key = random.PRNGKey(42)
    use_keys = random.split(key, num=3)
    indices1 = random.randint(use_keys[0], shape=(int(1.5 * num_points),),
                              minval=0, maxval=num_points)
    indices2 = random.randint(use_keys[1], shape=(int(1.5 * num_points),),
                              minval=0, maxval=num_points)
    inputs = random.normal(use_keys[2], shape=(num_points, dim))
    distance_fn = trimap.euclidean_dist
    dist_sliced = trimap.sliced_distances(
        indices1, indices2, inputs, distance_fn)
    dist_direct = distance_fn(inputs[indices1], inputs[indices2])
    npt.assert_equal(np.array(dist_sliced), np.array(dist_direct))

  def test_rejection_sample(self):
    in1dvec = jax.vmap(jnp.in1d)
    maxval = 10000
    key = random.PRNGKey(42)
    rejects = random.choice(key, maxval, shape=(100, 10), replace=False)
    samples = trimap.rejection_sample(key, (100, 100), maxval, rejects)
    npt.assert_equal(jnp.sum(jnp.any(in1dvec(samples, rejects))), jnp.array(0))

  def test_generate_triplets(self):
    key = random.PRNGKey(42)
    n_points = 1000
    n_inliers = 10
    n_outliers = 5
    n_random = 3
    n_extra = min(n_inliers + 50, n_points)
    # Currently testing it only for 'euclidean' distance. The test for other
    # cases breaks due to issues with the knn search NNDescent package, but
    # it works fine when tested in a colab.
    for distance in ['euclidean']:
      inputs = np.random.normal(size=(n_points, 100))
      index = NNDescent(inputs, metric=distance)
      index.prepare()
      neighbors = index.query(inputs, n_extra)[0]
      neighbors = np.concatenate(
          (np.arange(n_points).reshape([-1, 1]), neighbors), 1)
      distance_fn = trimap.get_distance_fn(distance)
      _, _, sig = trimap.find_scaled_neighbors(inputs, neighbors, distance_fn)
      triplets, _ = trimap.generate_triplets(
          key,
          inputs,
          n_inliers=n_inliers,
          n_outliers=n_outliers,
          n_random=n_random,
          distance=distance)
      similar_pairs_distances = distance_fn(inputs[triplets[:, 0]],
                                            inputs[triplets[:, 1]]) ** 2
      similar_pairs_distances /= (sig[triplets[:, 0]] * sig[triplets[:, 1]])
      outlier_pairs_distances = distance_fn(inputs[triplets[:, 0]],
                                            inputs[triplets[:, 2]]) ** 2
      outlier_pairs_distances /= (sig[triplets[:, 0]] * sig[triplets[:, 2]])
      npt.assert_array_less(similar_pairs_distances, outlier_pairs_distances)
    n_knn_triplets = inputs.shape[0] * n_inliers * n_outliers
    n_random_triplets = inputs.shape[0] * n_random
    npt.assert_equal(triplets.shape, [n_knn_triplets + n_random_triplets, 3])

  def test_transform(self):
    key = random.PRNGKey(42)
    key, use_key = random.split(key)
    inputs = random.normal(use_key, shape=(10000, 256))
    embedding = trimap.transform(key, inputs, n_dims=4)
    npt.assert_equal(embedding.shape[0], inputs.shape[0])
    npt.assert_equal(embedding.shape[1], 4)


if __name__ == '__main__':
  absltest.main()
