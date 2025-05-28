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

"""Tests the polynomial attention mechanism."""

from absl.testing import absltest
import jax.numpy as jnp
from polysketchformer import polynomial_attention


class PolynomialAttentionTest(absltest.TestCase):

  def test_polynomial_attention(self):
    query = jnp.array(
        [
            [
                [-0.17992814, 0.13316451, -0.41577485],
                [-0.98938537, -0.30148995, 0.44682142],
                [-0.5543319, 0.9078981, 0.28808194],
            ],
            [
                [-0.3609977, 0.4550869, -0.39748457],
                [0.09630831, -0.62415534, 0.54521275],
                [-0.26872134, -0.16309649, 0.41216722],
            ],
        ],
        dtype=jnp.float32,
    )
    key = jnp.array(
        [
            [
                [0.02172906, -0.32989815, -0.31889325],
                [-0.5485013, 1.947803, -0.09391224],
                [-0.4783525, 0.79820323, -0.38924676],
            ],
            [
                [-1.1240734, -0.27117178, 0.18005717],
                [-1.1917279, -0.4577331, -0.15926844],
                [-0.9664087, 0.12104954, -0.19269566],
            ],
        ],
        dtype=jnp.float32,
    )
    value = jnp.array(
        [
            [
                [-1.0134954, 0.10757159, 0.5540908],
                [0.04409654, -0.29590085, -0.01144499],
                [-0.29636988, 0.15430054, -0.37997785],
            ],
            [
                [1.1507518, -0.5717865, -0.4840106],
                [0.49385402, -0.29801255, -0.45857057],
                [0.09117171, -0.6056904, 0.43794355],
            ],
        ],
        dtype=jnp.float32,
    )
    expected_with_normalization_1_causal = jnp.array(
        [
            [
                [-5.2276322e-05, 5.5485666e-06, 2.8580127e-05],
                [-1.5096548e-05, -1.4719665e-05, 8.9632040e-06],
                [2.9759085e-02, -2.6589623e-01, -2.1500811e-02],
            ],
            [
                [2.2682974e-03, -1.1270734e-03, -9.5405447e-04],
                [7.6270266e-04, -3.8160163e-04, -3.3332876e-04],
                [4.0092841e-02, -2.0870507e-02, -1.9384813e-02],
            ],
        ],
        dtype=jnp.float32,
    )
    expected_with_normalization_1_non_causal = jnp.array(
        [
            [
                [-3.4789362e-03, -4.7322935e-03, -5.9931013e-03],
                [-1.8615023e-05, -1.2887557e-05, 4.4518115e-06],
                [2.9759085e-02, -2.6589623e-01, -2.1500811e-02],
            ],
            [
                [9.7960951e-03, -3.3337876e-02, 1.8235730e-02],
                [1.2667836e-03, -3.7565373e-03, 2.1103104e-03],
                [4.0092841e-02, -2.0870507e-02, -1.9384813e-02],
            ],
        ],
        dtype=jnp.float32,
    )
    implementation_result_causal = polynomial_attention.polynomial_attention(
        query, key, value, is_causal=True
    )
    implementation_result_non_causal = (
        polynomial_attention.polynomial_attention(
            query, key, value, is_causal=False
        )
    )

    self.assertTrue(
        jnp.allclose(
            implementation_result_causal,
            expected_with_normalization_1_causal,
            rtol=1e-3,
            atol=1e-7,
        )
    )
    self.assertTrue(
        jnp.allclose(
            implementation_result_non_causal,
            expected_with_normalization_1_non_causal,
            rtol=1e-3,
            atol=1e-7,
        )
    )


if __name__ == "__main__":
  absltest.main()
