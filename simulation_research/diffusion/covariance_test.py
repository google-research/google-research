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

"""Tests for structured covariance matrices."""

from absl.testing import absltest
import jax
import jax.numpy as jnp

from simulation_research.diffusion.diffusion import BrownianCovariance
from simulation_research.diffusion.diffusion import PinkCovariance
from simulation_research.diffusion.diffusion import WhiteCovariance


class CovarianceTest(absltest.TestCase):

  def test_covariance_logdet(self):
    """Test whether logdet method matches numpy logdet with dense matrix."""
    for sqrt_b in [WhiteCovariance, BrownianCovariance, PinkCovariance]:
      n = 64
      ident = jnp.eye(n)[:, :, None]
      sqrt_b_dense = sqrt_b.forward(ident)[Ellipsis, 0]
      _, slogdet = jnp.linalg.slogdet(sqrt_b_dense)
      logdet = slogdet * 2
      logdet2 = sqrt_b.logdet((ident + jnp.zeros(2)).shape)[0]
      err = (jnp.abs(logdet - logdet2) /
             jnp.maximum(jnp.abs(logdet), 1.)).mean()
      assert err < 1e-4, f"logdet fails with error {err} on {sqrt_b}"

  def test_covariance_inverse(self):
    """Test covariance forward and inverse are in fact inverses of each other."""
    for sqrt_b in [WhiteCovariance, BrownianCovariance, PinkCovariance]:
      n = 64
      identity = jnp.eye(n)[:, :, None]
      sqrt_b_dense = sqrt_b.forward(identity)[Ellipsis, 0]
      invsqrt_b_dense = sqrt_b.inverse(identity)[Ellipsis, 0]
      with jax.default_matmul_precision("float32"):
        id2 = sqrt_b_dense @ invsqrt_b_dense  # pylint: disable=invalid-name
      err2 = jnp.linalg.norm(id2 - jnp.eye(id2.shape[0]), ord=2)
      assert err2 < 1e-4, f"inverse of forward fails with error {err2} on {sqrt_b}"


if __name__ == "__main__":
  absltest.main()
