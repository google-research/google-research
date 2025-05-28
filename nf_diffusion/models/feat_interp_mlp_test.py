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

"""Tests for feat_interp_mlp."""

from absl.testing import absltest
import jax
import jax.numpy as jnp

from nf_diffusion.models import feat_interp_mlp


class FeatInterpMlpADTest(absltest.TestCase):

  def test_model_2d_shape(self):
    model = feat_interp_mlp.FeatIterpNFMLP(
        inp_dim=2,
        hid_dim=[32, 32, 32],
        out_dim=3,
        num_emb=300,
        feat_dim=64,
        res=(32, 32),
    )
    rng = jax.random.PRNGKey(0)
    x = jnp.zeros((4, 3, 5, 2))
    i = jnp.zeros((4, 3), dtype=jnp.int32)
    params = model.init(rng, i, x)

    y = model.apply(params, i, x)
    self.assertEqual(y.shape[:-1], x.shape[:-1])
    f = params["params"]["emb"]["embedding"]
    self.assertEqual(f.shape, (300, 32 * 32 * 64))


if __name__ == "__main__":
  absltest.main()
