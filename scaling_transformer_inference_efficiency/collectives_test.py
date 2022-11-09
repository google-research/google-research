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

"""Tests for collective einsum implementations."""

from absl.testing import absltest
import jax
from jax.experimental.maps import Mesh
from jax.experimental.maps import xmap
import jax.numpy as jnp
import numpy as np

from scaling_transformer_inference_efficiency import collectives

X_SIZE = 4
NUM_LAYERS = 2
LAYER = 1


def mesh4():
  devices = jax.devices()
  assert len(devices) == X_SIZE
  return Mesh(devices, ('x',))


def make(shape):
  x = jnp.float32(jnp.arange(np.product(shape)))
  return jnp.reshape(x, shape)


class CollectivesTest(absltest.TestCase):

  def test_matmul_reducescatter_no_collective(self):
    # matmul_reducescatter. Basic form is:
    #   [a, b.X] @ [b.X, c] -> [a, c.X]
    #
    # To express this in hard xmap we split out the dimensions that will be
    # sharded, so it becomes:
    #
    #   [a, B.X, b] @ [B.X, b, C, c] -> [a, C.X, c]
    lhs = make((3, X_SIZE, 2))
    rhs = make((NUM_LAYERS, X_SIZE, 2, X_SIZE, 5))
    expected = jnp.einsum('aBb,BbCc->aCc', lhs, rhs[LAYER])
    expected = jax.device_get(expected)

    def fn(lhs, rhs):
      result = collectives.matmul_reducescatter_no_collective(
          'ab,bCc->aCc', lhs, rhs, scatter_dimension=(1, 1),
          axis_name='x', layer=LAYER, subsplit_axis=-1)
      # Squeeze out C.X dim, which is now 1 on each device.
      result = jax.lax.squeeze(result, (1,))
      return result

    with mesh4():
      y = xmap(
          fn,
          in_axes=({
              1: 'x'
          }, {
              1: 'x'
          }),
          out_axes={1: 'x'},
          axis_resources={'x': 'x'})(lhs, rhs)
      y = jax.device_get(y)
      np.testing.assert_allclose(expected, y)


if __name__ == '__main__':
  absltest.main()
