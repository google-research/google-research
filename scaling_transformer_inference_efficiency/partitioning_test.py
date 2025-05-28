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

"""Tests for partitioning."""

import os

from absl.testing import absltest
import jax
from jax import core
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

 import resources
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import partitioning


_TOY_HPARAMS = checkpoint.HParams(
    layers=3,
    embed=128,
    ff=256,
    heads=2,
    qkv=32,
    max_len=128,
    vocab=32128,
)


class PartitioningTest(absltest.TestCase):


  def test_copy_to_device_from_shape(self):
    shape = core.ShapedArray((4, 4), dtype=jnp.bfloat16)
    mesh = partitioning.make_mesh()
    x = partitioning.copy_to_device(shape,
                                    NamedSharding(mesh, P('x', ('y', 'z'))),
                                    shape)
    self.assertEqual(x.shape, shape.shape)
    self.assertEqual(x.dtype, shape.dtype)

  def test_copy_to_device_from_array(self):
    array = jnp.zeros((4, 4), jnp.bfloat16)
    shape = core.ShapedArray((4, 4), dtype=jnp.bfloat16)
    mesh = partitioning.make_mesh()
    x = partitioning.copy_to_device(array,
                                    NamedSharding(mesh, P('x', ('y', 'z'))),
                                    shape)
    self.assertEqual(x.shape, array.shape)
    self.assertEqual(x.dtype, array.dtype)


if __name__ == '__main__':
  absltest.main()
