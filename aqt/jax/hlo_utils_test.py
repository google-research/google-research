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

"""Tests for aqt.jax.hlo_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from jax import random
import jax.numpy as jnp

from aqt.jax import hlo_utils


class HloUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='one_add',
          fn=lambda x: x + 1,
          fn_args=[1],
          ops_regex=r'add',
          exp_count=1,
      ),
      dict(
          testcase_name='two_adds',
          fn=lambda x, y: x + y + 1,
          fn_args=[1, 2],
          ops_regex=r'add',
          exp_count=2,
      ),
      dict(
          testcase_name='one_mult',
          fn=lambda x, y: x * y,
          fn_args=[2, 3],
          ops_regex=r'multiply',
          exp_count=1,
      ),
  )
  def test_load_hlo_proto_from_jax_fn_and_count_ops(self, fn,
                                                    fn_args, ops_regex,
                                                    exp_count):
    hlo_proto = hlo_utils.load_hlo_proto_from_jax_fn(
        fn, *fn_args)
    count = hlo_utils.count_ops_in_hlo_proto(hlo_proto, ops_regex=ops_regex)
    self.assertEqual(count, exp_count)

  class TestModelWith2DenseLayers(nn.Module):
    """Test model with two Dense layers."""

    @nn.compact
    def __call__(self, inputs, dtype=jnp.float32):
      x = nn.linear.Dense(features=2)(inputs)
      output = nn.linear.Dense(features=3)(x)
      return output

  def test_load_hlo_proto_from_model_and_count_ops(self):
    input_shapes = [(1, 2)]
    # with nn.stateful() as init_state:
    test_model = self.TestModelWith2DenseLayers()
    init_state = test_model.init(
        random.PRNGKey(0), *[jnp.ones(shape) for shape in input_shapes])

    hlo_proto = hlo_utils.load_hlo_proto_from_model(test_model, init_state,
                                                    input_shapes)
    count = hlo_utils.count_ops_in_hlo_proto(hlo_proto, ops_regex=r'dot')
    self.assertEqual(count, 2)


if __name__ == '__main__':
  absltest.main()
