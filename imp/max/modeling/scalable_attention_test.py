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

"""Tests for the scalable attention library."""

from absl.testing import absltest
from absl.testing import parameterized
import aqt.jax.v2.config as aqt_config
from jax import lax
from jax import random

from imp.max.core import utils
from imp.max.modeling import kernel_transformation as kt
from imp.max.modeling import scalable_attention as sa


SEED = 0
BATCH_ONE = 5
BATCH_TWO = 7
LENGTH = 20
NB_HEADS = 2
QK_DIM = 16
V_DIM = 12
NB_COORDINATES = 14
NUM_KERNEL_FEATURES = 16
AQT_CFG = aqt_config.DotGeneral(fwd=aqt_config.DotGeneralRaw.make(8, 8),
                                dlhs=aqt_config.DotGeneralRaw.make(8, 8),
                                drhs=aqt_config.DotGeneralRaw.make(8, 8))


class NonAutoregAttentionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'regular_two_batch_dims',
          True,
          kt.exp_softmax_kernel_transformation,
          True,
          False,
          False,
          False,
          False,
          0,
          lax.dot_general,
      ),
      (
          'regular_two_batch_dims_with_quantization',
          True,
          kt.exp_softmax_kernel_transformation,
          True,
          False,
          False,
          False,
          False,
          0,
          utils.make_dot_general(AQT_CFG),
      ),
      (
          'simplex_two_batch_dims',
          True,
          kt.exp_softmax_kernel_transformation,
          True,
          True,
          False,
          False,
          False,
          0,
          lax.dot_general,
      ),
      (
          'flt_regular_two_batch_dims',
          True,
          kt.exp_softmax_kernel_transformation,
          True,
          False,
          False,
          True,
          False,
          0,
          lax.dot_general,
      ),
      (
          'flt_simplex_two_batch_dims',
          True,
          kt.exp_softmax_kernel_transformation,
          True,
          True,
          False,
          True,
          False,
          0,
          lax.dot_general,
      ),
      (
          'flt_spe_simplex_two_batch_dims',
          True,
          kt.exp_softmax_kernel_transformation,
          True,
          True,
          True,
          True,
          False,
          0,
          lax.dot_general,
      ),
      (
          'flt_spe_grpe_simplex_two_batch_dims',
          True,
          kt.exp_softmax_kernel_transformation,
          True,
          True,
          True,
          True,
          True,
          0,
          lax.dot_general,
      ),
      (
          'relu_flt_spe_grpe_simplex_two_batch_dims',
          True,
          kt.relu_kernel_transformation,
          True,
          True,
          True,
          True,
          True,
          0,
          lax.dot_general,
      ),
      (
          'expplus_flt_spe_grpe_simplex_two_batch_dims',
          True,
          kt.expplus_softmax_kernel_transformation,
          True,
          True,
          True,
          True,
          True,
          0,
          lax.dot_general,
      ),
      (
          'exparf_flt_spe_grpe_simplex_two_batch_dims',
          True,
          kt.exparf_softmax_kernel_transformation,
          True,
          True,
          True,
          True,
          True,
          0,
          lax.dot_general,
      ),
      (
          'expsharp_flt_spe_grpe_simplex_two_batch_dims',
          True,
          kt.expsharp_softmax_kernel_transformation,
          True,
          True,
          True,
          True,
          True,
          0,
          lax.dot_general,
      ),
      (
          'expsharp_flt_spe_grpe_simplex_one_batch_dim',
          False,
          kt.expsharp_softmax_kernel_transformation,
          True,
          True,
          True,
          True,
          True,
          0,
          lax.dot_general,
      ),
  )
  def test_general_favor_attention(
      self,
      two_batch_dims,
      kernel_transformation,
      use_random_projections,
      simplex,
      spe,
      flt,
      grpe,
      bf_attention_global_size,
      dot_general,
  ):
    batch_one = BATCH_ONE
    batch_two = BATCH_TWO
    length = LENGTH
    nb_heads = NB_HEADS
    qk_dim = QK_DIM
    v_dim = V_DIM
    nb_coordinates = NB_COORDINATES
    num_kernel_features = NUM_KERNEL_FEATURES

    if two_batch_dims:
      query = random.normal(
          key=random.key(SEED),
          shape=(batch_one, batch_two, length, nb_heads, qk_dim),
      )
      key = random.normal(
          key=random.key(SEED),
          shape=(batch_one, batch_two, length, nb_heads, qk_dim),
      )
      value = random.normal(
          key=random.key(SEED),
          shape=(batch_one, batch_two, length, nb_heads, v_dim),
      )
    else:
      query = random.normal(
          key=random.key(SEED),
          shape=(batch_one, length, nb_heads, qk_dim),
      )
      key = random.normal(
          key=random.key(SEED),
          shape=(batch_one, length, nb_heads, qk_dim),
      )
      value = random.normal(
          key=random.key(SEED),
          shape=(batch_one, length, nb_heads, v_dim),
      )
    coords = random.normal(
        key=random.key(SEED),
        shape=(length, nb_coordinates),
    )
    flt_num_blobs_per_head = 25
    flt_num_rand_features = 32
    flt_params = random.normal(
        key=random.key(SEED),
        shape=(
            query.shape[-2],
            (nb_coordinates + 2) * flt_num_blobs_per_head,
        ),
    )
    grpe_params = random.normal(
        key=random.key(SEED),
        shape=(query.shape[-2], 2 * query.shape[-3] - 1),
    )

    result = sa.general_favor_attention(
        query=query,
        key=key,
        value=value,
        coords=coords,
        kernel_transformation=kernel_transformation,
        num_kernel_features=num_kernel_features,
        use_random_projections=use_random_projections,
        simplex=simplex,
        spe=spe,
        flt=flt,
        flt_params=flt_params,
        flt_num_blobs_per_head=flt_num_blobs_per_head,
        flt_num_rand_features=flt_num_rand_features,
        grpe=grpe,
        grpe_params=grpe_params,
        bf_attention_global_size=bf_attention_global_size,
        dot_general=dot_general,
        precision=None,
    )
    if two_batch_dims:
      assert result.shape == (batch_one, batch_two, length, nb_heads, v_dim)
    else:
      assert result.shape == (batch_one, length, nb_heads, v_dim)

if __name__ == '__main__':
  absltest.main()
