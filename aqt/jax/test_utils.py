# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Test utility for aqt.jax."""

import math
import typing
from typing import Any, Mapping, Union

from jax.config import config
import jax.numpy as jnp
import numpy as np

from aqt.jax import quantization
from aqt.jax import stats


def assert_all_close_prec(
    exact,
    res,
    prec,
):
  """Assert that |exact - res| < 2^-prec.

  Args:
    exact: The exact result of the computation.
    res: The computed result.
    prec: The quantization precision.
  """
  if prec is None:
    np.testing.assert_allclose(exact, res)
  elif isinstance(prec, quantization.QuantOps.FloatQuant):
    prec = typing.cast(quantization.QuantOps.FloatQuant, prec)
    # relative-error = |x' - x| / |x|
    #  Assume x' & x differ only in significant bits
    #
    #  => exp(x') = exp(x)
    #     2^exp(x) | mant(x') - mant(x) | / 2^exp(x) mant(x)
    #  1 <= mant(x) < 2
    #       mant(x') - mant(x) < 1.xx...x111111... - 1.xx...x < 2^-sig_bits
    #
    #  where xx...x has length equal to the number of significant bits s.t.
    #  |x' - x| / |x| = 2^-sig_bits / |x| < 2^-sig_bits
    np.testing.assert_allclose(exact, res, rtol=2**-prec.fp_spec.sig_bits)
  else:
    # decimal precision corresponding to prec [number of bits]
    dec_prec = prec * math.log(2, 10)
    atol = math.pow(2, -dec_prec)
    # |a - b| <= abs(atol + rtol * abs(b))
    np.testing.assert_allclose(exact, res, rtol=0, atol=atol)


def _iterate_stats_recursive(state1, state2, layer_name):
  """Yields a tensor for each equivalent leaf in two states."""
  if isinstance(state1, stats.Stats):
    stats1 = typing.cast(stats.Stats, state1)
    stats2 = typing.cast(stats.Stats, state2)
    yield layer_name, stats1.mean, stats2.mean
    yield layer_name, stats1.mean_abs, stats2.mean_abs
    yield layer_name, stats1.mean_sq, stats2.mean_sq
  elif hasattr(state1, 'items'):
    for module_name, state1_module in state1.items():
      state2_module = state2[module_name]
      yield from _iterate_stats_recursive(state1_module, state2_module,
                                          f'{layer_name}/{module_name}')
  else:
    yield layer_name, state1, state2


def _iterate_stats(state1, state2):
  yield from _iterate_stats_recursive(state1['get_bounds'],
                                      state2['get_bounds'], '')
  if 'stats_tag' in state1:
    yield from _iterate_stats_recursive(state1['stats_tag'],
                                        state2['stats_tag'], '')


def assert_stats_are_equal(state1, state2):
  """Asserts that the activation statistics in two Flax states are almost equal."""

  for layer_name, state1_stats, state2_stats in _iterate_stats(state1, state2):
    # The tolerance was chosen empirically to make the tests pass reliably for a
    # 3-layer model. Possibly the tolerance has to be high because of floating
    # point accumulated error over 3 layers and because masked-out tokens still
    # exerts a tiny but non-zero padding on statistics since they are masked out
    # in the attention layers by subtracting a large but not infinite negative
    # value from their position in the  Q*K output before taking a softmax to
    # get attention weights.
    np.testing.assert_allclose(
        state1_stats,
        state2_stats,
        err_msg=f'Stats changed for layer {layer_name}',
        atol=.01)


def assert_stats_are_unequal(state1, state2):
  """Asserts that in at least one layer, the statistics in two Flax states are unequal."""

  for _, state1_stats, state2_stats in _iterate_stats(state1, state2):
    if not np.allclose(state1_stats, state2_stats, atol=.01):
      return  # We have found a layer where the stats are different

  # If we reached here, all layers have the same stats
  assert False, 'Activation statistics are the same in all layers'


def configure_jax():
  config.update('jax_numpy_rank_promotion', 'raise')
