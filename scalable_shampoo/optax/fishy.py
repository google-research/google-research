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

"""Example for fishy based sampling."""
# A simple example on how to handle sampled fisher for final classification
# layer from: https://openreview.net/forum?id=cScb-RrBQC
#
import math

import jax
import jax.numpy as jnp


@jax.custom_vjp
def sampled_with_softmax(unused_rng, pre_act):
  """Custom autodiff wrapper for softmax."""
  return pre_act


def sampled_with_softmax_fwd(rng, pre_act):
  """Custom forward autodiff wrapper for softmax."""
  post_act = jax.nn.softmax(pre_act)
  return pre_act, (post_act, pre_act, rng)


def sampled_with_softmax_bwd(res, unused_g):
  """Custom backward autodiff wrapper for softmax."""
  post_act, pre_act, rng = res
  post_act_sample = jax.random.categorical(rng, logits=pre_act)
  out = post_act - jax.nn.one_hot(
      post_act_sample, num_classes=post_act.shape[-1], dtype=jnp.float32)
  out /= math.prod(post_act.shape[:-1])
  return (None, out)

sampled_with_softmax.defvjp(sampled_with_softmax_fwd, sampled_with_softmax_bwd)
