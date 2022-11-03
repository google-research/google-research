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

"""Utilities for side outputs in Flax models."""

import contextlib
from typing import Any, Optional

import flax
import gin
import jax
import jax.numpy as jnp

from gfsa import jax_util

_side_output_stack = flax.deprecated.nn.utils.CallStack()


class SideOutput(flax.deprecated.nn.Module):
  """Flax module to tag a side output, for later extraction."""

  def apply(self, value):
    if _side_output_stack:
      _side_output_stack[-1].store(value)


@contextlib.contextmanager
def collect_side_outputs():
  """Context manager to collect side outputs."""
  result = {}
  with flax.deprecated.nn.Collection().mutate() as side_output_collection:
    with _side_output_stack.frame(side_output_collection):
      yield result

  result.update(side_output_collection.as_dict())


def _bernoulli_logit_entropy(logits):
  """Computes elementwise entropy of an array of logits."""
  # -(p log p + (1-p) log (1-p))
  # = -p (log p - log(1-p)) - log (1-p)
  # = -p log (p / (1-p)) - log (1-p)
  return -jax.nn.sigmoid(logits) * logits - jax.nn.log_sigmoid(-logits)


def _categorical_logit_entropy(logits):
  """Computes group-wise entropy of an array of logits."""
  return -jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1)


@gin.configurable
def encourage_discrete_logits(logits,
                              distribution_type,
                              name = None,
                              regularize = True,
                              perturb_scale = None):
  """Encourage binary logits to be discrete.

  Args:
    logits: Pytree of binary logits.
    distribution_type: Either "categorical" or "binary".
    name: Name for the logits.
    regularize: Whether to register a regularization penalty for these logits.
      This penalty can be extracted with `collect_side_output_penalties`, and
      represents the average entropy of the logits.
    perturb_scale: Scale of logistic noise to perturb with, or None to not
      perturb. If not None, must be in a flax stochastic scope.

  Returns:
    Possibly perturbed version of logits.
  """
  assert distribution_type in {"categorical", "binary"}
  logit_leaves, treedef = jax.tree_util.tree_flatten(logits)
  if regularize:
    if distribution_type == "categorical":
      entropies = [_categorical_logit_entropy(x) for x in logit_leaves]
    elif distribution_type == "binary":
      entropies = [_bernoulli_logit_entropy(x) for x in logit_leaves]
    divisor = sum(x.size for x in entropies)
    mean_entropy = sum(jnp.sum(entropy) for entropy in entropies) / divisor
    # Tag it with Flax.
    SideOutput(mean_entropy, name=(name or "logit") + "_entropy")

  if perturb_scale:
    rng = flax.deprecated.nn.make_rng()
    subkeys = jax.random.split(rng, len(logit_leaves))

    if distribution_type == "categorical":
      logit_leaves = [
          leaf + perturb_scale * jax.random.gumbel(key, leaf.shape)
          for leaf, key in zip(logit_leaves, subkeys)
      ]

    elif distribution_type == "binary":
      logit_leaves = [
          leaf + perturb_scale * jax.random.logistic(key, leaf.shape)
          for leaf, key in zip(logit_leaves, subkeys)
      ]

  return jax.tree_util.tree_unflatten(treedef, logit_leaves)
