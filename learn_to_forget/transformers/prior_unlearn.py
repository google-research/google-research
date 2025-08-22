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

"""Implements prior work solutions to unlearning.

Included:
- Knowledge unlearning (https://arxiv.org/abs/2210.01504) based on updates of
the objective of unlearning (eq 1 in the paper) upon correspondence with the
authors
"""

from typing import Any, Mapping, Optional, Tuple

import gin
import jax
import jax.numpy as jnp
from t5x import losses
from t5x import metrics as metrics_lib
from t5x import models


MetricsMap = metrics_lib.MetricsMap
PyTree = Any


@gin.configurable(module='prior_unlearn')
class EncoderDecoderNegLikelihood(models.EncoderDecoderModel):
  """Implements knowledge unlearning with negative likelihood.

  This class is the same as the EncodeDecoderModel used for training, except the
  different loss function.
  """

  def loss_fn(
      self,
      params,
      batch,
      dropout_rng,
  ):
    """Computes negative loglikehood for unlearning.

    The loss does not use z_loss or loss_normalizing factor.

    Args:
      params: The model parameters.
      batch: A batch of inputs.
      dropout_rng: The rng for droput.

    Returns:
      loss: The loss computed over the batch of inputs and parameters.
      aux:
        weight_sum: sum of the per-token weights applied to the loss.
        metrics: a mapping of metrics computed for this batch.
    """

    logits = self._compute_logits(params, batch, dropout_rng)

    weights = batch.get('decoder_loss_weights', None)
    # Do not add z-loss term and do not do label smoothing
    loss, _, _ = losses.compute_weighted_cross_entropy(
        logits,
        targets=batch['decoder_target_tokens'],
        weights=weights)

    # The unlearning is done by gradient ascent, maximizing the negative
    # likelihood of tokens p(x_t | x_{<t})
    loss *= -1

    metrics = self._compute_metrics(
        logits=logits,
        targets=batch['decoder_target_tokens'],
        mask=weights,
        loss=loss,
        z_loss=0.0)

    return loss, metrics
