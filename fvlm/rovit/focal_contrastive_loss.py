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

# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Focal contrastive loss function.

Adapted from tf.nn.sigmoid_cross_entropy_with_logits and Cloud TPU detection:
https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
https://github.com/tensorflow/tpu/blob/master/models/official/detection/modeling/losses.py
"""

from typing import Optional

from flax.training import common_utils
import jax
import jax.numpy as jnp


def compute_focal_contrastive_loss(
    logits,
    targets,
    alpha = 0.5,
    gamma = 1.0,
    loss_normalizing_factor = None
):
  """Compute focal loss for logits and labels.

  Implementation follows:
  1. tf.nn.sigmoid_cross_entropy_with_logits
  2. third_party/cloud_tpu/models/detection/modeling/losses.py

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   alpha: A float32 scalar multiplying alpha to the loss from positive examples
     and (1-alpha) to the loss from negative examples.
   gamma: A float32 scalar modulating loss from hard and easy examples.
   loss_normalizing_factor: Constant to divide loss by. If not specified, loss
     will not be normalized. Intended for backward compatibility with T5-MTF
     training. Should not normally be used.

  Returns:
    A scalar loss.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  assert alpha <= 1.0 and alpha >= 0
  vocab_size = logits.shape[-1]
  soft_targets = common_utils.onehot(targets, vocab_size)
  cross_entropy = jnp.maximum(logits, 0) - logits * soft_targets + jnp.log(
      1 + jnp.exp(-jnp.absolute(logits)))

  neg_logits = -1.0 * logits
  modulator = jnp.exp(gamma * soft_targets * neg_logits -
                      gamma * jax.nn.softplus(neg_logits))
  loss = modulator * cross_entropy
  weighted_loss = 2 * jnp.where(
      soft_targets == 1.0, alpha * loss, (1.0 - alpha) * loss)
  if loss_normalizing_factor is not None:
    weighted_loss /= loss_normalizing_factor + 1e-20

  return jnp.sum(weighted_loss)
