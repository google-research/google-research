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

"""Loss functions for CIFAR10 training."""
from flax.training import common_utils
import jax
import jax.numpy as jnp


def pmean(x, config, axis_name='batch'):
  if config.debug_run:
    return x
  else:
    return jax.lax.pmean(x, axis_name)


def cross_entropy_loss(logits, labels):
  log_softmax_logits = jax.nn.log_softmax(logits)
  num_classes = log_softmax_logits.shape[-1]
  one_hot_labels = common_utils.onehot(labels, num_classes)
  return -jnp.sum(one_hot_labels * log_softmax_logits) / labels.size


def cross_entropy_loss_probs(probs, labels):
  log_probs = jnp.log(jax.lax.clamp(1e-8, probs, 1 - 1e-8))
  num_classes = probs.shape[-1]
  one_hot_labels = common_utils.onehot(labels, num_classes)
  return -jnp.sum(one_hot_labels * log_probs) / labels.size


def compute_metrics(config, logits, labels):
  """Compute metrics on accelerator."""
  loss = cross_entropy_loss(logits, labels)
  error_rate = jnp.mean(jnp.argmax(logits, -1) != labels)
  metrics = {
      'loss': loss,
      'error_rate': error_rate,
      'accuracy': 1 - error_rate,
  }
  metrics = pmean(metrics, config)
  return metrics


def compute_metrics_probs(probs, labels):
  """Compute metrics on CPU using probabilities for ensemble metrics."""
  loss = cross_entropy_loss_probs(probs, labels)
  error_rate = jnp.mean(jnp.argmax(probs, -1) != labels)
  metrics = {
      'loss': loss,
      'error_rate': error_rate,
      'accuracy': 1 - error_rate,
  }
  return metrics
