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

"""Helper functions for training baseline seq-to-seq model."""

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error

import collections

from flax import linen as nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np

from latent_programmer import decode
from latent_programmer import models


def create_learning_rate_scheduler(
    base_learning_rate=0.5,
    factors='constant * linear_warmup * rsqrt_normalized_decay',
    warmup_steps=16000,
    decay_factor=0.5,
    steps_per_decay=50000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    base_learning_rate: float, the starting constant for the lr schedule.
    factors: a string with factors separated by '*' that defines the schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.
    steps_per_cycle: Steps per cycle when using cosine decay.

  Returns:
    A function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(1.0, step - warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def compute_weighted_cross_entropy(logits, targets, weights=None):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: `[batch, length, num_classes]` float array.
   targets: categorical targets `[batch, length]` int array.
   weights: None or array of shape [batch, length, 1]

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))

  onehot_targets = common_utils.onehot(targets, logits.shape[-1])
  loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
  normalizing_factor = jnp.prod(jnp.asarray(targets.shape))
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: `[batch, length, num_classes]` float array.
   targets: categorical targets `[batch, length]` int array.
   weights: None or array of shape [batch, length, 1]

  Returns:
    Tuple of scalar accuracy and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  acc = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = jnp.prod(jnp.asarray(targets.shape))
  if weights is not None:
    acc = acc * weights
    normalizing_factor = weights.sum()

  return acc.sum(), normalizing_factor


def compute_metrics(logits, targets, weights):
  """Compute summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)
  acc, _ = compute_weighted_accuracy(logits, targets, weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
  }
  metrics = jax.lax.psum(metrics, 'batch')
  return metrics


# Train / eval / decode step functions.
# -----------------------------------------------------------------------------


def train_step(optimizer,
               inputs,
               outputs,
               programs,
               learning_rate_fn,
               config,
               train_rng=None):
  """Train on batch of program tasks."""
  # We handle PRNG splitting inside the top pmap, rather
  # than handling it outside in the training loop - doing the
  # latter can add some stalls to the devices.
  train_rng, new_train_rng = jax.random.split(train_rng)

  weights = jnp.where(programs > 0, 1, 0).astype(jnp.float32)

  def loss_fn(params):
    """Loss function used for training."""
    logits = models.ProgramTransformer(config).apply(
        {'params': params},
        inputs,
        outputs,
        programs,
        rngs={'dropout': train_rng})
    loss, weight_sum = compute_weighted_cross_entropy(logits, programs, weights)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  # Get metrics.
  metrics = compute_metrics(logits, programs, weights)
  metrics['learning_rate'] = lr
  return new_optimizer, metrics, new_train_rng


def eval_step(params, inputs, outputs, programs, config):
  weights = jnp.where(programs > 0, 1, 0).astype(jnp.float32)
  logits = models.ProgramTransformer(config).apply(
      {'params': params}, inputs, outputs, programs)

  return compute_metrics(logits, programs, weights)


def initialize_cache(inputs, outputs, programs, max_decode_len, config):
  """Initialize a cache for a given input shape and max decode length."""
  target_shape = (programs.shape[0], max_decode_len)
  initial_variables = models.ProgramTransformer(config).init(
      jax.random.PRNGKey(0),
      jnp.ones(inputs.shape, config.dtype),
      jnp.ones(outputs.shape, config.dtype),
      jnp.ones(target_shape, config.dtype))
  return initial_variables['cache']


def predict_step(params,
                 inputs,
                 outputs,
                 cache,
                 eos_token,
                 max_decode_len,
                 beam_size,
                 config):
  """Predict translation with fast decoding beam search on a batch."""
  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item's data is expanded in-place
  # rather than tiled.
  flat_encoded = decode.flat_batch_beam_expand(
      models.ProgramTransformer(config).apply(
          {'params': params},
          inputs,
          outputs,
          method=models.ProgramTransformer.encode),
      beam_size)

  encoded_padding_mask = jnp.where(outputs > 0, 1, 0).astype(jnp.float32)
  flat_encoded_padding_mask = decode.flat_batch_beam_expand(
      encoded_padding_mask, beam_size)

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    flat_logits, new_vars = models.ProgramTransformer(config).apply(
        {'params': params, 'cache': flat_cache},
        flat_ids,
        flat_encoded,
        flat_encoded_padding_mask,
        mutable=['cache'],
        method=models.ProgramTransformer.decode)
    new_flat_cache = new_vars['cache']
    # Remove singleton sequence-length dimension:
    # [batch * beam, 1, vocab] --> [batch * beam, vocab]
    flat_logits = flat_logits.squeeze(axis=1)
    return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  beam_seqs, _ = decode.beam_search(
      inputs,
      cache,
      tokens_ids_to_logits,
      beam_size=beam_size,
      alpha=0.6,
      bos_token=config.bos_token,
      eos_token=eos_token,
      max_decode_len=max_decode_len)

  # Beam search returns [n_batch, n_beam, n_length] with beam dimension
  # sorted in increasing order of log-probability.
  return beam_seqs


# Util functions for prediction
# -----------------------------------------------------------------------------


def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  tile_dims = [1] * len(x.shape)
  tile_dims[0] = batch_pad
  return np.concatenate([x, np.tile(x[-1], tile_dims)], axis=0)


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return x.reshape((n_device * n_batch,) + tuple(remaining_dims))


def per_host_sum_pmap(in_tree):
  """Execute psum on in_tree's leaves over one device per host."""
  host2devices = collections.defaultdict(list)
  for d in jax.devices():
    host2devices[d.host_id].append(d)
  devices = [host2devices[k][0] for k in host2devices]
  host_psum = jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i', devices=devices)
  def pre_pmap(xs):
    return jax.tree_map(lambda x: jnp.broadcast_to(x, (1,) + x.shape), xs)
  def post_pmap(xs):
    return jax.tree_map(lambda x: x[0], xs)
  return post_pmap(host_psum(pre_pmap(in_tree)))


def eval_predicted(predicted, inputs, outputs, parse_beam_fn):
  """Evaluate predicted program beams."""
  best_p, best_score = None, -1

  # predicted shape [beam_size, length]
  for beam in predicted:
    try:
      p = parse_beam_fn(beam)
      p_outs = [p(inp) for inp in inputs]
      score = np.sum([p_out == out for p_out, out in zip(p_outs, outputs)])
      if score > best_score:
        best_p, best_score = p, score
    except:  # pylint: disable=bare-except
      pass
    if best_score >= len(inputs):  # Found solution.
      break
  return best_p, best_score
