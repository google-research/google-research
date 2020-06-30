# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Standalone Flax models."""

import functools
import math
import operator as op
import os
import pprint

from flax import jax_utils
from flax import nn
from flax.training import checkpoints
from flax.training import common_utils
import gin
from gin import config
import jax
from jax import random
import jax.nn
import jax.numpy as jnp
import tensorflow.compat.v1 as tf
from protein_lm import data
from protein_lm import modules
from protein_lm import sampling
from protein_lm import utils


def parse_config(ckpt_dir):
  """Parses a FlaxLM config as a dict from checkpoint dir."""
  cfg = dict()
  with tf.gfile.GFile(os.path.join(ckpt_dir, 'config.gin')) as f:
    for line in f:
      if 'FlaxLM' in line and not line.startswith('#'):
        key, value = line.split(' = ')
        _, kwarg = key.split('.')
        value = config.parse_value(value)
        cfg[kwarg] = value
  return cfg


@functools.lru_cache()
def load_model(ckpt_dir, domain=None):
  """Loads a model from directory."""
  if domain is None:
    domain = data.protein_domain
  cfg = parse_config(ckpt_dir)
  print('Loading model with config:')
  pprint.pprint(cfg)
  model = FlaxLM(domain=domain, **cfg)
  model.load_checkpoint(ckpt_dir)
  return model


def train_step(optimizer,
               inputs,
               learning_rate_fn,
               example_weights=None,
               dropout_rng=None,
               grad_clip=None,
               bos_token=0):
  """Performs a single training step. Masks out BOS/PAD positions.

  Args:
    optimizer: Flax optimizer.
    inputs: [batch x length] inputs.
    learning_rate_fn: function from step idx --> learning rate.
    example_weights: Optional [batch] weights for the loss on each example.
      See utils.compute_weighted_cross_entropy for details.
    dropout_rng: RNG for dropout.
    grad_clip: If not None, clip gradients to [-x, +x].
    bos_token: Beginning of sentence token used to generate weight mask.

  Returns:
    new_optimizer, metrics, new_dropout_rng
  """

  # BOS token is equal to PAD when seen on output (loss) side, so this masks
  # out both BOS and PAD positions.
  token_weights = jnp.where(inputs != bos_token, 1, 0)

  # We handle PRNG splitting inside the top pmap, rather
  # than handling it outside in the training loop - doing the
  # latter can add some stalls to the devices.
  dropout_rng, new_dropout_rng = random.split(dropout_rng)

  def loss_fn(model):
    """Loss function used for training."""
    with nn.stochastic(dropout_rng):
      logits = model(inputs, train=True, cache=None)
    loss, weight_sum = utils.compute_weighted_cross_entropy(
        logits,
        inputs,
        token_weights=token_weights,
        example_weights=example_weights)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = optimizer.state.step
  lr = learning_rate_fn(step)

  # Get gradient
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')

  # Compute metrics from forward pass
  metrics = utils.compute_metrics(logits, inputs, token_weights)
  metrics['learning_rate'] = lr

  metrics['l2_param_sum'] = utils.l2_regularization(optimizer.target.params)

  # Gradient norms
  grad_l2_tree = utils.l2_norm(grad)
  grad_l2_sum = jax.tree_util.tree_reduce(op.add, grad_l2_tree)
  grad_l2_max = jax.tree_util.tree_reduce(jnp.maximum, grad_l2_tree)
  metrics['grad_l2_sum'] = grad_l2_sum
  metrics['grad_l2_max'] = grad_l2_max

  # TODO(ddohan): Clip by global grad norm.
  if grad_clip is not None:
    # Clip gradients after pmean aggregation
    clip = lambda g: jnp.clip(g, -grad_clip, grad_clip)  # pylint: disable=invalid-unary-operand-type
    grad = jax.tree_util.tree_map(clip, grad)

    # Metrics for clipped grads.
    clipped_grad_l2_tree = utils.l2_norm(grad)
    clipped_grad_l2_sum = jax.tree_util.tree_reduce(op.add,
                                                    clipped_grad_l2_tree)
    clipped_grad_l2_max = jax.tree_util.tree_reduce(jnp.maximum,
                                                    clipped_grad_l2_tree)
    metrics['gradclip_l2_sum'] = clipped_grad_l2_sum
    metrics['gradclip_l2_max'] = clipped_grad_l2_max

  # Apply gradients and return new optimizer
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
  return new_optimizer, metrics, new_dropout_rng


def eval_step(model, inputs, bos_token):
  weights = jnp.where(inputs != bos_token, 1, 0)
  outputs = inputs
  inputs = modules.shift_right(
      inputs, bos_token=bos_token)  # Do before input at test time.
  logits = model(inputs, train=False, cache=None)
  return utils.compute_metrics(logits, outputs, weights)


def predict_step(model, inputs, bos_token, output_head='logits'):
  inputs = modules.shift_right(
      inputs, bos_token=bos_token)  # Do before input at test time.
  logits = model(inputs, train=False, cache=None, output_head=output_head)
  return logits


def _tokens_to_logits(last_token, cache, model):
  """Computes the next token logits.

  Args:
    last_token: An array of shape (batch_size, 1) containing last token ids.
    cache: A flax.nn.attention.Cache object.
    model: A Jax decoder model to be used for computing the next token logits.

  Returns:
    logits: An array of shape (batch_size, vocab_size) with the logits.
    new_cache: A flax.nn.attention.Cache object with the updated cache.
  """
  # The returned logits have shape (batch_size, 1, vocab_size).
  with cache.mutate() as new_cache:
    logits = model(last_token, train=False, cache=new_cache)

  # Remove the singleton dimension to return shape (batch_size, vocab_size).
  logits = logits.squeeze(axis=1)
  return logits, new_cache


def sample_step(prompt,
                model,
                cache,
                rng,
                bos_token,
                eos_token,
                max_decode_len,
                tokens_to_logits=_tokens_to_logits,
                **kwargs):
  """Samples autoregressively from the model.

  Args:
    prompt: An array of shape (batch_size, prompt_length) containing the input
      prompt (the model consumes these tokens and starts generation after). For
      generic sampling, the prompt must be a single BOS token.
    model: A Jax decoder model to be used for computing the next token logits.
    cache: A flax.nn.attention.Cache object.
    rng: A jax.random.PRNGKey object.
    bos_token: An int indicating the BOS token id.
    eos_token: An int indicating the EOS token id. If None, we decode until
      reaching the maximum sequence length.
    max_decode_len: An int indicating the maximum sequence length.
    tokens_to_logits: A callable that computes the next token logits given the
      current cache and previous token.
    **kwargs: Named arguments passed to sampling.temperature_sample.

  Returns:
    An array of shape (batch_size, max_decode_len) containing sampled sequences.
      If variable-length, the sequences are right-padded with the EOS token.
  """
  tokens_to_logits = functools.partial(tokens_to_logits, model=model)
  return sampling.temperature_sample(
      prompt,
      init_cache=cache,
      tokens_to_logits=tokens_to_logits,
      max_decode_len=max_decode_len,
      rng=rng,
      bos_token=bos_token,
      eos_token=eos_token,
      **kwargs,
  )


def compute_logprob(inputs, model, mask_token=None):
  """Returns an array of log probabilities for the input sequences."""

  assert inputs.ndim == 2

  targets = inputs
  weights = jnp.where(targets != model.bos_token, 1, 0)
  if mask_token is not None:
    weights *= jnp.where(targets != mask_token, 1, 0)
  logits = model.score(inputs)
  assert logits.ndim == 3

  onehot_targets = common_utils.onehot(targets, logits.shape[-1])
  log_lik = jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
  log_lik *= weights
  log_prob = jnp.sum(log_lik, axis=-1)

  return log_prob


@gin.configurable
class FlaxLM(object):
  """Transformer language model built on Flax.

  Masks out domain.bos in loss and predict.

  TODO(b/153166233): Factor out to FlaxModel and FlaxLM/FlaxBERT.
  """

  def __init__(self,
               domain,
               batch_size=16,
               learning_rate=0.001,
               weight_decay=0.1,
               max_target_length=None,
               random_seed=0,
               emb_dim=32,
               num_heads=2,
               num_layers=4,
               qkv_dim=128,
               mlp_dim=512,
               dropout_rate=0.1,
               attention_dropout_rate=0.1,
               attention_fn=None,
               positional_encoding_module=modules.AddLearnedPositionalEncodings,
               grad_clip=None,
               **sampling_kwargs):
    """Creates an instance of this class.

    Args:
      domain: Sequin Domain for inputs and outputs.
      batch_size: batch size to default to.
      learning_rate: traininglearning rate.
      weight_decay: l2 weight decay strength.
      max_target_length: Maximum training length of inputs.
      random_seed: initial rng seed.
      emb_dim: dimension of embedding
      num_heads: number of heads
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      attention_fn: If given, called with qkv_dim to construct callable
        alternative to nn.dot_product_attention. See `make_fast_attention`.
      positional_encoding_module: A module used for adding positional encodings.
      grad_clip: If not None, clip gradients to [-x, +x].
      **sampling_kwargs: Named arguments passed to the sampling function, e.g.
        temperature=1., top_k=5.
    """
    self._length = domain.length
    self._batch_size = batch_size
    self._bos_token = domain.vocab.bos
    self._eos_token = domain.vocab.eos
    vocab_size = domain.vocab_size
    if self._bos_token is None:  # Add bos token.
      self._bos_token = len(domain.vocab.tokens)
      vocab_size += 1

    if max_target_length is None:
      max_target_length = domain.length + 1
    input_shape = (batch_size, max_target_length)
    learning_rate_fn = lambda timestep: learning_rate

    rng = random.PRNGKey(random_seed)
    rng, init_rng = random.split(rng)
    rng, self._sample_rng = random.split(rng)

    if attention_fn is None:
      attention_fn = nn.dot_product_attention
    else:
      attention_fn = attention_fn(qkv_dim=qkv_dim // num_heads)

    model_def = modules.TransformerLM.partial(
        vocab_size=vocab_size,
        max_len=max_target_length,
        bos_token=self._bos_token,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        attention_fn=attention_fn,
        positional_encoding_module=positional_encoding_module,
    )

    init_model, self._cache_def = utils.create_model_and_cache(
        init_rng, input_shape, model_def)
    self._optimizer = utils.create_adam_optimizer(
        init_model, learning_rate, weight_decay=weight_decay)
    del init_model  # Delete initial model.
    self._p_train_step = jax.pmap(
        functools.partial(
            train_step,
            learning_rate_fn=learning_rate_fn,
            grad_clip=grad_clip,
            bos_token=self._bos_token),
        axis_name='batch')
    self._p_eval_step = jax.pmap(
        functools.partial(eval_step, bos_token=self._bos_token),
        axis_name='batch')
    self._p_sample_step = jax.pmap(
        functools.partial(
            sample_step,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            max_decode_len=self._length + 1,
            **sampling_kwargs,
        ),
        axis_name='batch')
    self._p_predict_step = jax.pmap(
        functools.partial(predict_step, bos_token=self._bos_token),
        axis_name='batch')

    # We init the first set of dropout PRNG keys, but update it afterwards
    # inside the main pmap'd training update for performance.
    self._dropout_rngs = random.split(rng, jax.local_device_count())

    self._metrics_all = []
    self._train_step = 0

  @property
  def train_step(self):
    """Returns the current train step."""
    return self._train_step

  @property
  def bos_token(self):
    """Returns the BOS token id."""
    return self._bos_token

  @property
  def eos_token(self):
    """Returns the EOS token id."""
    return self._eos_token

  @property
  def length(self):
    """Returns the maximum sequence length."""
    return self._length

  @property
  def cache_def(self):
    """Returns the associated autoregressive cache_def."""
    return self._cache_def

  @property
  def optimizer(self):
    """Returns Flax optimizer containing optimizer and model parameters."""
    return self._optimizer

  def set_weights(self, optimizer):
    """Sets weights from unreplicated optimizer."""
    self._train_step = int(optimizer.state.step)
    self._optimizer = jax_utils.replicate(optimizer)

  def get_weights(self):
    """Returns unreplicated optimizer."""
    return jax_utils.unreplicate(self._optimizer)

  def save_checkpoint(self, ckpt_dir):
    """Saves unreplicated optimizer to ckpt_dir."""
    optimizer = jax_utils.unreplicate(self._optimizer)
    checkpoints.save_checkpoint(
        ckpt_dir,
        target=optimizer,
        step=self._train_step,
    )

  def load_checkpoint(self, ckpt_dir):
    """Loads optimizer from ckpt_dir."""
    target = jax_utils.unreplicate(self._optimizer)
    optimizer = checkpoints.restore_checkpoint(ckpt_dir, target=target)
    if optimizer is target:
      raise ValueError('Unable to load checkpoint from %s' % ckpt_dir)
    self.set_weights(optimizer)

  def fit(self,
          xs,
          ys=None,
          weights=None,
          epochs=1,
          batch_size=None,
          shuffle=True,
          max_steps=None,
          verbose=False):
    """Fits to sequences given as [N x length] token array."""
    # TODO(ddohan): Use other kwargs.
    del shuffle
    del weights
    del verbose
    del ys
    if batch_size is None:
      batch_size = self._batch_size
    if hasattr(xs, 'as_numpy_iterator'):
      # TF Dataset
      ds = xs.repeat(epochs)
      num_train_steps = max_steps
    elif hasattr(xs, 'element_spec'):
      # Dataset iterator.
      if epochs != 1:
        raise ValueError('Epochs must == 1 when using iterator input.')
      ds = xs
      num_train_steps = max_steps
    else:
      # Raw sequences which we turn into a dataset.
      ds = data.dataset_from_tensors(xs)
      ds = ds.shuffle(buffer_size=1024).repeat().batch(batch_size)
      num_train_steps = math.ceil((len(xs) * epochs) / float(batch_size))
      if max_steps:
        num_train_steps = min(num_train_steps, max_steps)

    if not num_train_steps:
      raise ValueError('Must set max_steps to nonzero value.')

    for _, batch in zip(range(num_train_steps), ds):
      batch = batch._numpy()  # pylint: disable=protected-access
      metrics = self.fit_batch(batch)

  def fit_batch(self, batch):
    self._train_step += 1
    batch = common_utils.shard(batch)
    self._optimizer, metrics, self._dropout_rngs = self._p_train_step(
        self._optimizer, batch, dropout_rng=self._dropout_rngs)
    self._metrics_all.append(metrics)
    return metrics

  def evaluate_batch(self, batch):
    """Computes metrics for given [batch x length] sequences."""
    batch = common_utils.shard(batch)
    metrics = self._p_eval_step(self._optimizer.target, batch)
    return metrics

  def score(self, xs):
    """Predicts logits for given [batch x length] sequences."""
    batch = common_utils.shard(xs)
    logits = self._p_predict_step(self._optimizer.target, batch)
    # Undo pmap batching
    logits = jnp.reshape(logits, [-1, self._length, logits.shape[-1]])
    return logits

  def sample_with_prompt(self, prompt, rng=None):
    """Draws prompt-guided samples from the model.

    # TODO(gandreea): We could handle variable length prompts by assuming the
    #   input prompt to be a list and padding with the out_of_prompt_token.

    Args:
      prompt: Iterable over equal-length sequences to use as input for sampling.
        The prompt is assumed to start with the BOS token.
      rng: A jax.random.PRNGKey object.

    Returns:
      An array of shape (len(prompt), self._length) containing sequences. If
        variable-length, the sequences are right-padded with the EOS token.
    """
    if rng is None:
      self._sample_rng, rng = jax.random.split(self._sample_rng)
    length = self._length + 1
    prompt = common_utils.shard(prompt)
    cache = jax_utils.replicate(
        self._cache_def.initialize_cache((prompt.shape[1], length)))
    samples = self._p_sample_step(
        prompt=prompt,
        model=self._optimizer.target,
        cache=cache,
        rng=jax.random.split(rng, num=len(jax.local_devices())),
    )

    # Remove the BOS token from the sampled sequences.
    samples = samples[:, :, 1:]

    # Undo pmap batching
    samples = jnp.reshape(samples, [-1, self._length])
    return samples

  def sample(self, batch_size, rng=None):
    """Draws samples from the model.

    Args:
      batch_size: An int indicating the number of samples to return.
      rng: A jax.random.PRNGKey object.

    Returns:
      An array of shape (batch_size, self._length) containing sequences. If
        variable-length, the sequences are right-padded with the EOS token.
    """
    # To draw generic samples, we initialize the prompt with the BOS token.
    prompt = jnp.ones((batch_size, 1)).astype(jnp.int32) * self._bos_token
    return self.sample_with_prompt(prompt, rng=rng)
