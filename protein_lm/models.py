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

# Lint as: python3
"""Standalone Flax models."""

import abc
import enum
import functools
import math
import operator as op
import os
import pprint
import time

from absl import logging
from flax import jax_utils
from flax.deprecated import nn
from flax.training import checkpoints
from flax.training import common_utils
import gin
from gin import config
import jax
from jax import random as jrandom
import jax.experimental.optimizers
import jax.nn
import jax.numpy as jnp
import numpy as onp
import tensorflow.compat.v1 as tf
import tree

from protein_lm import data
from protein_lm import evaluation
from protein_lm import modules
from protein_lm import sampling
from protein_lm import utils


class Mode(enum.Enum):
  train = 'train'
  eval = 'eval'
  predict = 'predict'
  sample = 'sample'


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


def save_model_kwargs(ckpt_dir, model):
  """Saves a dict FlaxLM config into the checkpoint dir."""
  model_kwargs = model.model_kwargs
  model_name = type(model).__name__
  with tf.gfile.GFile(os.path.join(ckpt_dir, 'config.gin'), 'w') as f:
    for key, value in model_kwargs.items():
      f.write('%s.%s = %s\n' % (model_name, key, str(value)))


@functools.lru_cache()
def load_model(ckpt_dir, model_cls, domain=None):
  """Loads a model from directory."""
  if domain is None:
    domain = data.protein_domain
  cfg = parse_config(ckpt_dir)
  print('Loading model with config:')
  pprint.pprint(cfg)
  model = model_cls(domain=domain, **cfg)
  model.load_checkpoint(ckpt_dir)
  return model


def train_step(optimizer,
               inputs,
               learning_rate_fn,
               dropout_rng,
               preprocess_fn,
               example_weights=None,
               grad_clip=None,
               epsilon=1e-9):
  """Performs a single training step. Masks out BOS/PAD positions.

  Args:
    optimizer: Flax optimizer.
    inputs: Inputs to model.preprocess which returns (inputs, targets, weights).
    learning_rate_fn: function from step idx --> learning rate.
    dropout_rng: RNG for dropout.
    preprocess_fn: function mapping
      (inputs, rng, mode) -> (inputs, targets, weights).
    example_weights: Optional [batch] weights for the loss on each example.
      See utils.compute_weighted_cross_entropy for details.
    grad_clip: If not None, clip gradients to [-x, +x].
    epsilon: Epsilon for denominator of loss averaging.

  Returns:
    new_optimizer, metrics, new_dropout_rng
  """

  # We handle PRNG splitting inside the top pmap, rather
  # than handling it outside in the training loop - doing the
  # latter can add some stalls to the devices.
  dropout_rng, new_dropout_rng = jrandom.split(dropout_rng)
  dropout_rng, preprocess_rng = jrandom.split(dropout_rng)

  inputs, targets, weights = preprocess_fn(
      inputs, rng=preprocess_rng, mode=Mode.train)

  if isinstance(targets, dict):
    classification_targets = targets['classification']
    classification_weights = weights['classification']

    regression_targets = targets['regression']
    regression_weights = weights['regression']
  else:
    # Default to classification loss.
    classification_targets = targets
    classification_weights = weights
    regression_targets = None

  if classification_targets is None and regression_targets is None:
    raise ValueError('No targets specified for train step.')

  if classification_weights is None and regression_weights is None:
    raise ValueError('No weights specified for train step.')

  def loss_fn(model):
    """Loss function used for training."""
    # Stateful collection for tracking internal state like activations.
    with nn.stateful() as batch_stats:
      with nn.stochastic(dropout_rng):
        outputs = model(inputs, train=True, cache=None)

      if isinstance(outputs, dict):
        logits = outputs.get('logits', None)
        regression_predictions = outputs.get('regression', None)
      else:
        logits = outputs
        regression_predictions = None

    mean_loss = 0.0

    # Classification loss
    if classification_targets is not None:
      classification_loss, classification_weight_sum = utils.compute_weighted_cross_entropy(
          logits,
          classification_targets,
          token_weights=classification_weights,
          example_weights=example_weights)
      classification_weight_sum = jnp.maximum(classification_weight_sum,
                                              epsilon)
      # Handle case where nothing is masked out in BERT
      # (Only occurs with very short sequences).
      mean_classification_loss = classification_loss / classification_weight_sum
      mean_loss += mean_classification_loss

    if regression_targets is not None:
      regression_loss, regression_weight_sum = utils.compute_weighted_mse(
          regression_predictions,
          regression_targets,
          weights=regression_weights)
      regression_weight_sum = jnp.maximum(regression_weight_sum, epsilon)
      mean_regression_loss = regression_loss / regression_weight_sum
      outputs['regression_loss'] = mean_regression_loss

      # TODO(ddohan): Allow weighting each loss separately.
      mean_loss += mean_regression_loss

    return mean_loss, (outputs, batch_stats)

  step = optimizer.state.step
  lr = learning_rate_fn(step)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, (outputs, batch_stats)), grad = grad_fn(optimizer.target)

  try:
    grad = jax.lax.pmean(grad, 'batch')
  except NameError:
    pass

  if grad_clip is not None:
    # Clip gradients after pmean aggregation
    unclipped_grad = grad
    grad = jax.experimental.optimizers.clip_grads(grad, grad_clip)

  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  # TODO(ddohan): Avoid computing metrics except when needed.
  if isinstance(outputs, dict):
    logits = outputs.get('logits', None)
  else:
    logits = outputs

  metrics = dict()
  if logits is not None:
    classification_metrics = utils.compute_metrics(logits,
                                                   classification_targets,
                                                   classification_weights)
    metrics.update(classification_metrics)
  if regression_targets is not None:
    # TODO(ddohan): Implement regression metrics.
    logging.info('No regression targets yet')
    # regression = outputs.get('regression', None)
    # regression_metrics = utils.compute_metrics(logits, regression_targets,
    #                                                classification_weights)
  metrics['learning_rate'] = lr

  # Training metrics
  metrics['l2_param_sum'] = utils.l2_regularization(optimizer.target.params)

  # Gradient norms
  grad_l2_tree = utils.l2_norm(grad)
  grad_l2_sum = jax.tree_util.tree_reduce(op.add, grad_l2_tree)
  grad_l2_max = jax.tree_util.tree_reduce(jnp.maximum, grad_l2_tree)
  metrics['l2_grad_sum'] = grad_l2_sum
  metrics['l2_grad_max'] = grad_l2_max

  # Store any tagged metrics
  batch_stats = batch_stats.as_dict()
  if batch_stats:

    def clean_name(k):
      return 'nn/' + k.replace('MultiHeadDotProductAttention_', '').replace(
          '/Transformer1DBlock_', '')

    stats = {clean_name(k): v['tag'] for k, v in batch_stats.items()}
    metrics.update(stats)

  if grad_clip is not None:
    # Unclipped gradient norms (if applicable).
    grad_l2_tree = utils.l2_norm(unclipped_grad)
    grad_l2_sum = jax.tree_util.tree_reduce(op.add, grad_l2_tree)
    grad_l2_max = jax.tree_util.tree_reduce(jnp.maximum, grad_l2_tree)
    metrics['l2_noclip_grad_sum'] = grad_l2_sum
    metrics['l2_noclip_grad_max'] = grad_l2_max

  return new_optimizer, metrics, new_dropout_rng


def eval_step(model, inputs, preprocess_fn):
  inputs, targets, weights = preprocess_fn(inputs, rng=None, mode=Mode.eval)
  logits = model(inputs, train=False, cache=None)
  return utils.compute_metrics(logits, targets, weights)


def predict_step(model, inputs, preprocess_fn, output_head='logits'):
  inputs, _, _ = preprocess_fn(inputs, rng=None, mode=Mode.predict)
  logits = model(inputs, train=False, cache=None, output_head=output_head)
  return logits


def _tokens_to_logits(last_token, cache, model, internal_state=None):
  """Computes the next token logits.

  Args:
    last_token: An array of shape (batch_size, 1) containing last token ids.
    cache: A flax.nn.attention.Cache object.
    model: A Jax decoder model to be used for computing the next token logits.
    internal_state: A dict with internal state received from the previous time
      step. If None, no information is shared across time steps.

  Returns:
    logits: An array of shape (batch_size, vocab_size) with the logits.
    new_cache: A flax.nn.attention.Cache object with the updated cache.
    new_internal_state: A dict with internal state passed to the next time step.
  """
  del internal_state  # Not used.
  # The returned logits have shape (batch_size, 1, vocab_size).
  with cache.mutate() as new_cache:
    logits = model(last_token, train=False, cache=new_cache)

  # Remove the singleton dimension to return shape (batch_size, vocab_size).
  logits = logits.squeeze(axis=1)
  return logits, new_cache, None


def sample_step(prompt,
                model,
                cache,
                rng,
                masked_tokens,
                eos_token,
                pad_token,
                max_decode_len,
                tokens_to_logits=_tokens_to_logits,
                **sampling_kwargs):
  """Samples autoregressively from the model.

  Args:
    prompt: An array of shape (batch_size, prompt_length) containing the input
      prompt (the model consumes these tokens and starts generation after). For
      generic sampling, the prompt must be a single BOS token.
    model: A Jax decoder model to be used for computing the next token logits.
    cache: A flax.nn.attention.Cache object.
    rng: A jax.random.PRNGKey object.
    masked_tokens: A list of ints indicating tokens to mask out during sampling.
    eos_token: An int indicating the EOS token id. If None, we decode until
      reaching the maximum sequence length.
    pad_token: An int token used to pad sequences after the eos token. If none,
      we set pad_token to eos_token.
    max_decode_len: An int indicating the maximum sequence length.
    tokens_to_logits: A callable that computes the next token logits given the
      current cache and previous token.
    **sampling_kwargs: Named arguments passed to sampling.temperature_sample.

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
      eos_token=eos_token,
      pad_token=pad_token,
      masked_tokens=masked_tokens,
      **sampling_kwargs,
  )


def compute_logprob(inputs, model, mask_token=None):
  """Returns an array of log probabilities for the input sequences."""

  assert inputs.ndim == 2

  targets = inputs
  weights = jnp.where(targets != model.pad_token, 1, 0)
  if mask_token is not None:
    weights *= jnp.where(targets != mask_token, 1, 0)
  logits = model.score(inputs)
  assert logits.ndim == 3

  onehot_targets = common_utils.onehot(targets, logits.shape[-1])
  log_lik = jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
  log_lik *= weights
  log_prob = jnp.sum(log_lik, axis=-1)

  return log_prob


def preprocess_causal(batch, bos_token, pad_token, mode):
  """Preprocessing for causal language modeling.

  Right shifts and shards.

  Args:
    batch: [batch x length] tokens.
    bos_token: Int ID to use as beginning of sentence token.
    pad_token: Padding token which should be masked out in loss.
    mode: Mode value.

  Returns:
    Tuple of [batch x length] inputs, targets, per position weights. Targets
      will have random positions masked out with either a MASK token, or a
      randomly chosen token from the vocabulary.
  """
  if mode == Mode.sample:
    inputs = batch
  else:
    inputs = modules.shift_right(batch, bos_token=bos_token)

  targets = batch
  # Mask out PAD in loss.
  if pad_token is None:
    weights = jnp.ones_like(targets)
  else:
    weights = jnp.where(targets != pad_token, 1, 0)
  return inputs, targets, weights


@gin.configurable
class FlaxModel(abc.ABC):
  """Model built on Flax."""

  def __init__(self,
               domain=data.protein_domain,
               model_cls=modules.Transformer,
               random_seed=0,
               batch_size=None,
               grad_clip=None,
               learning_rate=0.001,
               weight_decay=0.1,
               cache=True,
               pmap=True,
               attention_fn=None,
               with_bos=False,
               with_mask=False,
               store_metrics=False,
               sampling_kwargs=None,
               **model_kwargs):
    """Creates a Flax model for sequence prediction.

    Args:
      domain: discrete domain.
      model_cls: Flax.nn.Module to train.
      random_seed: Random seed.
      batch_size: Default batch size.
      grad_clip: Gradient clipping in optimizer.
      learning_rate: learning rate in optimizer, or callable mapping a step to
        current learning rate.
      weight_decay: L2 decay for AdamW.
      cache: Whether to create a cache.
      pmap: Whether to pmap inference (and JIT as a side effect).
      attention_fn: Function to use in place of nn.dot_product_attention.
      with_bos: Whether to ensure vocab contains BOS.
      with_mask: Whether to ensure vocab contains MASK.
      store_metrics: Whether to store train and evaluation metrics.
      sampling_kwargs: Additional config options for sample step.
      **model_kwargs: Additional config options for `model_cls.partial`.
    """
    self._batch_size = batch_size  # Default batch size

    # TODO(b/157255958): Reenable tracking metrics inside class.
    self._store_metrics = store_metrics
    if store_metrics:
      self._metrics_train = []
      self._metrics_test = []
      self._epoch_train = []
      self._epoch_test = []

    self._pmap = pmap
    self._sampling_kwargs = sampling_kwargs
    self._model_kwargs = model_kwargs
    self._opt_hparams = dict(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip)

    # TODO(ddohan): Reimplement __getstate__ and __setstate__ to support pickle,
    # and use these functions to init model.
    self._set_domain(domain=domain, with_bos=with_bos, with_mask=with_mask)
    self._init_model(
        model_cls=model_cls,
        random_seed=random_seed,
        pmap=pmap,
        cache=cache,
        attention_fn=attention_fn,
        sampling_kwargs=sampling_kwargs,
        model_kwargs=model_kwargs,
        **self._opt_hparams)

  def _set_domain(self, domain, with_bos, with_mask):
    """Set vocabulary based on domain."""
    self.domain = domain
    self._length = domain.length
    self._bos_token = domain.vocab.bos
    self._eos_token = domain.vocab.eos
    self._pad_token = domain.vocab.pad
    self._mask_token = domain.vocab.mask

    vocab_size = domain.vocab_size
    if with_bos and self._bos_token is None:  # Add bos token.
      self._bos_token = vocab_size
      vocab_size += 1
    if with_mask and self._mask_token is None:  # Add mask token.
      self._mask_token = vocab_size
      vocab_size += 1
    self._vocab_size = vocab_size

  def _get_masked_tokens(self):
    """Get list of token IDs to mask for a given domain."""
    tokens = []
    for token in [self._bos_token, self._pad_token, self._mask_token]:
      if token is not None:
        tokens.append(token)
    return tokens

  def _init_model(self,
                  model_cls,
                  pmap,
                  learning_rate,
                  weight_decay,
                  grad_clip,
                  attention_fn,
                  random_seed,
                  cache=True,
                  sampling_kwargs=None,
                  model_kwargs=None):
    """Initialize model."""
    model_kwargs = model_kwargs or dict()
    model_def = model_cls.partial(
        vocab_size=self._vocab_size,
        max_len=self.domain.length,
        # Don't attend to PAD tokens
        pad_token=self._pad_token,
        attention_fn=attention_fn,
        **model_kwargs)

    if callable(learning_rate):
      learning_rate_fn = learning_rate
    else:
      learning_rate_fn = lambda step: learning_rate

    train_fn = functools.partial(
        train_step,
        learning_rate_fn=learning_rate_fn,
        grad_clip=grad_clip,
        preprocess_fn=self.preprocess)
    eval_fn = functools.partial(eval_step, preprocess_fn=self.preprocess)
    predict_fn = functools.partial(predict_step, preprocess_fn=self.preprocess)

    sampling_kwargs = sampling_kwargs or dict()
    masked_tokens = self._get_masked_tokens()
    sample_fn = functools.partial(
        sample_step,
        masked_tokens=masked_tokens,
        eos_token=self._eos_token,
        pad_token=self._pad_token,
        max_decode_len=self._length + 1,
        **sampling_kwargs)

    # Default to pmapped versions.
    if pmap:
      train_fn = jax.pmap(train_fn, axis_name='batch')
      eval_fn = jax.pmap(eval_fn, axis_name='batch')
      sample_fn = jax.pmap(sample_fn, axis_name='batch')
      predict_fn = jax.pmap(predict_fn, axis_name='batch')

    self._train_fn = train_fn
    self._predict_fn = predict_fn
    self._sample_fn = sample_fn
    self._eval_fn = eval_fn

    rng = jrandom.PRNGKey(random_seed)
    rng, init_rng = jrandom.split(rng)
    rng, self._sample_rng = jrandom.split(rng)

    # We init the first set of dropout PRNG keys, but update it afterwards
    # inside the main pmap'd training update for performance.
    if self._pmap:
      self._dropout_rngs = jrandom.split(rng, jax.local_device_count())
    else:
      self._dropout_rngs = rng

    # Note: any batch size can be used later. This is arbitrary for init.
    input_shape = (self._batch_size or 2, self.domain.length)
    if cache:
      init_model, self._cache_def = utils.create_model_and_cache(
          init_rng, input_shape, model_def)
    else:
      init_model = utils.create_model(init_rng, input_shape, model_def)
      self._cache_def = None
    self._optimizer = utils.create_adam_optimizer(
        init_model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        pmap=self._pmap)
    del init_model  # Delete initial model.

  def preprocess(self, batch, rng, mode):
    """Unpack batch of data to (inputs, targets, weights).

    batch may be one of:
      - a [batch x length] batch of input data.
        Results in (batch, None, None)
      - a tuple of (inputs, targets)
        Results in (inputs, targets, ones_like(targets))
      - a tuple of (inputs, targets, weights)
        Passed through unchanged.
      - a dict containing 'inputs', 'targets', and
        optionally 'weights'.
        Results in (inputs, targets, weights or ones_like(targets))

    Args:
      batch: Batch of data.
      rng: Ignored. Jax random seed.
      mode: member of Mode enum.

    Returns:
      Tuple of (inputs, targets, weights).
        `targets` and `weights` are None if `targets` is not provided.
    """
    del rng
    if isinstance(batch, tuple):
      if len(batch) == 2:
        inputs, targets = batch
        weights = jnp.ones_like(targets)
      elif len(batch) == 3:
        inputs, targets, weights = batch
      else:
        raise ValueError(
            'Must provide (inputs, targets) or (inputs, targets, weights)')
    elif isinstance(batch, dict):
      inputs = batch['inputs']
      targets = batch['targets']
      weights = batch.get('targets', None)
      if weights is None:
        weights = jnp.ones_like(targets)
    else:
      inputs = batch
      targets = None
      weights = None

    if targets is None and mode not in (Mode.predict, Mode.sample):
      raise ValueError('Must provide targets for train and eval.')

    return inputs, targets, weights

  @property
  def train_step(self):
    """Returns the current train step."""
    step = self.optimizer.state.step
    if self._pmap:
      step = step[0]
    return step

  @property
  def bos_token(self):
    """Returns the BOS token id."""
    return self._bos_token

  @property
  def eos_token(self):
    """Returns the EOS token id."""
    return self._eos_token

  @property
  def pad_token(self):
    """Returns the BOS token id."""
    return self._pad_token

  @property
  def mask_token(self):
    """Returns the MASK token id."""
    return self._mask_token

  @property
  def length(self):
    """Returns the maximum sequence length."""
    return self._length

  @property
  def vocab_size(self):
    """Returns the vocabulary size used for training."""
    return self._vocab_size

  @property
  def optimizer(self):
    """Returns Flax optimizer containing optimizer and model parameters."""
    return self._optimizer

  @property
  def model_kwargs(self):
    """Returns the model kwargs as a dictionary."""
    return self._model_kwargs

  @property
  def pmap(self):
    """Returns whether or not the optimizer was trained with pmap."""
    return self._pmap

  def set_weights(self, optimizer):
    """Sets weights from unreplicated optimizer."""
    if self._pmap:
      optimizer = jax_utils.replicate(optimizer)
    self._optimizer = optimizer

  def get_weights(self):
    """Returns unreplicated optimizer."""
    optimizer = self.optimizer
    if self._pmap:
      optimizer = jax_utils.unreplicate(optimizer)
    return optimizer

  def save_checkpoint(self, ckpt_dir):
    """Saves unreplicated optimizer to ckpt_dir."""
    optimizer = self.get_weights()
    checkpoints.save_checkpoint(
        ckpt_dir,
        target=optimizer,
        step=self.train_step,
    )

  def load_checkpoint(self, ckpt_dir):
    """Loads optimizer from ckpt_dir."""
    target = self.get_weights()
    optimizer = checkpoints.restore_checkpoint(ckpt_dir, target=target)
    if optimizer is target:
      raise ValueError('Unable to load checkpoint from %s' % ckpt_dir)
    self.set_weights(optimizer)

  def fit(self, xs, epochs=1, batch_size=None, max_steps=10**6):
    """Fits to sequences given as [N x length] token array."""
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

    metrics = []
    start = time.time()
    max_steps = max_steps or 10**6
    for _, batch in zip(range(num_train_steps), ds):
      metrics.append(self.fit_batch(batch))
    finish = time.time()
    average = evaluation.combine_metrics(metrics)
    average['runtime'] = finish - start
    average['rate'] = len(metrics) / (finish - start)

    if self._store_metrics:
      average = tree.map_structure(onp.array, average)
      self._epoch_train.append(average)
    return dict(last=evaluation.combine_metrics([metrics[-1]]), average=average)

  def evaluate(self, ds, steps=None):
    """Test model on data generator."""
    return evaluation.evaluate(model=self, eval_ds=ds, num_eval_steps=steps)

  def fit_batch(self, batch):
    """Update model on batch of sequences of shape [batch x length]."""
    batch = tree.map_structure(jnp.asarray, batch)
    if self._pmap:
      batch = common_utils.shard(batch)
    self._optimizer, metrics, self._dropout_rngs = self._train_fn(
        optimizer=self.optimizer, inputs=batch, dropout_rng=self._dropout_rngs)
    if self._store_metrics:
      metrics = tree.map_structure(onp.array, metrics)
      self._metrics_train.append(metrics)
    return metrics

  def score(self, batch):
    """Predicts logits for given [batch x length] sequences."""
    batch = tree.map_structure(jnp.asarray, batch)
    if self._pmap:
      batch = common_utils.shard(batch)
    logits = self._predict_fn(self.optimizer.target, batch)
    # Undo pmap batching
    if self._pmap:
      logits = jnp.reshape(logits, [-1, logits.shape[-2], logits.shape[-1]])
    return logits

  def evaluate_batch(self, batch):
    """Computes metrics for given [batch x length] sequences."""
    batch = tree.map_structure(jnp.asarray, batch)
    if self._pmap:
      batch = common_utils.shard(batch)
    metrics = self._eval_fn(self.optimizer.target, batch)
    if self._store_metrics:
      metrics = tree.map_structure(onp.array, metrics)
      self._metrics_test.append(metrics)
    return metrics


@gin.configurable
class FlaxLM(FlaxModel):
  """Transformer with causal attention, right shift, and generative sampling."""

  def __init__(self,
               domain=data.protein_domain,
               model_cls=modules.Transformer,
               **kwargs):

    model_cls = model_cls.partial(causal=True)
    super().__init__(
        domain=domain, model_cls=model_cls, cache=True, with_bos=True, **kwargs)

  def preprocess(self, batch, rng, mode):
    del rng
    return preprocess_causal(
        batch=batch,
        bos_token=self._bos_token,
        pad_token=self._pad_token,
        mode=mode)

  @property
  def cache_def(self):
    """Returns the associated autoregressive cache_def."""
    return self._cache_def

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

    if self._pmap:
      prompt = common_utils.shard(prompt)
      cache = self.cache_def.initialize_cache((prompt.shape[1], length))
      cache = jax_utils.replicate(cache)
      rng = jax.random.split(rng, num=len(jax.local_devices()))
    else:
      cache = self.cache_def.initialize_cache((prompt.shape[0], length))

    samples = self._sample_fn(
        prompt=prompt,
        model=self.optimizer.target,
        cache=cache,
        rng=rng,
    )

    # Remove the BOS token from the sampled sequences.
    samples = samples[Ellipsis, 1:]

    # Undo pmap batching
    if self._pmap:
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


@gin.configurable
class FlaxBERT(FlaxModel):
  """Transformer with all-to-all attention and token dropout."""

  def __init__(self,
               domain=data.protein_domain,
               model_cls=modules.Transformer,
               mask_rate=0.15,
               random_token_proportion=0.8,
               mask_token_proportion=0.1,
               **kwargs):
    """Create BERT model.


    For each token in input, masks with probability `mask_rate`. A masked token
    is replaced with:
    - MASK with probability `mask_token_proportion`,
    -  a random token with `random_token_proportion`,
    - left unchanged but included in loss with the remaining probability.

    Args:
      domain: Domain to operate over.
      model_cls: Flax Module operating on sequences.
      mask_rate: Probability of replacing a token and including in the loss
      random_token_proportion: Portion of masked tokens to replace with a
        randomly chosen token.
      mask_token_proportion: Portion of masked tokens to replace with MASK.
      **kwargs: Arguments passed to FlaxModel.
    """
    model_cls = model_cls.partial(causal=False)
    self._mask_rate = mask_rate
    total = random_token_proportion + mask_token_proportion
    if total < 0 or total > 1:
      raise ValueError('Sum of random proportion and mask proportion must be'
                       ' in [0, 1] range.')
    self._masker = BertMasker(
        domain,
        mask_rate=mask_rate,
        mask_token_proportion=mask_token_proportion,
        random_token_proportion=random_token_proportion)

    super().__init__(
        domain=domain,
        model_cls=model_cls,
        cache=False,
        with_mask=True,
        **kwargs)

  def preprocess(self, batch, rng, mode):
    return self._masker(inputs=batch, mode=mode, rng=rng)

  def sample(self, masked_inputs, rng):
    """Fill in MASK positions in inputs."""
    mask_positions = masked_inputs == self.domain.vocab.mask
    logits = self.score(masked_inputs)

    # Mask out MASK token.
    mask = common_utils.onehot(
        jnp.array([self.domain.vocab.mask]),
        num_classes=logits.shape[-1],
        on_value=sampling.LARGE_NEGATIVE)
    logits = logits + mask
    samples = jax.random.categorical(rng, logits=logits)
    infilled = onp.where(mask_positions, samples, masked_inputs)
    return infilled


def preprocess_masked(inputs, random_tokens, mask_token, pad_token, mask_rate,
                      mask_token_proportion, random_token_proportion, mode,
                      rng):
  """Preprocess inputs for masked language modeling.

  Args:
    inputs: [batch x length] input tokens.
    random_tokens: Set of tokens usable for replacing
    mask_token: Int ID to mask blanks with.
    pad_token: Int ID for PAD token. Positions left unchanged.
    mask_rate: Proportion of tokens to mask out.
    mask_token_proportion: Replace this proportion of chosen positions with
      MASK.
    random_token_proportion: Replace this proportion of chosen positions with
      randomly sampled tokens
    mode: Mode key.
    rng: Jax RNG.

  Returns:
    Tuple of [batch x length] inputs, targets, per position weights. targets
      will have random positions masked out with either a MASK token, or a
      randomly chosen token from the vocabulary.
  """
  total = random_token_proportion + mask_token_proportion
  if total < 0 or total > 1:
    raise ValueError('Sum of random proportion and mask proportion must be'
                     ' in [0, 1] range.')
  targets = inputs

  if mode == Mode.predict:
    weights = jnp.full_like(targets, 1)
    masked_inputs = inputs  # Pass through
  else:
    if rng is None:
      if mode is not Mode.eval:
        raise ValueError('Must provide RNG unless in eval mode.')
      # TODO(b/157055145): How to keep same eval set across runs?
      # Make each sequences mask invariant to other members
      # of the batch. Right now there is batch size dependence.
      rng = jrandom.PRNGKey(jnp.sum(inputs))

    # Get positions to leave untouched
    is_pad = inputs == pad_token

    # Positions to mask
    rng, subrng = jax.random.split(rng)
    should_mask = jrandom.bernoulli(subrng, p=mask_rate, shape=inputs.shape)
    should_mask = jnp.where(is_pad, 0, should_mask)  # Don't mask out padding.

    # Generate full array of random tokens.
    rng, subrng = jax.random.split(rng)
    random_ids = jax.random.randint(
        subrng, inputs.shape, minval=0, maxval=len(random_tokens))

    fullrandom = random_tokens[random_ids]
    # Full array of MASK tokens
    fullmask = jnp.full_like(inputs, mask_token)

    # Build up masked array by selecting from inputs/fullmask/fullrandom.
    rand = jax.random.uniform(rng, shape=inputs.shape)
    masked_inputs = inputs
    # Remaining probability mass stays original values after MASK and RANDOM.
    # MASK tokens.
    masked_inputs = jnp.where(rand < mask_token_proportion, fullmask,
                              masked_inputs)
    # Random tokens.
    masked_inputs = jnp.where(
        jnp.logical_and(rand >= mask_token_proportion,
                        rand < mask_token_proportion + random_token_proportion),
        fullrandom, masked_inputs)

    # Only replace positions where `should_mask`
    masked_inputs = jnp.where(should_mask, masked_inputs, inputs)
    weights = should_mask

  return masked_inputs, targets, weights


class BertMasker():
  """Construct BERT masker given a domain."""

  def __init__(self,
               domain,
               mask_rate=0.15,
               mask_token_proportion=0.1,
               random_token_proportion=0.8):
    vocab = domain.vocab
    if vocab.mask is None:
      raise ValueError('Vocabulary must specify a MASK token.')
    special_tokens = [vocab.bos, vocab.eos, vocab.mask, vocab.pad]
    special_tokens = [x for x in special_tokens if x is not None]
    normal_tokens = [x for x in vocab.token_ids if x not in special_tokens]
    self._domain = domain
    self._special_tokens = jnp.array(special_tokens)
    self._normal_tokens = jnp.array(normal_tokens)
    self._mask_rate = mask_rate
    self._mask_token_proportion = mask_token_proportion
    self._random_token_proportion = random_token_proportion

  def __call__(self, inputs, mode, rng):
    inputs, targets, weights = preprocess_masked(
        inputs=inputs,
        mode=mode,
        rng=rng,
        random_tokens=self._normal_tokens,
        mask_token=self._domain.vocab.mask,
        pad_token=self._domain.vocab.pad,
        mask_rate=self._mask_rate,
        mask_token_proportion=self._mask_token_proportion,
        random_token_proportion=self._random_token_proportion)
    return inputs, targets, weights
