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
"""Embedding API for pretrained models."""

import functools
from flax.training import common_utils

import gin
import jax
import jax.numpy as jnp
import tensorflow.compat.v1 as tf

from protein_lm import data
from protein_lm import models
from protein_lm import utils

SMALL_NEGATIVE = -1e10


def _encode_string_sequences(string_sequences, domain, length):
  """Encodes string sequences as sequences of int tokens.

  Args:
    string_sequences: An iterable over strings.
    domain: An instance of VariableLengthDiscreteDomain.
    length: If provided, crop sequences to this length, otherwise use
      domain.length.

  Returns:
    A jax array of shape (batch_size, length) with the encoded sequences.
  """
  if domain is None:
    domain = data.protein_domain

  if length is None:
    length = domain.length

  # Encode sequences, mark the end with a single EOS, and pad with PAD.
  batch = domain.encode(string_sequences, pad=False)

  max_input_length = max(len(s) for s in string_sequences)
  crop_length = min(max_input_length, length)
  # We perform the padding manually since domain.encode(..., pad=True)
  # uses EOS for padding. We use tf directly rather than seq_utils since
  # the latter performs `pre` truncation.
  batch = [list(elem) + [domain.vocab.eos] for elem in batch]
  batch = tf.keras.preprocessing.sequence.pad_sequences(
      batch, maxlen=crop_length, value=domain.vocab.pad,
      padding='post', truncating='post')

  return jnp.asarray(batch)


@gin.configurable
def sum_reducer(embedding, mask):
  """Returns the sum across the unmasked dimensions.

  Args:
    embedding: An array of shape (batch_size, length, emb_size).
    mask: An array of shape (batch_size, length).

  Returns:
    An array of shape (batch_size, emb_size).
  """
  return jnp.sum(embedding * mask[Ellipsis, jnp.newaxis], axis=1)


@gin.configurable
def mean_reducer(embedding, mask):
  """Returns the mean across the unmasked dimensions.

  Args:
    embedding: An array of shape (batch_size, length, emb_size).
    mask: An array of shape (batch_size, length).

  Returns:
    An array of shape (batch_size, emb_size).
  """
  return sum_reducer(embedding, mask) / jnp.sum(mask, axis=-1, keepdims=True)


@gin.configurable
def max_reducer(embedding, mask):
  """Returns the max across the unmasked dimensions.

  Args:
    embedding: An array of shape (batch_size, length, emb_size).
    mask: An array of shape (batch_size, length).

  Returns:
    An array of shape (batch_size, emb_size).
  """
  mask = (-mask + 1) * SMALL_NEGATIVE
  return jnp.max(embedding + mask[Ellipsis, jnp.newaxis], axis=1)


@gin.configurable
def masked_reduce_fn(embedding,
                     inputs,
                     reducer_fn=mean_reducer,
                     domain=None,
                     ignore_eos=False,
                     ignore_bos=True,
                     ignore_pad=True,
                     ignore_mask=True):
  """Takes the mean across the length dimension, ignoring special tokens.

  Args:
    embedding: An array of shape (batch_size, length, emb_size).
    inputs: An array of shape (batch_size, length).
    reducer_fn: A callable to perform the reduction given embedding and mask.
    domain: An instance of VariableLengthDiscreteDomain.
    ignore_eos: Whether to ignore EOS tokens.
    ignore_bos: Whether to ignore BOS tokens.
    ignore_pad: Whether to ignore PAD tokens.
    ignore_mask: Whether to ignore MASK tokens.

  Returns:
    An array of shape (batch_size, emb_size) with the aggregated embeddings.
  """
  if domain is None:
    domain = data.protein_domain

  mask_tokens = []
  if ignore_eos:
    mask_tokens.append(domain.vocab.eos)
  if ignore_bos:
    mask_tokens.append(domain.vocab.bos)
  if ignore_pad:
    mask_tokens.append(domain.vocab.pad)
  if ignore_mask:
    mask_tokens.append(domain.vocab.mask)

  mask = jnp.ones_like(inputs)
  for token in mask_tokens:
    if token is not None:
      mask *= inputs != token

  return reducer_fn(embedding, mask)


@functools.lru_cache(10)
def get_embed_fn(model=None,
                 checkpoint_dir=None,
                 domain=None,
                 output_head='output_emb',
                 reduce_fn=None,
                 length=None):
  """Get a function that maps sequences to fixed-length embedding vectors.

  Args:
    model: A FlaxModel (e.g. FlaxLM or FlaxBERT).
    checkpoint_dir: A string directory where the model checkpoint is stored.
    domain: An instance of VariableLengthDiscreteDomain.
    output_head: Which model output to return. See embed.FlaxModel.
    reduce_fn: Postprocessing function to apply on top of embeddings, such as
      `masked_reduce_fn`. The reduce_fn takes and input padded embeddings
      and padded inputs (to allow masking the pad dimensions). If None, no
      reduction is made.
    length: Input sequences will be cropped and padded to have length
      N = min(max_len, length), where max_len is the length of the longest
      sequence in the input data. If length is None, domain.length is used when
      computing N.

  Returns:
    Function which accepts sequences and returns batched embeddings. If the
      the sequences are strings, we first encode them into the domain.
      Otherwise, we assume that they are already encoded.
  """
  if model is None:
    if checkpoint_dir is None:
      raise ValueError('Must provide a loaded model or checkpoint directory.')
    # Note that this assumes that the model_cls is stored in the config dict.
    model = models.load_model(checkpoint_dir=checkpoint_dir)
  else:
    if checkpoint_dir is not None:
      raise ValueError('Provide only one of `model` or checkpoint directory.')

  if domain is None:
    domain = data.protein_domain

  def predict_fn(model_target, inputs):
    emb = models.predict_step(
        model_target,
        inputs,
        preprocess_fn=model.preprocess,
        output_head=output_head)

    if reduce_fn:
      # Pass the inputs to allow padding-aware aggregation.
      emb = reduce_fn(emb, inputs)
    return emb

  if model.pmap:
    p_predict_step = jax.pmap(predict_fn, axis_name='batch')
  else:
    p_predict_step = predict_fn

  def _embed(protein_sequences):
    """Encode proteins into a batch, embed, and run reduce_fn on output."""
    if isinstance(protein_sequences[0], str):
      batch = _encode_string_sequences(protein_sequences,
                                       domain=domain, length=length)
    else:
      if not domain.are_valid(protein_sequences).any():
        raise ValueError('Input int-encoded sequences are not valid members '
                         'of input domain.')
      batch = protein_sequences

    if model.pmap:
      batch = common_utils.shard(batch)
    result = p_predict_step(model.optimizer.target, batch)

    if model.pmap:
      # Combine the leading two dimensions (ndevices, batch_size / n_devices)
      result = jax.numpy.reshape(result, [-1] + list(result.shape[2:]))
    return result

  return _embed


@gin.configurable
class ProteinLMEmbedder(object):
  """Embeddings from a pretrained language model.

  Stateful wrapper around get_embed_fn that calls the embed_fn on batches.
  """

  def __init__(self,
               model=None,
               checkpoint_dir=None,
               domain=None,
               output_head='output_emb',
               reduce_fn=None,
               length=None,
               batch_size=64):
    """Creates an instance of this class."""
    self._embed_fn = get_embed_fn(
        model=model,
        checkpoint_dir=checkpoint_dir,
        domain=domain,
        output_head=output_head,
        reduce_fn=reduce_fn)
    self._batch_size = batch_size
    self._domain = domain
    self._length = length

  def __call__(self, sequences):
    """Embeds int or string sequences."""
    if isinstance(sequences[0], str):
      sequences = _encode_string_sequences(sequences, domain=self._domain,
                                           length=self._length)
    return utils.batch_apply(self._embed_fn, sequences, self._batch_size)
