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

"""Input pipeline for a text8 / enwik8 dataset."""

import os
from typing import Dict
import zipfile

import jax
import ml_collections
import numpy as np
import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE
Features = Dict[str, tf.Tensor]


def _crop(x, max_length):
  """Select (optionally random) crop from sequence."""
  # Optionally sample random starting position.
  start = tf.random.uniform(
      (), dtype=tf.int32, maxval=tf.maximum(1, tf.shape(x)[0] - max_length + 1))

  x = x[start:(start + max_length)]
  return x


class CharLevelTokenizer():
  """Tokenizes strings to a char-level embedding."""

  def __init__(self, raw_train):
    assert isinstance(raw_train, str)
    train_chars = sorted(set(raw_train))
    chars = [' ',
             'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    assert set(train_chars) == set(chars), f'{train_chars} != {chars}'

    # Do not use zero as input.
    self.i_to_s = {i+1: char for i, char in enumerate(chars)}
    self.s_to_i = {char: i+1 for i, char in enumerate(chars)}

    self.maximum = max(self.i_to_s.keys())

  def vocab_size(self):
    return len(self.i_to_s) + 1  # For the zero token.

  def decode_unknown(self, i):
    if i == self.maximum + 1:
      return '_'
    else:
      return str(i)

  def tokenize(self, string):
    return np.array([self.s_to_i[s] for s in string])

  def detokenize(self, tokens):
    return ''.join([
        self.i_to_s[i] if i in self.i_to_s else self.decode_unknown(i)
        for i in tokens
    ])


class ByteTokenizer():
  """Tokenizes strings to a char-level embedding."""

  def __init__(self):
    # For the zero token everything is shifted.
    self.maximum = 255 + 1

  def vocab_size(self):
    return self.maximum + 1  # For the zero token.

  def tokenize(self, string):
    # Add one for special token zero.
    return np.array(list(string), dtype=np.int32) + 1

  def detokenize(self, tokens):
    """Detokenizes an array of tokens to a string."""
    tokens = np.array(tokens)

    # Shift and deal with padding token.
    tokens = tokens - 1
    tokens[tokens < 0] = 48  # ASCII index for digit zero

    # Convert to bytes.
    tokens = np.array(tokens, dtype=np.uint8)
    # Create byte string.
    byte_string = b''.join(tokens)
    # Decode the byte string.
    string = byte_string.decode('utf-8', errors='replace')
    return string


def prepare_eval_ds_from_tokens(config,
                                tokens):
  """This function prepares an eval dataset to have the correct context.

  What this function does in words:
    1. It batches the tokens data.
    2. In case of a context, it computes which indices _precede_ the batch,
       and then those are used to retrieve the tokens preceding the batches.
  The main reason for putting this in a seperate function is the context logic,
  which is a little tedious if the context length and the sequence length are
  not the same.

  Args:
    config: A ml_collections config.
    tokens: An np.ndarray containing integers.

  Returns:
    A tensorflow dataset.
  """
  length = len(tokens)
  assert length % config.seq_length == 0

  # Here the character indices of the datapoints are collected.
  input_idcs = np.arange(length).reshape(-1, config.seq_length)

  tokens_inputs = tokens[input_idcs][:, :, None]  # Add channel axis.

  if config.context_length > 0:
    start_idcs = input_idcs[:, 0:1]

    # Context idcs start at the same index, since they will be applied to a
    # shifted array of exactly context_size.
    context_idcs = start_idcs + np.arange(config.context_length)[None, :]

    tokens_padded = np.concatenate(
        [np.zeros(config.context_length, dtype=tokens.dtype), tokens])
    tokens_context = tokens_padded[context_idcs]
    ds = tf.data.Dataset.from_tensor_slices(
        {'inputs': tokens_inputs, 'context': tokens_context})
  else:
    ds = tf.data.Dataset.from_tensor_slices(
        {'inputs': tokens_inputs})

  ds = ds.batch(config.test_batch_size, drop_remainder=False)
  ds = ds.prefetch(AUTOTUNE)

  return ds


def get_datasets(config,
                 *,
                 shuffle_buffer_size = 1000_000):
  """Load and return dataset of batched examples for use during training."""
  assert config.batch_size % jax.process_count() == 0
  per_process_batch_size = config.batch_size // jax.process_count()

  if config.dataset_name == 'text8':
    path = os.path.join(config.text8_path, 'text8.zip')
    with tf.io.gfile.GFile(path, 'rb') as z:
      raw = zipfile.ZipFile(z).read('text8').decode('utf-8')
  elif config.dataset_name == 'enwik8':
    path = os.path.join(config.text8_path, 'enwik8.zip')
    with tf.io.gfile.GFile(path, 'rb') as z:
      raw = zipfile.ZipFile(z).read('enwik8')  # Do not decode opposed to text8.
  else:
    raise ValueError

  # Standard text8/enwik8 splits, both datasets have the same number of tokens.
  assert len(raw) == 100000000, f'{len(raw)} != 10000000'
  train_data = raw[:90000000]
  eval_data = raw[90000000:95000000]
  test_data = raw[95000000:]

  if config.dataset_name == 'text8':
    tokenizer = CharLevelTokenizer(train_data)
  elif config.dataset_name == 'enwik8':
    tokenizer = ByteTokenizer()
  else:
    raise ValueError

  train_tokens = tokenizer.tokenize(train_data)

  # Pad with zero tokens for the first batch.
  if config.context_length > 0:
    pad = np.zeros(shape=(config.context_length,), dtype=train_tokens.dtype)
    train_tokens = np.concatenate([pad, train_tokens], axis=0)

  chunk_size = config.seq_length + config.context_length

  train_ds = tf.data.Dataset.from_tensor_slices(train_tokens)

  # We batch sequences of 4 times chunk_size, and then crop with size
  # 1 x 'chunk_size' for the purpose of augmenting the data somewhat. Although
  # this does leave out some crops over the borders of the 4 * 'chunk_size'
  # chunks, in practice this is not really an issue and also done by others
  # in a similar fashion.
  train_ds = train_ds.batch(4 * chunk_size, drop_remainder=True)

  # We are not sharding here, as the data is small enough, note that we rely on
  # tensorflow random to produce different orders.
  train_ds = train_ds.shuffle(shuffle_buffer_size)

  # We take random crops of size 'chunk_size' from the previously chunked
  # pieces of '4 x chunk_size'. This is a form of data augmentation.
  def crop(batch):
    return _crop(batch, config.seq_length + config.context_length)
  train_ds = train_ds.map(crop, num_parallel_calls=AUTOTUNE)

  train_ds = train_ds.batch(
      per_process_batch_size, drop_remainder=True, num_parallel_calls=AUTOTUNE)

  # For the training chunks, this final step is need to separate the context
  # from the inputs.
  def prepare_inputs(batch):
    if config.context_length > 0:
      context = batch[:, :config.context_length]
      inputs = batch[:, config.context_length:, None]  # Channel axis.
      return {'inputs': inputs, 'context': context}
    else:
      return {'inputs': batch[:, :, None]}  # Channel axis.

  train_ds = train_ds.map(prepare_inputs, num_parallel_calls=AUTOTUNE)

  train_ds = train_ds.prefetch(AUTOTUNE)

  eval_tokens = tokenizer.tokenize(eval_data)
  eval_ds = prepare_eval_ds_from_tokens(config, eval_tokens)

  test_tokens = tokenizer.tokenize(test_data)
  test_ds = prepare_eval_ds_from_tokens(config, test_tokens)

  return train_ds, eval_ds, test_ds, tokenizer
