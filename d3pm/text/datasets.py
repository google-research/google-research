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

"""Single import to access / gin register tasks."""

import functools
import os
from typing import Callable, Mapping, Optional
import urllib
import zipfile

import gin
import numpy as np
import seqio
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tf_text

from d3pm.text import types
from d3pm.text import utils

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

LM1B_VOCABULARY_PATH = os.path.join(DATA_DIR, 'lm1b-sentencepiece-8k.model')

DatasetFn = Callable[Ellipsis, Mapping[str, types.Dataset]]


class DatasetRegistry:
  """A registry containing all datasets supported by the codebase."""

  def __init__(self):
    self.datasets = {}

  def clear(self):
    self.datasets = {}

  def register(self, name, dataset_fn):
    self.datasets[name] = dataset_fn

  def list_datasets(self):
    """Returns a string containing a list of the available datasets."""

    msg = 'Available Datasets:\n\n'
    for name in self.datasets:
      msg += '* ' + name + '\n'

    return msg

  def load(self, name, *, batch_size,
           **kwargs):
    """Load a dataset registered with the DatasetRegistry."""
    if not isinstance(name, str):
      raise TypeError('Dataset name must be a string.')

    if name not in self.datasets:
      info_string = self.list_datasets()
      raise ValueError(
          f'Unable to find a dataset with the name {name}.\n\n{info_string}.')

    dataset_fn = self.datasets[name]
    return dataset_fn(batch_size=batch_size, **kwargs)


_REGISTRY = DatasetRegistry()


def register(dataset_fn, name=None):
  """Load a dataset registered with the D3PM dataset registry.

  Args:
    dataset_fn: a dataset function to register.
    name: the name of the dataset to load.

  Returns:
    a training and validation dataset.
  """
  if name is None:
    name = dataset_fn.__name__

  _REGISTRY.register(name, dataset_fn)

  return dataset_fn


def crop(x, max_length, sample):
  """Select (optionally random) crop from sequence."""
  if sample:
    start = tf.random.uniform(
        (),
        dtype=tf.int32,
        maxval=tf.maximum(1,
                          tf.shape(x)[0] - max_length + 1))
  else:
    start = 0

  x = x[start:(start + max_length)]
  return x


def list_datasets():
  """Lists the datasets currently loaded by the codebase."""
  return _REGISTRY.list_datasets()


@gin.configurable(module='datasets')
def load(name,
         *,
         batch_size,
         split=None,
         preprocessors=None,
         prefetch=True,
         **kwargs):
  """Load a dataset registered with the D3PM dataset registry.

  Args:
    name: the name of the dataset to load.
    batch_size: the batch size to use with the dataset.
    split: if not None, specifies one of ['train', 'valid', 'test'] splits to
      load instead of all datasets.
    preprocessors: a list of functions to apply to the datasets.
    prefetch: if True, will prefetch datasets.
    **kwargs: any kwargs to be passed to the corresponding load_dataset
      function.

  Returns:
    a training and validation dataset.
  """
  datasets = _REGISTRY.load(name, batch_size=batch_size, **kwargs)

  if preprocessors is not None:
    datasets = {k: v.map(preprocessors) for k, v in datasets.items()}

  if prefetch:
    datasets = {k: v.prefetch() for k, v in datasets.items()}

  if split is not None:
    return datasets[split]
  else:
    return datasets


class FakeVocab:
  """A toy vocab wrapping a tf_text tokenizer."""

  def __init__(self, vocab_size):
    self.vocab_size = vocab_size

  def encode(self, text):
    raise NotImplementedError

  def decode(self, tokens):
    tokens = tf.strings.as_string(tokens)
    text = tf.strings.reduce_join(tokens, axis=-1)
    return text


class TFTextVocabulary:
  """A toy vocab wrapping a tf_text tokenizer."""

  def __init__(self, tokenizer, num_extra_tokens=0):
    self.tokenizer = tokenizer
    self.num_extra_tokens = num_extra_tokens

  @property
  def vocab_size(self):
    return 27 + self.num_extra_tokens

  def encode(self, text, shape=None):
    """Tokenizes a text example."""
    tokens = self.tokenizer.tokenize(text)

    if shape is not None:
      tokens.set_shape(shape)

    tokens = tokens - 97
    mask = tf.cast(tokens < 0, tf.int32)
    tokens = mask * tf.constant(
        value=26, shape=tokens.shape) + (1 - mask) * tokens
    return tokens

  def encode_tf(self, text, shape=None):
    return self.encode(text, shape=shape)

  def decode(self, tokens):
    mask = tf.cast(tokens == 26, tf.int32)
    extra_mask = tf.cast(tokens > 26, tf.int32)

    tokens = tokens + 97
    tokens = mask * tf.constant(32, shape=tokens.shape) + (1 - mask) * tokens
    tokens = extra_mask * tf.constant(
        45, shape=tokens.shape) + (1 - extra_mask) * tokens

    return self.tokenizer.detokenize(tokens)

  def decode_tf(self, text):
    return self.decode(text)


class D3PMVocabulary:
  """A toy vocab wrapping a tf_text tokenizer."""

  def __init__(self, vocab, num_extra_tokens = 0):
    self.vocab = vocab
    self.num_extra_tokens = num_extra_tokens

  @property
  def vocab_size(self):
    length = self.vocab.vocab_size
    return length + self.num_extra_tokens

  def encode(self, text, **kwargs):
    """Tokenizes a text example."""
    return self.vocab.encode(text, **kwargs)

  def encode_tf(self, text, **kwargs):
    """Tokenizes a text example."""
    return self.vocab.encode_tf(text, **kwargs)

  def decode(self, tokens, **kwargs):
    tokens = tf.clip_by_value(tokens, 0, self.vocab.vocab_size - 1)
    return self.vocab.decode(tokens, **kwargs)

  def decode_tf(self, tokens, **kwargs):
    tokens = tf.clip_by_value(tokens, 0, self.vocab.vocab_size - 1)
    return self.vocab.decode_tf(tokens, **kwargs)


class PermutationVocab:
  """A vocab that applies a permutation."""

  def __init__(self, vocab, permutation):
    self.vocab = vocab
    self.permutation = tf.constant(np.array(permutation).astype(np.int32))
    self.inverse = tf.constant(np.argsort(permutation).astype(np.int32))

  @property
  def vocab_size(self):
    return self.vocab.vocab_size

  def encode(self, text, **kwargs):
    """Tokenizes a text example."""
    return tf.gather(self.inverse, self.vocab.encode(text, **kwargs))

  def decode(self, tokens, **kwargs):
    return self.vocab.decode(tf.gather(self.permutation, tokens), **kwargs)

  def apply_permutation(self, tokens, **kwargs):
    del kwargs

    return tf.gather(self.inverse, tokens)


@functools.partial(register, name='lm1b')
@gin.configurable(denylist=['batch_size'], module='datasets')
def load_lm1b(
    batch_size,
    max_length = 64,
    pack=True,
    num_extra_tokens = 0,
    permutation_file=None,
    delimit_sentences = True,
    shuffle_buffer_size = 2048,
    alt_vocab_path = None,
):
  """Load the LM1B dataset for use with a D3PM model.

  Args:
    batch_size (int): total batch size (multiply by num_devices for multi-device
      training.
    max_length (int): maximum sequence length (will be padded to this length).
    pack (bool): if True, packs sequences with max_length tokens (multiple
      sequences will be packed together and cropped to max length).
    num_extra_tokens: the number of extra tokens to add to the vocab.
    permutation_file: File containing a permutation to apply to the vocab.
    delimit_sentences: if True, adds <S>, </S> markers to each sentence.
    shuffle_buffer_size: size of shuffle buffer to use.
    alt_vocab_path: if provided, overrides the default vocabulary path.

  Returns:
    train and validation datasets.
  """
  datasets = tfds.load('lm1b')

  if not datasets:
    datasets = {
        split: tfds.load('lm1b', split=split) for split in ['train', 'test']
    }

  vocab_path = alt_vocab_path if alt_vocab_path else LM1B_VOCABULARY_PATH
  vocab = seqio.SentencePieceVocabulary(vocab_path)
  vocab = D3PMVocabulary(vocab, num_extra_tokens=num_extra_tokens)

  for split, ds in datasets.items():
    ds = ds.map(lambda x: x['text'])

    if delimit_sentences:
      ds = ds.map(lambda x: tf.strings.reduce_join(['<S>', x, '</S>']))

    ds = ds.map(vocab.encode_tf)
    ds = ds.prefetch(tf.data.AUTOTUNE).shuffle(shuffle_buffer_size)

    if pack:
      ds = ds.unbatch().batch(max_length, drop_remainder=False)

    ds = ds.padded_batch(
        batch_size,
        padded_shapes=max_length,
        padding_values=0,
        drop_remainder=True)

    ds = ds.map(lambda x: {'targets': x})

    if permutation_file:
      with tf.io.gfile.Open(permutation_file, 'rb') as fp:
        permutation = np.load(fp)
      vocab = PermutationVocab(vocab, permutation)

      def reencode(batch):
        return {'targets': vocab.apply_permutation(batch['targets'])}

      ds = ds.map(reencode)

    datasets[split] = ds

  return utils.wrap_datasets(
      train=datasets['train'], test=datasets['test'], vocab=vocab)


@functools.partial(register, name='text8')
@gin.configurable(denylist=['batch_size'], module='datasets')
def load_text8(
    batch_size,
    max_length = 256,
    repeat = -1,
    num_extra_tokens = 0,
    sample_crop_train = True,
    alt_data_dir = None,
):
  """Load the 27-char text8 dataset.

  This function will write the text8 dataset to the `data` directory if it is
  not already found. This will be pulled from a zip file found at
  http://mattmahoney.net/dc/text8.zip if not provided in the data directory.

  Args:
    batch_size (int): total batch size (multiply by num_devices for multi-device
      training.
    max_length (int): maximum sequence length (will be padded to this length).
    repeat (int): number of times to repeat the dataset.
    num_extra_tokens: the number of extra tokens to add to the vocab.
    sample_crop_train: if True, will randomly crop segments from the training
      set (but test and valid sets will still be cropped canonically).
    alt_data_dir: if provided, a data directory to store text8 data, overriding
      the default.

  Returns:
    train and validation datasets.
  """

  def split_chars(arr):
    return tf.sparse.reshape(tf.compat.v1.string_split([arr], sep=''), (-1, 1))

  data_dir = alt_data_dir if alt_data_dir is not None else DATA_DIR

  tokenizer = tf_text.UnicodeCharTokenizer()
  vocab = TFTextVocabulary(tokenizer, num_extra_tokens=num_extra_tokens)

  def tokenize(text, shape=(max_length,)):
    text = tf.strings.reduce_join(text, axis=0)
    tokens = vocab.encode(text, shape=shape)
    return tokens

  if not tf.io.gfile.exists(os.path.join(data_dir, 'text8.train.txt')):
    if not tf.io.gfile.exists(os.path.join(data_dir, 'text8.zip')):
      url = 'http://mattmahoney.net/dc/text8.zip'
      print('Downloading text8 from URL {}.'.format(url))
      urllib.request.urlretrieve(url, data_dir)

    with tf.io.gfile.GFile(os.path.join(data_dir, 'text8.zip'), 'rb') as f:
      rawdata = zipfile.ZipFile(f).read('text8').decode('utf-8')

    splits = {
        'train': rawdata[:90000000],
        'valid': rawdata[90000000:95000000],
        'test': rawdata[95000000:],
    }

    for split, data in splits.items():
      with tf.io.gfile.GFile(
          os.path.join(DATA_DIR, 'text8.' + split + '.txt'), 'w') as f:
        f.write(data)

  def load_text8_split(split, random_crop=False, repeat=-1):
    path = os.path.join(data_dir, 'text8.' + split + '.txt')

    ds = tf.data.TextLineDataset(path).map(split_chars).unbatch()

    if random_crop:
      ds = ds.batch(2 * max_length, drop_remainder=True)
      ds = ds.map(lambda x: tf.reshape(tf.sparse.to_dense(x), (-1,)))
      ds = ds.map(functools.partial(tokenize, shape=(2 * max_length,))).cache()

      fn = functools.partial(
          crop,  # pylint: disable=protected-access
          max_length=max_length,
          sample=True)

      ds = ds.map(fn).repeat(repeat)

      def set_shape(x):
        x.set_shape(max_length)
        return x

      ds = ds.map(set_shape)

    else:
      ds = ds.batch(max_length, drop_remainder=True)
      ds = ds.map(lambda x: tf.reshape(tf.sparse.to_dense(x), (-1,)))
      ds = ds.map(tokenize).cache().repeat(repeat)

    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return ds

  random_crop = {
      'train': sample_crop_train,
      'valid': False,
      'test': False,
  }

  datasets = {
      k: load_text8_split(k, random_crop=v, repeat=repeat)
      for k, v in random_crop.items()
  }

  def rename(batch):
    return {'targets': batch}

  datasets = {split: ds.map(rename) for split, ds in datasets.items()}

  return utils.wrap_datasets(
      train=datasets['train'],
      valid=datasets['valid'],
      test=datasets['test'],
      vocab=vocab)
