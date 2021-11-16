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

"""Classes representing vocabularies (alphabets) over protein strings."""

import itertools
from typing import Iterable, Optional, Set, Sequence, List

import gin
import tensorflow as tf
import tensorflow_probability as tfp


@tf.keras.utils.register_keras_serializable()
@gin.configurable
class Vocabulary:
  """Vocabulary to encode string into integers."""

  MASK = '*'

  def __init__(self,
               tokens,
               specials,
               padding = '_',
               order = None):
    """Initializess the vocabulary.

    Args:
      tokens: (Iterable) the main tokens of the vocabulary.
      specials: (Iterable) the special characters, such as EOS, gap, etc.
      padding: the character for padding.
      order: the order between the tokens(0), specials (1) and padding(2).
    """
    self.tokens = tuple(tokens)
    self.specials = tuple(specials)
    self._padding = padding
    self._order = order
    groups = (tokens, specials, [padding])
    order = range(3) if order is None else order
    groups = [groups[i] for i in order]
    self._voc = list(itertools.chain.from_iterable(groups)) + [self.MASK]
    self._indices = {t: i for i, t in enumerate(self._voc)}

  def get_config(self):
    """For keras serialization compatibility."""
    return dict(tokens=self.tokens,
                specials=self.specials,
                padding=self._padding,
                order=self._order)

  def __len__(self):
    return len(self._voc)

  def __contains__(self, token):
    return token in self._indices

  def encode(self, s, skip = None):
    """Returns a list of int-valued tokens representing the input sequence."""
    skip = set() if skip is None else set(skip)
    return [self._indices[c] for c in s if c not in skip]

  def decode(self,
             encoded,
             remove_padding=True,
             remove_specials=True):
    """Returns string representing the input sequence from its int encoding."""
    remove = set(self.specials) if remove_specials else set()
    if remove_padding:
      remove.add(self._padding)
    return ''.join(
        [x for x in [self._voc[i] for i in encoded] if x not in remove])

  @property
  def padding_code(self):
    return self.get(self._padding)

  def get_specials(self, with_padding = True):
    return (self.specials if not with_padding
            else self.specials + (self._padding,))

  @property
  def mask_code(self):
    return self.get(self.MASK)

  def get(self, token, default_value=None):
    """Returns the int encoding of the token if exists or the default value."""
    return self._indices.get(token, default_value)

  def compute_mask(self, inputs, tokens):
    """Computes mask for a batch of input tokens.

    Args:
      inputs: a tf.Tensor of indices.
      tokens: a sequence of strings containing all tokens of interest.

    Returns:
      A binary tf.Tensor of the same size as `inputs`, with value True for
      indices in `inputs` which correspond to no token in `tokens`.
    """
    mask = tf.ones_like(inputs, dtype=tf.bool)
    for token in tokens:
      idx = self._indices[token]
      mask = tf.math.logical_and(mask, inputs != idx)
    return mask

  def padding_mask(self, inputs):
    """Computes padding mask for a batch of input tokens."""
    return self.compute_mask(inputs, [self._padding])

  def special_token_mask(self,
                         inputs,
                         with_mask_token = True):
    """Computes special token mask for a batch of input tokens."""
    tokens = (self.specials + (self.MASK,) if with_mask_token
              else self.specials)
    return self.compute_mask(inputs, tokens)

  def translate(self, target):
    """Mapping to translate between two instances of Vocabulary.

    Args:
      target: The target Vocabulary instance to be "translated" to.

    Returns:
      A list, mapping the ints of the self vocabulary to the ones of the target.
    """
    return [target.get(token, target.padding_code) for token in self._voc]


proteins = Vocabulary(
    tokens='LAVGESIRDTKPFNQYHMWCUOBZX',
    specials=('<', '>'),
    padding='_',
    order=(2, 0, 1)
)
gin.constant('vocabulary.proteins', proteins)


alternative = Vocabulary(
    tokens='ACDEFGHIKLMNPQRSTVWYBOUXZ',
    specials=('.', '-', '<', '>'),
    padding='_',
    order=(0, 1, 2)
)
gin.constant('vocabulary.alternative', alternative)


@gin.configurable
def get_default(vocab = alternative):
  """A convenient function to gin configure the default vocabulary."""
  return vocab


@gin.configurable
class Sampler:
  """Samples (integer-valued) tokens uniformly at random in the vocabulary.

  (Ignores any special tokens in the vocabulary).
  """

  def __init__(self,
               vocab = proteins,
               logits = None):
    self._integer_tokens = [vocab.get(tk) for tk in vocab.tokens]
    self.num_tokens = len(vocab.tokens)
    self._logits = (tf.zeros(self.num_tokens, tf.float32) if logits is None
                    else tf.convert_to_tensor(logits, tf.float32))
    self._sampler = tfp.distributions.Categorical(
        logits=self._logits, dtype=tf.int32)

  def sample(self, shape):
    return tf.gather(self._integer_tokens, self._sampler.sample(shape))
