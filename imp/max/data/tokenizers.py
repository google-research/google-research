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

"""A simple tokenizer interface with basic implementations."""

from typing import Any, Sequence

from dmvr import tokenizers
import seqio
import tensorflow as tf

from imp.max.core import constants
from imp.max.data import loading


def get_default_vocabulary():
  """Returns the default t5 vocabulary."""
  return seqio.SentencePieceVocabulary(
      constants.Tokenizer.T5_DEFAULT_SPM_PATH,
      extra_ids=constants.Tokenizer.T5_DEFAULT_EXTRA_IDS)

VOCABULARY: seqio.SentencePieceVocabulary = get_default_vocabulary()


# ----------------------------------------------------------------------
# ------------------------- Tokenizer Classes --------------------------
# ----------------------------------------------------------------------

# Base Tokenizers from DMVR
TextTokenizer = tokenizers.TextTokenizer
WordTokenizer = tokenizers.WordTokenizer
BertTokenizer = tokenizers.BertTokenizer


# TODO(b/232303743): move all tokenization to SeqIO and remove this class.
class SentencePieceTokenizer(tokenizers.SentencePieceTokenizer):
  """SentencePieceTokenizer overridden by a T5 tokenizer.

  Allows for initialization of extra tokens like `<extra_id_0>` which is not
  handled by the DMVR `SentencePieceTokenizer`.
  """

  def __init__(self,
               vocabulary,
               ignore_bos_eos = False):
    """Initializes the `SentencePieceTokenizer`.

    Args:
      vocabulary: a `seqio.SentencePieceVocabulary`
      ignore_bos_eos: if True, ignores appending bos/eos. This option should be
        set when using T5 tokenizers, as they may not automatically append
        bos/eos tokens. Otherwise, tokenization may break the DMVR tokenization
        semantics, causing unintended truncation of the first/last token in each
        sequence.
    """
    self._vocabulary = vocabulary
    self._ignore_bos_eos = ignore_bos_eos
    super().__init__(self._vocabulary.sentencepiece_model_file)
    self._sp_model = self._vocabulary.tokenizer

  def initialize(self):
    """Runs any additional initialization."""
    # Override the tokenizer model to properly load the extra IDs
    self._tf_sp_model = self._vocabulary.tf_tokenizer

  def string_tensor_to_indices(self,
                               string_tensor,
                               prepend_bos = False,
                               append_eos = False,
                               max_num_tokens = 32,
                               **unused_kwargs):
    """Encodes a tensor of strings to indices.

    Args:
      string_tensor: the input tensor of strings.
      prepend_bos: if True, prepends the bos token unless ignore_bos_eos is set.
      append_eos: if True, appends the eos token unless ignore_bos_eos is set.
      max_num_tokens: the number of tokens that all sequences will be padded
        or truncated to.

    Returns:
      the output tensor of indices.

    Raises:
      RuntimeError: if initialize() was not called.
    """

    if self._tf_sp_model is None:
      raise RuntimeError('Model was not initialized. Call `initialize` method.')

    tokenized = self._tf_sp_model.tokenize(string_tensor)

    if not self._ignore_bos_eos:
      tokenized = tokenized if prepend_bos else tokenized[Ellipsis, 1:]
      tokenized = tokenized if append_eos else tokenized[Ellipsis, :-1]

    # Pad to `max_num_tokens`.
    shape = None if max_num_tokens is None else (None, max_num_tokens)
    tokenized = tokenized.to_tensor(default_value=self._pad_token, shape=shape)
    return tokenized


# ----------------------------------------------------------------------
# ------------------------ Processing functions ------------------------
# ----------------------------------------------------------------------


def tokenize(features, tokenizer,
             raw_string_name, tokenized_name, prepend_bos,
             append_eos, max_num_tokens,
             keep_raw_string):
  """Tokenize raw string with tokenizer.

  Args:
    features: A dictionary of features.
    tokenizer: An instance of a text tokenizer.
    raw_string_name: The name of the raw sring feature in features.
    tokenized_name: The name of the desired tokenized feature in the output.
    prepend_bos: Whether or not to prepend BOS in the tokenizer.
    append_eos: Whether or not to append EOS in the tokenizer.
    max_num_tokens: Number of tokens in final result. The tokenized sentence
      will be either crop or padded using the tokenizer pad id.
    keep_raw_string: Whether or not to keep the raw string in the output.

  Returns:
    A FeaturesDict containing the tokenized string.
  """
  raw_caption = features[raw_string_name]
  tokenized = tokenizer.string_tensor_to_indices(
      raw_caption,
      prepend_bos=prepend_bos,
      append_eos=append_eos,
      max_num_tokens=max_num_tokens)
  if not keep_raw_string:
    del features[raw_string_name]
  features[tokenized_name] = tokenized
  return features


def crop_or_pad_words(words,
                      max_num_words,
                      pad_value = 0):
  """Crop or pad given sequence of word indices.

  Args:
    words: Tensor of shape [T, sentence_length] of word indices.
    max_num_words: Maximum number of words in final result.
    pad_value: Value to be used in paddings.

  Returns:
    A Tensor of shape [T, max_num_words].
  """
  if max_num_words < 1:
    raise ValueError(f'Invalid number of words ({max_num_words}) requested.')
  num_words = tf.shape(input=words)[1]
  words = tf.pad(
      tensor=words[:, :max_num_words],
      paddings=((0, 0), (0, tf.maximum(0, max_num_words - num_words))),
      constant_values=pad_value)
  words.set_shape((None, max_num_words))
  return words


def get_tokenizer(tokenizer = constants.T5_EN):
  """Returns matching tokenizer."""
  tokenizer = tokenizer.lower()

  if tokenizer == constants.T5_EN:
    return SentencePieceTokenizer(VOCABULARY, ignore_bos_eos=True)

  vocabulary_path = loading.get_vocab_path(tokenizer)
  if constants.HOWTO100M in tokenizer:
    return WordTokenizer(vocabulary_path)
  elif constants.BERT in tokenizer:
    return BertTokenizer(vocabulary_path)
  else:
    raise ValueError(f'Tokenizer {tokenizer} not supported.')
