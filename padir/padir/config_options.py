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

"""Configuration options for fast decoding."""

import enum

import gin
import jax.numpy as jnp
import seqio

from padir.padir.utils import vocab_utils


_VOCAB_FILE_BINDING = 'seqio.SentencePieceVocabulary.sentencepiece_model_file'
_VOCAB_SIZE_BINDING = 'network.PadirModelConfig.vocab_size'
_EXTRA_IDS_BINDING = 'seqio.SentencePieceVocabulary.extra_ids'
_LENGTH_ID_BINDING = 'network.PadirModelConfig.length_id'

_CUSTOM_VOCAB_FILE = None


@gin.constants_from_enum
@enum.unique
class DecoderInputScheme(enum.Enum):
  """Options to initialize the decoder inputs."""

  RANDOM = 'random'  # Random tokens.
  MASK = 'mask'  # [MASK] tokens.
  MASK_RANGE = 'mask_range'  # [MASK_1] ... [MASK_N] tokens.


@gin.constants_from_enum
@enum.unique
class LossWeightingScheme(enum.Enum):
  """Options for decoder loss weights."""

  ALL = 'all'  # all target tokens (excluding padding).
  NOISE = 'noise'  # all noised tokens + bos + eos
  MASK = 'mask'  # [MASK] tokens + bos + eos (omits noisy but correct/random).


@gin.constants_from_enum
@enum.unique
class RemaskingScheme(enum.Enum):
  """Options to remask decoder inputs across decoding iterations."""

  INITIAL_TOKENS = 'initial_tokens'  # Tokens from decoder initialization.
  PREVIOUS_TOKENS = 'previous_tokens'  # Tokens from the previous iteration.
  LOGITS_SUM = 'logits_sum'  # Obtained by summing logits across iterations.

  def get_replacements(
      self,
      initial_tokens,  # [B, L]
      previous_tokens,  # [B, L]
      logits_sum,  # [B, L, V]
  ):
    """Returns the replacement tokens based on the remasking scheme.

    Args:
      initial_tokens: [B, L] int32 tokens used as initial decoder input tokens.
      previous_tokens: [B, L] int32 tokens used as decoder input tokens for the
        current decoding iteration.
      logits_sum: [B, L, V] logits sum across all decoding iterations so far.

    Returns:
      int32 tokens to use as replacements
    """
    if self == RemaskingScheme.INITIAL_TOKENS:
      replacements = initial_tokens
    elif self == RemaskingScheme.PREVIOUS_TOKENS:
      replacements = previous_tokens
    elif self == RemaskingScheme.LOGITS_SUM:
      replacements = jnp.argmax(logits_sum, axis=-1)
    else:
      assert False
    assert replacements.ndim == 2  # [B, L]
    return replacements


@gin.constants_from_enum
@enum.unique
class Vocab(enum.Enum):
  """Vocabulary options."""

  T5 = 't5'
  MT5 = 'mt5'
  CUSTOM = 'custom'

  def get_vocab_file(self):
    if self == Vocab.T5:
      vocab_file = 'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model'
    elif self == Vocab.MT5:
      vocab_file = 'gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model'
    else:  # self == Vocab.CUSTOM:
      assert _CUSTOM_VOCAB_FILE is not None, 'Custom vocab file not set.'
      vocab_file = _CUSTOM_VOCAB_FILE
    return vocab_file

  def get_seqio_vocab(self):
    vocab_file = self.get_vocab_file()
    extra_ids = 100 if self == Vocab.CUSTOM else 0
    return seqio.SentencePieceVocabulary(vocab_file, extra_ids=extra_ids)

  def update_gin_bindings(self):
    # Hack some gin bindings since T5X parses gin later.
    length_id = vocab_utils.get_length_id(self.get_seqio_vocab())
    gin.bind_parameter(_LENGTH_ID_BINDING, length_id)
    gin.bind_parameter(_VOCAB_FILE_BINDING, self.get_vocab_file())
    if self == Vocab.T5:
      gin.bind_parameter(_VOCAB_SIZE_BINDING, 32128)
    elif self == Vocab.MT5:
      gin.bind_parameter(_VOCAB_SIZE_BINDING, 250112)
    else:  # self == Vocab.CUSTOM:
      gin.bind_parameter(_VOCAB_SIZE_BINDING, 32128)
      gin.bind_parameter(_EXTRA_IDS_BINDING, 100)
