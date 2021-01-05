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

"""Library for generating TF examples for OpenKP data."""

import collections
import json
from typing import Any, List, Mapping, Optional, Sequence, Text

import attr
import tensorflow.compat.v1 as tf

from etcmodel.models import input_utils


@attr.s
class KeyPhrase:
  """Key phrase, a list of words."""
  words = attr.ib(type=List[Text])

  @words.validator
  def validate(self, unused_attribute, value):
    if not isinstance(value, list):
      raise ValueError(
          f'Expected `words` to be a list but got `{type(value)}`.')
    for word in value:
      if not isinstance(word, str):
        raise ValueError(
            f'Expected each `word` to be a str but got `{type(word)}`.')


def _float_to_bool(float_value: float):
  if float_value == 0.0:
    return False
  elif float_value == 1.0:
    return True
  else:
    raise ValueError(f'Expected 0.0 or 1.0 float value but got {float_value}.')


@attr.s
class VdomFeatures:
  """Features for a VDOM element (or its parent)."""

  # Left side of element.
  x_coord = attr.ib(validator=attr.validators.instance_of(float), type=float)

  width = attr.ib(validator=attr.validators.instance_of(float), type=float)

  # Top of element.
  y_coord = attr.ib(validator=attr.validators.instance_of(float), type=float)

  height = attr.ib(validator=attr.validators.instance_of(float), type=float)

  is_block = attr.ib(converter=_float_to_bool, type=bool)

  is_inline = attr.ib(converter=_float_to_bool, type=bool)

  is_heading = attr.ib(converter=_float_to_bool, type=bool)

  is_leaf = attr.ib(converter=_float_to_bool, type=bool)

  font_size = attr.ib(converter=round, type=int)

  is_bold = attr.ib(converter=_float_to_bool, type=bool)

  @classmethod
  def from_floats(cls, floats: Sequence[float]) -> 'VdomFeatures':
    return cls(*floats)


@attr.s
class VdomElement:
  """VDOM element."""
  id = attr.ib(validator=attr.validators.instance_of(int), type=int)
  text = attr.ib(validator=attr.validators.instance_of(str), type=Text)
  features = attr.ib(type=VdomFeatures)
  parent_features = attr.ib(type=VdomFeatures)

  # The start index is inclusive.
  start_idx = attr.ib(validator=attr.validators.instance_of(int), type=int)

  # The end index is exclusive.
  end_idx = attr.ib(validator=attr.validators.instance_of(int), type=int)

  @classmethod
  def from_dict(cls, json_object: Mapping[Text, Any]) -> 'VdomElement':
    # First 10 floats are features for the element itself.
    features = json_object['feature'][:10]
    # Last 10 floats are features for the parent.
    parent_features = json_object['feature'][10:]
    return cls(
        id=json_object['Id'],
        text=json_object['text'],
        features=VdomFeatures.from_floats(features),
        parent_features=VdomFeatures.from_floats(parent_features),
        start_idx=json_object['start_idx'],
        end_idx=json_object['end_idx'])


def _exclusive_vocab(instance, unused_attr, value):
  if not value and not instance.bert_vocab_path:
    raise ValueError(
        'One of `bert_vocab_path`, `spm_model_path` has to be set.')
  if value and instance.bert_vocab_path:
    raise ValueError(
        'Only one of `bert_vocab_path`, `spm_model_path` can be set.')


@attr.s(auto_attribs=True)
class EtcFeaturizationConfig:
  """Config options for converting `OpenKpExample` to `OpenKpEtcFeatures`."""

  # Maximum number of long tokens.
  long_max_length: int

  # Maximum number of global tokens.
  global_max_length: int

  # Maximum number of Unicode code points in any example URL.
  url_max_code_points: int

  # Path to the BERT vocabulary file to use.
  bert_vocab_path: Text

  # Whether to lower case text for wordpiece tokenization. Ignored when
  # sentencepiece is used. This is also ignored for finding the keyphrase
  # occurrences, which are always searched in lower case.
  do_lower_case: bool

  # Path to a sentencepiece model file to use instead of a BERT vocabulary file.
  # If given, we use the tokenization code from ALBERT instead of BERT.
  # `bert_vocab_path` must be set to an empty string if `spm_model_path` is not
  # None.
  spm_model_path: Optional[Text] = attr.ib(
      default=None, validator=_exclusive_vocab)

  # If set, then the VDOM structure is discarded, and 1 global token is created
  # per fixed_block_len long tokens. Also, no visual features are created.
  fixed_block_len: Optional[int] = None

  # Integer maximum number of words in a key phrase.
  kp_max_length: int = 5


@attr.s(auto_attribs=True)
class OpenKpExample:
  """OpenKP example structure."""
  url: Text
  text: Text
  vdom: List[VdomElement]

  # This is the label, which won't be available for evaluation examples.
  key_phrases: Optional[List[KeyPhrase]] = None

  @classmethod
  def from_json(cls, json_str: Text) -> 'OpenKpExample':
    json_obj = json.loads(json_str)
    vdom_obj = json.loads(json_obj['VDOM'])
    if 'KeyPhrases' in json_obj:
      key_phrases = [KeyPhrase(x) for x in json_obj['KeyPhrases']]
    else:
      key_phrases = None
    return cls(
        url=json_obj['url'],
        text=json_obj['text'],
        vdom=[VdomElement.from_dict(x) for x in vdom_obj],
        key_phrases=key_phrases)

  def to_etc_features(self, tokenizer,
                      config: EtcFeaturizationConfig) -> 'OpenKpEtcFeatures':
    """Generates ETC features for the example.

    Examples with wordpiece tokens beyond `config.long_max_length` or
    VDOM elements beyond `config.global_max_length` will be truncated.

    Args:
      tokenizer: The tokenizer (e.g., BERT tokenizer) to be used to get the
        wordpieces of a sentence. This tokenizer should support a `tokenize()`
        method that should return a list of tokens.
      config: `EtcFeaturizationConfig` object.

    Returns:
      An `OpenKpEtcFeatures` object.

    Raises:
      ValueError: Too many url code points in url.
    """
    etc_features = OpenKpEtcFeatures()
    self._populate_url_feature(etc_features, config)
    if config.fixed_block_len:
      self._populate_etc_nonvisual_features_fixed(etc_features, tokenizer,
                                                  config)
    else:
      self._populate_etc_nonvisual_features_vdom(etc_features, tokenizer,
                                                 config)
      self._populate_etc_visual_features(etc_features, config)
    self._pad_etc_features(etc_features, config)
    return etc_features

  def _populate_url_feature(self, etc_features: 'OpenKpEtcFeatures',
                            config: EtcFeaturizationConfig) -> None:
    """Populates and pads the `url_code_points` feature."""
    url_code_points = [ord(x) for x in self.url]
    if len(url_code_points) > config.url_max_code_points:
      raise ValueError(f'Too many url code points in url: {self.url}')
    padding_len = config.url_max_code_points - len(url_code_points)
    url_code_points.extend([-1] * padding_len)
    etc_features.url_code_points = url_code_points

  def _populate_etc_nonvisual_features_vdom(
      self, etc_features: 'OpenKpEtcFeatures', tokenizer,
      config: EtcFeaturizationConfig) -> None:
    """Populate the non-visual features of `etc_features`.

    This function populates the long and global masks, the keyphrase labels,
    and the token/index features, but not the visual features described by the
    VDOM like position and font properties.

    1 VDOM element is mapped to 1 global token.

    Args:
      etc_features: `OpenKpEtcFeatures` to populate.
      tokenizer: The tokenizer (e.g., BERT tokenizer) to be used to get the
        wordpieces of a sentence. This tokenizer should support a `tokenize()`
        method that should return a list of tokens.
      config: `EtcFeaturizationConfig` object.
    """
    vdom_max_len = config.global_max_length
    long_max_len = config.long_max_length

    etc_features.long_token_ids = []
    etc_features.long_word_idx = []
    etc_features.long_vdom_idx = []
    etc_features.global_token_ids = [1] * vdom_max_len

    if self.key_phrases:
      # Note that we search for the first occurrence of the keyphrase in the
      # text irespectively of upper/lower case, while config.do_lower_case
      # controls the language model only. The padding spaces are added to match
      # complete words only, not substrings.
      all_key_text = set(
          ' ' + ' '.join(x.words).lower() + ' ' for x in self.key_phrases)  # pylint: disable=not-an-iterable
    else:
      all_key_text = set()
    all_key_text = sorted(all_key_text)  # For deterministic tests.
    missing_key_text = set(all_key_text)

    long_tokens_count = 0
    word_idx = 0
    all_phrase_occurrences = {}
    first_occurrences_k = {k: [] for k in range(1, 1 + config.kp_max_length)}
    for vdom_idx, vdom_element in enumerate(self.vdom):
      vdom_word_idx = word_idx

      new_text_tokens_ids = []
      new_long_word_idx = []
      text_words = vdom_element.text.split(' ')
      for text_word in text_words:
        text_token = tokenizer.tokenize(text_word)
        long_token_id = tokenizer.convert_tokens_to_ids(text_token)
        new_text_tokens_ids.extend(long_token_id)
        new_long_word_idx.extend([word_idx] * len(long_token_id))
        word_idx += 1

      new_long_tokens_count = len(new_text_tokens_ids)
      long_tokens_count += new_long_tokens_count
      if vdom_idx >= vdom_max_len or long_tokens_count > long_max_len:
        break

      etc_features.long_vdom_idx.extend([vdom_idx] * new_long_tokens_count)
      etc_features.long_input_mask.extend([1] * new_long_tokens_count)
      # Note: len(text_words) can be larger than new_long_tokens_count since
      # there are trivial inputs such as `text_words=['']` that have no tokens
      # but contribute to the number of words.
      etc_features.long_word_input_mask.extend([1] * len(text_words))
      etc_features.global_input_mask.append(1)
      etc_features.long_token_ids.extend(new_text_tokens_ids)
      etc_features.long_word_idx.extend(new_long_word_idx)

      # Note that we search for the first occurrence of the keyphrase in the
      # text irespectively of upper/lower case, while config.do_lower_case
      # controls the language model only. The padding spaces are added to match
      # complete words only, not substrings.
      search_text = ' ' + vdom_element.text.lower() + ' '

      # Find keyphrases.
      for key_text in all_key_text:
        if key_text in missing_key_text and key_text in search_text:
          char_offset = search_text.index(key_text)
          word_offset = search_text[:char_offset].count(' ')
          etc_features.label_start_idx.append(vdom_word_idx + word_offset)
          # Subtract 2 for the padding spaces.
          etc_features.label_phrase_len.append(1 - 2 + key_text.count(' '))
          missing_key_text.remove(key_text)

      # Find multiple occurrences of all phrases, but only those that respect
      # VDOM boundaries.
      search_words = vdom_element.text.lower().split(' ')
      for k in range(1, 1 + config.kp_max_length):
        for i in range(len(search_words)):
          if i + k > len(search_words):
            first_occurrences_k[k].append(vdom_word_idx + i)
            continue
          # Note that some search_words[i] may be empty.
          search_phrase = ' '.join(search_words[i:i + k])
          if search_phrase in all_phrase_occurrences:
            first_occurrences_k[k].append(all_phrase_occurrences[search_phrase])
          else:
            word_offset = vdom_word_idx + i
            first_occurrences_k[k].append(word_offset)
            all_phrase_occurrences[search_phrase] = word_offset

    assert self.key_phrases is None or len(missing_key_text) < len(all_key_text)

    num_words = len(etc_features.long_word_input_mask)
    assert num_words <= config.long_max_length, (
        f'Skipping example {self.url} because number of words {num_words} '
        f'is more than long_max_length {config.long_max_length}.')
    for k in range(1, 1 + config.kp_max_length):
      assert len(first_occurrences_k[k]) == num_words
      # Pad.
      first_occurrences_k[k].extend(range(num_words, config.long_max_length))
      etc_features.long_word_first_occurrence.extend(first_occurrences_k[k])

  def _populate_etc_nonvisual_features_fixed(
      self, etc_features: 'OpenKpEtcFeatures', tokenizer,
      config: EtcFeaturizationConfig) -> None:
    """Populate the non-visual features of `etc_features`.

    This function populates the long and global masks, the keyphrase labels,
    and the token/index features, but not the visual features described by the
    VDOM like position and font properties.

    1 global token is created for every config.fixed_block_len long tokens.

    Args:
      etc_features: `OpenKpEtcFeatures` to populate.
      tokenizer: The tokenizer (e.g., BERT tokenizer) to be used to get the
        wordpieces of a sentence. This tokenizer should support a `tokenize()`
        method that should return a list of tokens.
      config: `EtcFeaturizationConfig` object.
    """
    global_max_len = config.global_max_length
    long_max_len = config.long_max_length

    etc_features.long_token_ids = []
    etc_features.long_word_idx = []
    etc_features.long_vdom_idx = []
    etc_features.global_token_ids = [1] * global_max_len

    for word_idx, text_word in enumerate(self.text.split(' ')):
      text_token = tokenizer.tokenize(text_word)
      long_token_ids = tokenizer.convert_tokens_to_ids(text_token)
      long_new_len = len(long_token_ids)

      long_total_len = len(etc_features.long_token_ids) + long_new_len
      if long_total_len > long_max_len:
        break
      global_total_len = long_total_len // config.fixed_block_len
      if long_total_len % config.fixed_block_len != 0:
        global_total_len += 1
      if global_total_len > global_max_len:
        break

      for i in range(len(etc_features.long_token_ids), long_total_len):
        etc_features.long_vdom_idx.append(i // config.fixed_block_len)
      etc_features.long_input_mask.extend([1] * long_new_len)
      # Note: len(text_words) can be larger than new_long_tokens_count since
      # there are trivial inputs such as `text_words=['']` that have no tokens
      # but contribute to the number of words.
      etc_features.long_word_input_mask.append(1)

      etc_features.long_token_ids.extend(long_token_ids)
      etc_features.long_word_idx.extend([word_idx] * long_new_len)

    assert etc_features.long_vdom_idx
    num_globals = etc_features.long_vdom_idx[-1] + 1
    etc_features.global_input_mask = [1] * num_globals

    assert etc_features.long_word_idx
    num_words = etc_features.long_word_idx[-1] + 1

    if self.key_phrases:
      # Note that we search for the first occurrence of the keyphrase in the
      # text irespectively of upper/lower case, while config.do_lower_case
      # controls the language model only.
      all_key_text = set(' '.join(x.words).lower() for x in self.key_phrases)  # pylint: disable=not-an-iterable
      all_key_text = sorted(all_key_text)  # For deterministic tests.
      missing_key_text = set(all_key_text)

      search_text = ' '.join(self.text.lower().split(' ')[:num_words])

      for key_text in all_key_text:
        if key_text in missing_key_text and key_text in search_text:
          char_offset = search_text.index(key_text)
          word_offset = search_text[:char_offset].count(' ')
          etc_features.label_start_idx.append(word_offset)
          etc_features.label_phrase_len.append(1 + key_text.count(' '))
          missing_key_text.remove(key_text)

      assert len(missing_key_text) < len(all_key_text)

  def _populate_etc_visual_features(self, etc_features: 'OpenKpEtcFeatures',
                                    config: EtcFeaturizationConfig) -> None:
    """Populate the visual features of `etc_features`.

    This function populates only the visual features described by the VDOM,
    like position and font properties. The non-visual features must be
    populated before calling this function.

    Args:
      etc_features: `OpenKpEtcFeatures` to populate.
      config: `EtcFeaturizationConfig` object.
    """
    del config  # Unused.
    assert etc_features.global_input_mask
    vdom_count = sum(etc_features.global_input_mask)

    for vdom_element in self.vdom[:vdom_count]:
      features = vdom_element.features
      etc_features.global_x_coords.append(features.x_coord)
      etc_features.global_y_coords.append(features.y_coord)
      etc_features.global_widths.append(features.width)
      etc_features.global_heights.append(features.height)
      global_font_id = font_size_to_font_id(features.font_size)
      etc_features.global_font_ids.append(global_font_id)
      etc_features.global_block_indicator.append(int(features.is_block))
      etc_features.global_inline_indicator.append(int(features.is_inline))
      etc_features.global_heading_indicator.append(int(features.is_heading))
      etc_features.global_leaf_indicator.append(int(features.is_leaf))
      etc_features.global_bold_indicator.append(int(features.is_bold))

      parent_features = vdom_element.parent_features
      etc_features.global_parent_x_coords.append(parent_features.x_coord)
      etc_features.global_parent_y_coords.append(parent_features.y_coord)
      etc_features.global_parent_widths.append(parent_features.width)
      etc_features.global_parent_heights.append(parent_features.height)
      global_parent_font_id = font_size_to_font_id(parent_features.font_size)
      etc_features.global_parent_font_ids.append(global_parent_font_id)
      etc_features.global_parent_heading_indicator.append(
          int(parent_features.is_heading))
      etc_features.global_parent_leaf_indicator.append(
          int(parent_features.is_leaf))
      etc_features.global_parent_bold_indicator.append(
          int(parent_features.is_bold))

  def _pad_etc_features(self, etc_features: 'OpenKpEtcFeatures',
                        config: EtcFeaturizationConfig) -> None:
    """Pad all features of `etc_features` (long, global and key_phrase).

    Args:
      etc_features: `OpenKpEtcFeatures` to populate.
      config: `EtcFeaturizationConfig` object.
    """
    long_count = len(etc_features.long_input_mask)
    long_padding = config.long_max_length - long_count
    long_features = (
        etc_features.long_input_mask,
        etc_features.long_token_ids,
        etc_features.long_word_idx,
        etc_features.long_vdom_idx,
    )
    for long_feature in long_features:
      long_feature.extend([0] * long_padding)
      assert len(long_feature) == config.long_max_length, (
          f'{long_feature} has wrong length ({len(long_feature)}, expected '
          f'{config.long_max_length})')
    etc_features.long_word_input_mask.extend(
        [0] * (config.long_max_length - len(etc_features.long_word_input_mask)))
    len_word_mask = len(etc_features.long_word_input_mask)
    assert len_word_mask == config.long_max_length, (
        f'Skipping example {self.url} because number of words {len_word_mask} '
        f'is more than long_max_length {config.long_max_length}.')
    if not config.fixed_block_len:
      assert len(etc_features.long_word_first_occurrence) == (
          config.long_max_length * config.kp_max_length)

    vdom_count = len(etc_features.global_input_mask)
    global_padding = config.global_max_length - vdom_count
    global_features = (etc_features.global_input_mask,)
    visual_features = (
        etc_features.global_x_coords,
        etc_features.global_y_coords,
        etc_features.global_widths,
        etc_features.global_heights,
        etc_features.global_font_ids,
        etc_features.global_block_indicator,
        etc_features.global_inline_indicator,
        etc_features.global_heading_indicator,
        etc_features.global_leaf_indicator,
        etc_features.global_bold_indicator,
        etc_features.global_parent_x_coords,
        etc_features.global_parent_y_coords,
        etc_features.global_parent_widths,
        etc_features.global_parent_heights,
        etc_features.global_parent_font_ids,
        etc_features.global_parent_heading_indicator,
        etc_features.global_parent_leaf_indicator,
        etc_features.global_parent_bold_indicator,
    )
    for global_feature in global_features:
      global_feature.extend([0] * global_padding)
      assert len(global_feature) == config.global_max_length
    if not config.fixed_block_len:
      for global_feature in visual_features:
        global_feature.extend([0] * global_padding)
        assert len(global_feature) == config.global_max_length

    key_phrase_padding = 3 - len(etc_features.label_start_idx)
    etc_features.label_start_idx.extend([-1] * key_phrase_padding)
    assert 3 == len(etc_features.label_start_idx)
    etc_features.label_phrase_len.extend([-1] * key_phrase_padding)
    assert 3 == len(etc_features.label_phrase_len)


@attr.s
class OpenKpEtcFeatures:
  """Tensor features to represent an OpenKP example for ETC.

  All `long_` features have the same long input length (e.g. 4096),
  and all `global_` features have the same global input length (e.g. 512).
  The `label_` features have length 3 since there are up to 3 answer key
  phrases.
  """

  # Unicode code points for the example URL, padded with `-1` values to the
  # right. The URL serves as the example id.
  url_code_points = attr.ib(factory=list, type=List[int])

  # Index of the first *word* in each correct key phrase (up to 3). `-1`
  # values are used for padding if there are fewer than 3 key phrases.
  # For now we just pick the 1st occurrence of each key phrase (ignoring case)
  # if there are multiple occurrences in the text.
  label_start_idx = attr.ib(factory=list, type=List[int])

  # Length of the key phrase (up to 3) corresponding to the start index from
  # `label_start_idx` above. `-1` values are used for padding if there are
  # fewer than 3 key phrases.
  label_phrase_len = attr.ib(factory=list, type=List[int])

  # Wordpiece ids for the text.
  long_token_ids = attr.ib(factory=list, type=List[int])

  # Index of the word each wordpiece belongs to. The words result from splitting
  # the text by ' ' and use 0-based indices.
  long_word_idx = attr.ib(factory=list, type=List[int])

  # Index of the global token (VDOM element) each wordpiece belongs to.
  long_vdom_idx = attr.ib(factory=list, type=List[int])

  # Mask of long input tokens, with `1` for actual tokens and `0` for padding.
  long_input_mask = attr.ib(factory=list, type=List[int])

  # Auxiliary mask with the same numbers of `1` as the number of words,
  # [1] * num_words + [0] * padding.
  long_word_input_mask = attr.ib(factory=list, type=List[int])

  # For all phrases in the input text (of length 1 to kp_max_length words) store
  # the positions in text where they occur for the first time. Still don't allow
  # for phrases to cross VDOM boundaries. The list is flattened into shape
  # [kp_max_length * long_max_length].
  long_word_first_occurrence = attr.ib(factory=list, type=List[int])

  # Wordpiece ids for the vdom elements. This shares the same vocabulary as
  # `long_token_ids`, and for now we just populate it as all `1`s, just like
  # pretraining (`1` was used for cased/uncased wordpiece vocabs and also for
  # sentencepiece vocab).
  global_token_ids = attr.ib(factory=list, type=List[int])

  # Mask of global input tokens, with `1` for actual tokens and `0` for padding.
  global_input_mask = attr.ib(factory=list, type=List[int])

  # X coordinate of VDOM elements.
  global_x_coords = attr.ib(factory=list, type=List[float])

  # Y coordinate of VDOM elements.
  global_y_coords = attr.ib(factory=list, type=List[float])

  # Width of VDOM elements.
  global_widths = attr.ib(factory=list, type=List[float])

  # Height of VDOM elements.
  global_heights = attr.ib(factory=list, type=List[float])

  # Font size id of VDOM elements. Note this is *not* the raw integer font size.
  # It's instead an id associated with a font size grouping, with valid values
  # from 0 to `FONT_ID_VOCAB_SIZE` (exclusive).
  global_font_ids = attr.ib(factory=list, type=List[int])

  # "Boolean" indicators (`1` for True and `0` for False) representing whether
  # the VDOM elements are blocks.
  global_block_indicator = attr.ib(factory=list, type=List[int])

  # "Boolean" indicators (`1` for True and `0` for False) representing whether
  # the VDOM elements are inline.
  global_inline_indicator = attr.ib(factory=list, type=List[int])

  # "Boolean" indicators (`1` for True and `0` for False) representing whether
  # the VDOM elements are headings.
  global_heading_indicator = attr.ib(factory=list, type=List[int])

  # "Boolean" indicators (`1` for True and `0` for False) representing whether
  # the VDOM elements are leaves.
  global_leaf_indicator = attr.ib(factory=list, type=List[int])

  # "Boolean" indicators (`1` for True and `0` for False) representing whether
  # the VDOM elements are bold.
  global_bold_indicator = attr.ib(factory=list, type=List[int])

  # X coordinate of VDOM parent elements.
  global_parent_x_coords = attr.ib(factory=list, type=List[float])

  # Y coordinate of VDOM parent elements.
  global_parent_y_coords = attr.ib(factory=list, type=List[float])

  # Width of VDOM parent elements.
  global_parent_widths = attr.ib(factory=list, type=List[float])

  # Height of VDOM parent elements.
  global_parent_heights = attr.ib(factory=list, type=List[float])

  # Font size id of VDOM parent elements. Note this is *not* the raw integer
  # font size. It's instead an id associated with a font size grouping, with
  # valid values from 0 to `FONT_ID_VOCAB_SIZE` (exclusive).
  global_parent_font_ids = attr.ib(factory=list, type=List[int])

  # Notes: Parent VDOM element always have `block=True` and `inline=False`,
  # so we omit these features.

  # "Boolean" indicators (`1` for True and `0` for False) representing whether
  # the VDOM parent elements are headings.
  global_parent_heading_indicator = attr.ib(factory=list, type=List[int])

  # "Boolean" indicators (`1` for True and `0` for False) representing whether
  # the VDOM parent elements are leaves.
  global_parent_leaf_indicator = attr.ib(factory=list, type=List[int])

  # "Boolean" indicators (`1` for True and `0` for False) representing whether
  # the VDOM parent elements are bold.
  global_parent_bold_indicator = attr.ib(factory=list, type=List[int])

  def to_tf_example(self) -> tf.train.Example:
    """Returns a TF Example."""
    # All features are int features except for these float features.
    float_features = {
        'global_x_coords', 'global_y_coords', 'global_widths', 'global_heights',
        'global_parent_x_coords', 'global_parent_y_coords',
        'global_parent_widths', 'global_parent_heights'
    }

    fields = attr.asdict(self)
    assert all(x in fields for x in float_features)

    features = collections.OrderedDict()
    for name in attr.fields_dict(OpenKpEtcFeatures).keys():
      values = getattr(self, name)
      if name in float_features:
        features[name] = input_utils.create_float_feature(values)
      else:
        features[name] = input_utils.create_int_feature(values)
    return tf.train.Example(features=tf.train.Features(feature=features))


# Cutoffs must be in increase order, and each defines a maximum font value for
# a font id. Values above the last cutoff get a final distinct id.
_FONT_SIZE_CUTOFFS = [0] + list(range(8, 25)) + [29] + [34] + [39] + [44] + [49]

FONT_ID_VOCAB_SIZE = len(_FONT_SIZE_CUTOFFS) + 1


def font_size_to_font_id(font_size: int) -> int:
  """Returns the font id for a given font size.

  Args:
    font_size: Integer font size.

  Returns:
    An id between 0 (inclusive) and `FONT_ID_VOCAB_SIZE` (exclusive).
  """
  for idx, cutoff in enumerate(_FONT_SIZE_CUTOFFS):
    if font_size <= cutoff:
      return idx

  assert font_size > _FONT_SIZE_CUTOFFS[-1]
  return len(_FONT_SIZE_CUTOFFS)
