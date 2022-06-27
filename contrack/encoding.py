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

"""Encode tokens, entity references and predictions as numerical vectors."""

import inspect
import json
import os
import sys
from typing import Any, List, Optional, Text, Tuple, Type, Union

from absl import logging
import numpy as np
import tensorflow as tf

MAX_NUM_ENTITIES = 20

EnrefArray = Union[tf.Tensor, np.ndarray]


class Section(object):
  """Represents a section (i.e. a range) within a data array."""

  def __init__(self, array, start, size):
    self.array = array
    self.start = start
    self.size = size

  def slice(self):
    return self.array[Ellipsis, self.start:(self.start + self.size)]

  def replace(self, array):
    if isinstance(self.array, tf.Tensor):
      self.array = tf.concat([
          self.array[Ellipsis, :self.start], array,
          self.array[Ellipsis, (self.start + self.size):]
      ], -1)
    else:
      self.array[Ellipsis, self.start:(self.start + self.size)] = array
    return self.array


class TypeSection(Section):
  """A section which specifies the encoding type (enref, token, prediction)."""
  SIZE = 3

  def is_token(self):
    return self.array[Ellipsis, self.start + 2]

  def set_token(self):
    self.array[Ellipsis, self.start] = 0.0
    self.array[Ellipsis, self.start + 2] = 1.0

  def is_enref(self):
    return self.array[Ellipsis, self.start]

  def set_enref(self):
    self.array[Ellipsis, self.start] = 1.0
    self.array[Ellipsis, self.start + 2] = 0.0


class EnrefMetaSection(Section):
  """Encodes whether a token is an enref and if its new or new continued."""
  SIZE = 3

  def is_enref(self):
    return self.array[Ellipsis, self.start]

  def set_is_enref(self, value):
    self.array[Ellipsis, self.start] = 1.0 if value else 0.0

  def is_new(self):
    return self.array[Ellipsis, self.start + 1]

  def set_is_new(self, value):
    self.array[Ellipsis, self.start + 1] = 1.0 if value else 0.0

  def is_new_continued(self):
    return self.array[Ellipsis, self.start + 2]

  def set_is_new_continued(self, value):
    self.array[Ellipsis, self.start + 2] = 1.0 if value else 0.0

  def get_is_new_slice(self):
    return self.array[Ellipsis, self.start + 1:self.start + self.size]

  def replace_is_new_slice(self, array):
    self.array = tf.concat([
        self.array[Ellipsis, :self.start + 1], array,
        self.array[Ellipsis, (self.start + self.size):]
    ], -1)
    return self.array


class EnrefIdSection(Section):
  SIZE = MAX_NUM_ENTITIES

  def get(self):
    index = np.argmax(self.slice())
    return index

  def set(self, enref_id):
    self.array[Ellipsis, self.start:(self.start + self.size)] = 0.0
    self.array[Ellipsis, self.start + enref_id] = 1.0


class EnrefPropertiesSection(Section):
  """Encodes the grammatical gender and whether an enref is a group."""
  SIZE = 6
  DOMAINS = ['people', 'locations']
  PROPERTIES = ['female', 'male', 'neuter']

  def get_domain(self):
    array = self.array[Ellipsis, self.start:self.start + 2]
    if np.max(array) <= 0.0:
      return 'unknown'
    index = np.argmax(array)
    return self.DOMAINS[index]

  def set_domain(self, domain):
    self.array[Ellipsis, self.start:(self.start + 2)] = 0.0
    if domain == 'unknown':
      return
    index = self.DOMAINS.index(domain)
    self.array[Ellipsis, self.start + index] = 1.0

  def get_gender(self):
    array = self.array[Ellipsis, (self.start + 2):(self.start + 5)]
    if np.max(array) <= 0.0:
      return 'unknown'
    index = np.argmax(array)
    return self.PROPERTIES[index]

  def set_gender(self, gender):
    self.array[Ellipsis, (self.start + 2):(self.start + 5)] = 0.0
    if gender == 'unknown':
      return
    index = self.PROPERTIES.index(gender)
    self.array[Ellipsis, self.start + 2 + index] = 1.0

  def is_group(self):
    return self.array[Ellipsis, self.start + 5]

  def set_is_group(self, value):
    self.array[Ellipsis, self.start + 5] = 1.0 if value else 0.0


class EnrefMembershipSection(Section):
  """Encodes the members of a group, if an enref refers to multiple entities."""
  SIZE = MAX_NUM_ENTITIES

  def __init__(self, array, start, size):
    Section.__init__(self, array, start, size)
    self.names = None

  def get_ids(self):
    ids = np.where(self.slice() > 0.0)[0].tolist()
    return ids

  def get_names(self):
    return self.names

  def set(self, ids, names = None):
    self.names = names
    self.array[Ellipsis, self.start:(self.start + self.size)] = 0.0
    for enref_id in ids:
      self.array[Ellipsis, self.start + enref_id] = 1.0


class EnrefContextSection(Section):
  """Encodes if an enref is a sender or recipient and the message offset."""
  SIZE = 7

  def is_sender(self):
    return self.array[Ellipsis, self.start]

  def set_is_sender(self, value):
    self.array[Ellipsis, self.start] = 1.0 if value else 0.0

  def is_recipient(self):
    return self.array[Ellipsis, self.start + 1]

  def set_is_recipient(self, value):
    self.array[Ellipsis, self.start + 1] = 1.0 if value else 0.0

  def get_message_offset(self):
    digit = 1
    message_offset = 0
    for i in range(2, self.SIZE):
      message_offset += int(self.array[Ellipsis, self.start + i]) * digit
      digit *= 2
    return message_offset

  def set_message_offset(self, offset):
    for i in range(2, self.SIZE):
      if offset & 0x01:
        self.array[Ellipsis, self.start + i] = 1.0
      else:
        self.array[Ellipsis, self.start + i] = 0.0
      offset = offset >> 1


class TokenPaddingSection(Section):
  """An empty section sized so that enref and token encodings align."""
  SIZE = (
      EnrefIdSection.SIZE + EnrefPropertiesSection.SIZE +
      EnrefMembershipSection.SIZE + EnrefContextSection.SIZE)


class SignalSection(Section):
  """Encodes optional token signals collected during preprocessing."""
  SIZE = 10
  SIGNALS = {
      'first_name': 0,
      'sports_team': 1,
      'athlete': 2,
  }

  def set(self, signals):
    self.array[Ellipsis, self.start:(self.start + self.size)] = 0.0
    for signal in signals:
      index = self.SIGNALS[signal]
      self.array[Ellipsis, self.start + index] = 1.0

  def get(self):
    signals = []
    for index, signal in enumerate(self.SIGNALS):
      if self.array[Ellipsis, self.start + index] > 0.0:
        signals.append(signal)
    return signals


class WordvecSection(Section):
  """Contains the word2vec embedding of a token."""
  SIZE = 300

  def get(self):
    return self.slice()

  def set(self, wordvec):
    self.array[Ellipsis, self.start:(self.start + self.size)] = wordvec


class BertSection(Section):
  """Contains the BERT embedding of a token."""
  SIZE = 768

  def get(self):
    return self.slice()

  def set(self, bertvec):
    self.array[Ellipsis, self.start:(self.start + self.size)] = bertvec


class Encoding(object):
  """Provides an API to access data within an array."""

  def __init__(self, array, layout):
    assert isinstance(array, (np.ndarray, tf.Tensor))

    self.array = array
    index = 0
    for (name, section_class) in layout:
      section = section_class(array, index, section_class.SIZE)
      setattr(self, name, section)
      index += section_class.SIZE

    self.sections_size = index


class EnrefEncoding(Encoding):
  """An API to access and modify contrack entity references within an array."""

  def __init__(self, array, layout):
    Encoding.__init__(self, array, layout)

    self.entity_name = None
    self.word_span = None
    self.span_text = None

  def populate(self, entity_name, word_span,
               span_text):
    self.entity_name = entity_name
    self.word_span = word_span
    self.span_text = span_text

  def __repr__(self):
    descr = ''
    if self.entity_name is not None:
      descr += '%s ' % self.entity_name

    descr += '(%d%s%s) ' % (self.enref_id.get(),
                            'n' if self.enref_meta.is_new() > 0.0 else '', 'c'
                            if self.enref_meta.is_new_continued() > 0.0 else '')
    if self.word_span is not None:
      descr += '%d-%d ' % self.word_span
    if self.span_text is not None:
      descr += '(%s) ' % self.span_text
    if self.enref_properties is not None:
      is_group = self.enref_properties.is_group() > 0.0
      domain = self.enref_properties.get_domain()
      descr += domain[0]
      if domain == 'people' and not is_group:
        descr += ':' + self.enref_properties.get_gender()
      if is_group:
        descr += ':g %s' % self.enref_membership.get_ids()
    if self.signals is not None and self.signals.get():
      descr += str(self.signals.get())
    return descr


class TokenEncoding(Encoding):
  """An API to access and modify contrack tokens within an array."""

  def __init__(self, array, layout):
    Encoding.__init__(self, array, layout)

  def populate(self, token, signals, wordvec,
               bertvec):
    self.token = token
    self.signals.set(signals)
    self.wordvec.set(wordvec)
    self.bert.set(bertvec)

  def __repr__(self):
    signals = self.signals.get()
    signals_str = str(signals) if signals else ''
    return '%s%s' % (self.token, signals_str)


class PredictionEncoding(Encoding):
  """An API to access and modify prediction values stored in an array."""

  def __init__(self, array, layout):
    Encoding.__init__(self, array, layout)

  def __repr__(self):
    descr = '(%d%s%s) ' % (self.enref_id.get(),
                           'n' if self.enref_meta.is_new() > 0.0 else '', 'c'
                           if self.enref_meta.is_new_continued() > 0.0 else '')
    if self.enref_properties is not None:
      is_group = self.enref_properties.is_group() > 0.0
      domain = self.enref_properties.get_domain()
      descr += domain[0]
      if domain == 'people' and not is_group:
        descr += ':' + self.enref_properties.get_gender()
      if is_group:
        descr += ': %s' % self.enref_membership.get_ids()
    return descr


class Encodings(object):
  """Organize access to data encoded in numerical vectors."""

  def __init__(self):
    self.enref_encoding_layout = [('type', TypeSection),
                                  ('enref_meta', EnrefMetaSection),
                                  ('enref_id', EnrefIdSection),
                                  ('enref_properties', EnrefPropertiesSection),
                                  ('enref_membership', EnrefMembershipSection),
                                  ('enref_context', EnrefContextSection),
                                  ('signals', SignalSection),
                                  ('wordvec', WordvecSection),
                                  ('bert', BertSection)]
    self.enref_encoding_length = sum(
        [class_name.SIZE for (_, class_name) in self.enref_encoding_layout])
    logging.info('EnrefEncoding (length: %d): %s', self.enref_encoding_length,
                 [f'{s}: {c.SIZE}' for s, c in self.enref_encoding_layout])

    self.token_encoding_layout = [('type', TypeSection),
                                  ('enref_meta', EnrefMetaSection),
                                  ('padding', TokenPaddingSection),
                                  ('signals', SignalSection),
                                  ('wordvec', WordvecSection),
                                  ('bert', BertSection)]
    self.token_encoding_length = sum(
        [class_name.SIZE for (_, class_name) in self.token_encoding_layout])
    assert self.enref_encoding_length == self.token_encoding_length
    logging.info('TokenEncoding (length: %d): %s', self.token_encoding_length,
                 [f'{s}: {c.SIZE}' for s, c in self.token_encoding_layout])

    self.prediction_encoding_layout = [
        ('enref_meta', EnrefMetaSection),
        ('enref_id', EnrefIdSection),
        ('enref_properties', EnrefPropertiesSection),
        ('enref_membership', EnrefMembershipSection),
    ]
    self.prediction_encoding_length = sum([
        class_name.SIZE for (_, class_name) in self.prediction_encoding_layout
    ])
    logging.info('PredictionEncoding (length: %d): %s',
                 self.prediction_encoding_length,
                 [f'{s}: {c.SIZE}' for s, c in self.prediction_encoding_layout])

  @classmethod
  def load_from_json(cls, path):
    """Loads the encoding layout from a json file."""
    classes = inspect.getmembers(sys.modules[__name__])
    with tf.io.gfile.GFile(path, 'r') as file:
      encodings_dict = json.loads(file.read())

    enc = Encodings()
    enc.enref_encoding_layout = []
    for name, cls_name in encodings_dict['enref_encoding_layout']:
      section_cls = next(o for (n, o) in classes if n.endswith(cls_name))
      enc.enref_encoding_layout.append((name, section_cls))
    enc.enref_encoding_length = sum(
        [class_name.SIZE for (_, class_name) in enc.enref_encoding_layout])

    enc.token_encoding_layout = []
    for name, cls_name in encodings_dict['token_encoding_layout']:
      section_cls = next(o for (n, o) in classes if n.endswith(cls_name))
      enc.token_encoding_layout.append((name, section_cls))
    enc.token_encoding_length = sum(
        [class_name.SIZE for (_, class_name) in enc.token_encoding_layout])
    assert enc.enref_encoding_length == enc.token_encoding_length

    enc.prediction_encoding_layout = []
    for name, cls_name in encodings_dict['prediction_encoding_layout']:
      section_cls = next(o for (n, o) in classes if n.endswith(cls_name))
      enc.prediction_encoding_layout.append((name, section_cls))
    enc.prediction_encoding_length = sum(
        [class_name.SIZE for (_, class_name) in enc.prediction_encoding_layout])

    return enc

  def as_enref_encoding(self, array):
    return EnrefEncoding(array, self.enref_encoding_layout)

  def new_enref_array(self):
    return np.array([0.0] * self.enref_encoding_length)

  def new_enref_encoding(self):
    enc = EnrefEncoding(self.new_enref_array(), self.enref_encoding_layout)
    enc.type.set_enref()
    return enc

  def as_token_encoding(self, array):
    return TokenEncoding(array, self.token_encoding_layout)

  def new_token_array(self):
    return np.array([0.0] * self.token_encoding_length)

  def new_token_encoding(self, token, signals,
                         wordvec, bertvec):
    enc = TokenEncoding(self.new_token_array(), self.token_encoding_layout)
    enc.type.set_token()
    enc.populate(token, signals, wordvec, bertvec)
    return enc

  def as_prediction_encoding(self, array):
    return PredictionEncoding(array, self.prediction_encoding_layout)

  def new_prediction_array(self):
    return np.array([0.0] * self.prediction_encoding_length)

  def new_prediction_encoding(self):
    enc = PredictionEncoding(self.new_prediction_array(),
                             self.prediction_encoding_layout)
    return enc

  def build_enref_from_prediction(
      self, token,
      prediction):
    """Build new enref from prediction logits."""
    if prediction.enref_meta.is_enref() <= 0.0:
      return None

    new_array = np.array(token.array)
    enref = self.as_enref_encoding(new_array)
    enref.type.set_enref()

    enref.enref_meta.replace(
        np.where(prediction.enref_meta.slice() > 0.0, 1.0, 0.0))
    enref.enref_id.set(prediction.enref_id.get())
    enref.enref_properties.replace(
        np.where(prediction.enref_properties.slice() > 0.0, 1.0, 0.0))
    if prediction.enref_properties.is_group() > 0.0:
      enref.enref_membership.replace(
          np.where(prediction.enref_membership.slice() > 0.0, 1.0, 0.0))
    else:
      enref.enref_membership.set([])
    enref.signals.set([])

    return enref

  def build_enrefs_from_predictions(
      self, tokens,
      predictions,
      words,
      prev_enrefs):
    """Build new enrefs from prediction logits."""
    # Identify spans.
    spans = []
    current_span = None
    for i, pred_enc in enumerate(predictions):
      if current_span and (pred_enc.enref_meta.is_enref() <= 0.0 or
                           current_span[1] != pred_enc.enref_id.get()):
        spans.append((current_span[0], i))
        current_span = None
      if not current_span and pred_enc.enref_meta.is_enref() > 0.0:
        current_span = (i, pred_enc.enref_id.get())
    if current_span:
      spans.append((current_span[0], len(predictions)))

    # Create enrefs for spans
    enrefs = []
    for (start, end) in spans:
      enref = self.build_enref_from_prediction(tokens[start],
                                               predictions[start])
      enref.wordvec.set(np.mean([tokens[i].wordvec.get()
                                 for i in range(start, end)], 0))
      enref.bert.set(np.mean([tokens[i].bert.get()
                              for i in range(start, end)], 0))
      span_text = ' '.join([words[i] for i in range(start, end)])

      name = words[start]
      if enref.enref_meta.is_new() <= 0.0:
        for e in prev_enrefs:
          if e.enref_id.get() == enref.enref_id.get():
            name = e.entity_name
            break
      enref.populate(name, (start, end), span_text)
      enrefs.append(enref)

    return enrefs

  def save(self, path):
    """Saves encoding to json file."""
    encodings_dict = {
        'enref_encoding_layout': [
            (n, c.__name__) for (n, c) in self.enref_encoding_layout
        ],
        'token_encoding_layout': [
            (n, c.__name__) for (n, c) in self.token_encoding_layout
        ],
        'prediction_encoding_layout': [
            (n, c.__name__) for (n, c) in self.prediction_encoding_layout
        ],
    }

    filepath = os.path.join(path, 'encodings.json')
    with tf.io.gfile.GFile(filepath, 'w') as file:
      json.dump(encodings_dict, file, indent=2)
