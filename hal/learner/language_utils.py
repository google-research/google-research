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

"""Utilities for language."""
# pylint: disable=not-an-iterable

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import re

import numpy as np


_RELATION_SYNONYMS = {
    'on the left side': ['left', 'on the left'],
    'on the right side': ['right', 'on the right'],
    'in front of': ['front of']
}
_MATERIAL_SYNONYMS = {
    'matte': ['rubber', ''],
    'rubber': ['matte', ''],
    'shiny': ['metallic', ''],
    'metallic': ['shiny', '']
}
_OBJECT_SYNONYMS = {
    'object': ['sphere', 'object', 'thing'],
    'sphere': ['object', 'ball', 'thing'],
    'ball': ['sphere', 'object', 'thing'],
    'objects': ['spheres', 'objects', 'things'],
    'spheres': ['objects', 'balls', 'things'],
    'balls': ['spheres', 'objects', 'things']
}
_ADJECTIVE_SYNONYMS = {'any': ['']}
_MISC_SYNONYMS = {'are': ['is']}

_CLEVR_SYNONYM_TABLES = [
    _RELATION_SYNONYMS, _MATERIAL_SYNONYMS, _OBJECT_SYNONYMS,
    _ADJECTIVE_SYNONYMS, _MISC_SYNONYMS
]

_COLORS = [
    {
        'red': ['']
    },
    {
        'blue': ['']
    },
    {
        'cyan': ['']
    },
    {
        'purple': ['']
    },
    {
        'green': ['']
    },
]

_OTHER_COLORS = {
    'red': ['blue', 'cyan', 'purple', 'green'],
    'blue': ['red', 'cyan', 'purple', 'green'],
    'cyan': ['blue', 'red', 'purple', 'green'],
    'purple': ['blue', 'cyan', 'red', 'green'],
    'green': ['blue', 'cyan', 'purple', 'red'],
}

_OTHER_DIRECTIONS = {
    'left': ['right'],
    'right': ['left'],
    'front': ['behind'],
    'behind': ['front'],
}


def get_vocab_path(cfg):
  """Get path to the list of vocabularies."""
  vocab_path = None
  if not vocab_path:
    vocab_path = cfg.vocab_path
  return vocab_path


def instruction_type(instruction):
  if len(instruction) < 40:
    return 'unary'
  else:
    return 'regular'


def pad_to_max_length(data, max_l=None, eos_token=0):
  """Pad a list of sequence to the maximum length."""
  eos = eos_token
  if not max_l:
    max_l = -1
    for p in data:
      max_l = max(max_l, len(p))
  data_padded = []
  for p in data:
    if len(p) == max_l:
      data_padded.append(list(p))
    else:
      p = list(p) + [eos] * (max_l - len(p))
      data_padded.append(p)
  return np.array(data_padded)


def pad_sequence(data, max_l=None, eos_token=0):
  """Pad a sequence to max_l with eos_token."""
  eos = eos_token
  if len(data) == max_l:
    return np.array(data)
  elif len(data) > max_l:
    raise ValueError('data longer than max_l')
  else:
    data = list(data) + [eos] * (max_l - len(data))
    return np.array(data)


def paraphrase_sentence(text, synonym_tables=None, delete_color=False, k=2):
  """Paraphrase a sentence.

  Args:
    text: text to be paraphrased
    synonym_tables: a table that contains synonyms for all the words
    delete_color: whether to delete colors from sentences
    k: number of words to replace

  Returns:
    paraphrased text
  """
  if not synonym_tables:
    synonym_tables = _CLEVR_SYNONYM_TABLES
  tables = random.sample(synonym_tables, k)
  if delete_color and random.uniform(0, 1) < 0.5:
    tables = random.sample(_COLORS, 5)
    subed = False
    for t in tables:
      if subed:
        break
      for w in t:
        if w in text:
          text = re.sub(w, random.choice(t[w]), text)
          subed = True
  else:
    for t in tables:
      for w in t:
        if w in text:
          text = re.sub(w, random.choice(t[w]), text)
  return text


def negate_unary_sentence(text):
  """Negate a instruction involving a single object."""
  words = text.split(' ')
  mutate_candiate = {}
  for i, w in enumerate(words):
    if w in _OTHER_COLORS:
      mutate_candiate['color'] = (i, w)
    elif w in _OTHER_DIRECTIONS:
      mutate_candiate['direction'] = (i, w)
  toss = random.random()
  if toss < 0.33 and 'color' in mutate_candiate:
    i, color = mutate_candiate['color']
    new_color = random.choice(_OTHER_COLORS[color])
    words[i] = new_color
  elif 0.33 < random.random() < 0.66 and 'direction' in mutate_candiate:
    i, direction = mutate_candiate['direction']
    new_direction = random.choice(_OTHER_DIRECTIONS[direction])
    words[i] = new_direction
  elif 'direction' in mutate_candiate and 'color' in mutate_candiate:
    i, color = mutate_candiate['color']
    new_color = random.choice(_OTHER_COLORS[color])
    words[i] = new_color
    i, direction = mutate_candiate['direction']
    new_direction = random.choice(_OTHER_DIRECTIONS[direction])
    words[i] = new_direction
  else:
    return None
  mutated_text = ' '.join(words)
  return mutated_text
