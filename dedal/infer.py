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

"""Run inference of the alignment and provides a wrapper."""

import functools
from typing import Any, Tuple

import numpy as np
import tensorflow as tf

from dedal import alignment
from dedal import vocabulary
from dedal.data import transforms


def preprocess(left, right, max_length = 512):
  """Prepares the data to be fed to the DEDAL network."""
  seqs = {'left': left, 'right': right}
  seqs = {k: v.strip().upper() for k, v in seqs.items()}
  keys = list(seqs.keys())
  transformations = [
      transforms.Encode(vocab=vocabulary.alternative, on=keys),
      transforms.EOS(vocab=vocabulary.alternative, on=keys),
      transforms.CropOrPad(
          size=max_length, vocab=vocabulary.alternative, on=keys)]
  for t in transformations:
    seqs = t(seqs)
  return tf.stack([seqs['left'], seqs['right']], axis=0)


def postprocess(output, length_1, length_2):
  """Post process the output of the inferred alignment."""
  score, paths, sw_params = tf.nest.map_structure(
      functools.partial(tf.squeeze, axis=0), output)
  # Stacks SW params, flipping sign of gap penalties for convenience.
  substitution_scores, gap_open, gap_extend = sw_params
  gap_open = tf.broadcast_to(-gap_open, tf.shape(substitution_scores))
  gap_extend = tf.broadcast_to(-gap_extend, tf.shape(substitution_scores))
  sw_params = tf.stack([substitution_scores, gap_open, gap_extend], axis=-1)
  # Discards padding.
  paths, sw_params = tf.nest.map_structure(
      lambda t: t[:length_1, :length_2], (paths, sw_params))
  # Converts alignment output to 3-state representation.
  states = tf.stack([alignment.paths_to_state_indicators(paths, s)
                     for s in ('match', 'gap_open', 'gap_extend')], axis=-1)
  return score, states, sw_params


def expand(inputs):
  """Expands a flat dict based on the key structure.

  The output of a model might be in the form {'output_4_1_1: tensor, ...} which
  means that the tensor is nest inside a tuple of depth 3, on the 4th position
  at the first level, the first one at the second etc. The goal of this function
  is to build back the tuple from the dict.

  Args:
    inputs: The inputs to be expanded. Only acts if it is a dictionary,
      otherwise this is a no-op.

  Returns:
    A tuple which structure matches the one of the keys of the dict.
  """
  if not isinstance(inputs, dict):
    return inputs

  expansion = dict()
  for k, v in inputs.items():
    p = k.split('_')[1:]
    pos = int(p[0])
    if len(p) == 1:
      expansion[pos] = v
    else:
      if pos not in expansion:
        expansion[pos] = {}
      expansion[pos][f'output_{"_".join(p[1:])}'] = v

  res = {}
  for k, v in expansion.items():
    res[k] = expand(v)
  return tuple([v for k, v in sorted(res.items())])


class Alignment:
  """Represents and manipulates alignments."""

  def __init__(self, left, right, scores, path, sw_params):
    self.left = left
    self.right = right
    self.scores = scores.numpy()
    self.path = path.numpy()
    self.sw_params = sw_params.numpy() if sw_params is not None else None

    self.start = None
    self.end = None
    self.left_match = None
    self.right_match = None
    self.matches = None
    self.expand()

  def _position_to_char(self, i, j, s):
    if s == 0:  # Match
      if self.left[i] == self.right[j]:
        return '|'
      elif self.sw_params is not None and self.sw_params[i, j, s] > 0:
        return ':'
      else:
        return '.'
    return ' '

  def expand(self):
    """String representation of an alignment."""
    indices = tf.where(self.path).numpy()
    indices = indices[np.argsort(indices[:, 0] + indices[:, 1])]
    start = indices[0, :2]
    end = indices[-1, :2]

    # Summary matches in terms of exact matches and, if available, substitution
    # scores.
    a_x, a_y = [], []
    summary = []
    for t, (i, j, s) in enumerate(indices):
      summary.append(self._position_to_char(i, j, s))
      if s == 0:  # Match.
        a_x.append(self.left[i])
        a_y.append(self.right[j])
      else:  # Gap
        i_prev, j_prev = indices[t - 1][:2]
        if (i - i_prev) == 1 and (j - j_prev) == 0:  # Gap in Y.
          a_x.append(self.left[i])
          a_y.append('-')
        elif (i - i_prev) == 0 and (j - j_prev) == 1:  # Gap in X.
          a_x.append('-')
          a_y.append(self.right[j])
        else:  # Incorrectly formatted alignment.
          raise ValueError('Alignment is inconsistent.')

    self.start = start
    self.end = end
    self.left_match = ''.join(a_x)
    self.right_match = ''.join(a_y)
    self.matches = ''.join(summary)

  def __len__(self):
    return len(self.matches)

  @property
  def identity(self):
    return self.matches.count('|')

  @property
  def similarity(self):
    return self.identity + self.matches.count(':')

  @property
  def gaps(self):
    return sum([x == '-' or y == '-'
                for x, y in zip(self.left_match, self.right_match)])

  def __str__(self):
    start = [str(a).rjust(4).ljust(6) for a in self.start]
    end = [str(a).rjust(4).ljust(6) for a in self.end]
    left_match = start[0] + ''.join(self.left_match) + end[0]
    right_match = start[1] + ''.join(self.right_match) + end[1]
    links = ' ' * 6 + ''.join(self.matches) + ' ' * 6
    return '\n'.join([left_match, links, right_match])


def align(model,
          left,
          right,
          max_length = 512):
  """Aligns the left and right proteins with a loaded models.

  Args:
    model: This can be a loaded tf.saved_model or loaded from tfhub.
    left: a protein sequence as string.
    right: a protein sequence as string.
    max_length: the expected length of proteins the model was trained for.

  Returns:
    An alignment object. Which contains the scores, path, Smith Waterman
    parameters as well as positions of the alignments in the sequence.
  """
  inputs = preprocess(left, right, max_length)
  output = model(inputs)
  output = expand(output)
  scores, path, params = postprocess(output, len(left), len(right))
  return Alignment(left, right, scores, path, params)
