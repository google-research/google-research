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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Single-residue encoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy


class ResidueIdentityEncoder(object):
  """Residue identity (one-hot) encoder.

  Attributes:
    encoding_size: (int) The number of encoding dimensions for a single residue.
  """

  def __init__(self, alphabet):
    """Constructor.

    Args:
      alphabet: (seq<char>) The alphabet of valid tokens for the sequence;
        e.g., the 20x 1-letter residue codes for standard peptides.
    """
    self._alphabet = [l.upper() for l in alphabet]
    self._letter_to_id = dict((letter, id) for (id, letter)
                              in enumerate(self._alphabet))
    self.encoding_size = len(self._alphabet)

  def encode(self, residue):
    """Encodes a single residue as a one hot identity vector.

    Args:
      residue: (str) A single-character string representing one residue; e.g.,
        'A' for Alanine.
    Returns:
      A numpy.ndarray(shape=(A,), dtype=float) with a single non-zero (1) value;
      the identity index for each residue in the alphabet is given by the
      residue's index in the alphabet ordered sequence; i.e., for the alphabet
      'ACDE', a 'C' would be encoded as [0, 1, 0, 0].
    """
    onehot = numpy.zeros(self.encoding_size, dtype=float)
    onehot[self._letter_to_id[residue]] = 1
    return onehot
