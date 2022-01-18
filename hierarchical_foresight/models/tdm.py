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

"""Temporal Distance Model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow.compat.v1 as tf


class TemporalModel(snt.AbstractModule):
  """Temporal Distance Model."""

  def __init__(self, name='itm'):
    super(TemporalModel, self).__init__(name=name)
    self.enc = snt.nets.ConvNet2D([16, 32, 64, 128], [3, 3, 3, 3],
                                  [2, 2, 2, 2], ['VALID'])

    self.f1 = snt.Linear(output_size=512, name='f1')
    self.f2 = snt.Linear(output_size=512, name='f2')
    self.f3 = snt.Linear(output_size=256, name='f3')
    self.f4 = snt.Linear(output_size=128, name='f4')
    self.f5 = snt.Linear(output_size=2, name='f5')

  def _build(self, s1, s2):
    s = tf.concat([s1, s2], 3)
    e = self.enc(s)
    e1 = tf.reshape(e, [-1, 3*3*128])
    emb1 = tf.nn.relu(self.f1(e1))
    emb2 = tf.nn.relu(self.f2(emb1))
    emb3 = tf.nn.relu(self.f3(emb2))
    emb4 = tf.nn.relu(self.f4(emb3))
    emb5 = self.f5(emb4)
    return emb5


class TemporalModelLF(snt.AbstractModule):
  """Tempral distance model."""

  def __init__(self, name='itm'):
    super(TemporalModelLF, self).__init__(name=name)
    self.enc = snt.nets.ConvNet2D([16, 32, 64, 128], [3, 3, 3, 3],
                                  [2, 2, 2, 2], ['VALID'])
    self.f1 = snt.Linear(output_size=512, name='f1')
    self.f2 = snt.Linear(output_size=512, name='f2')
    self.f3 = snt.Linear(output_size=256, name='f3')
    self.f4 = snt.Linear(output_size=128, name='f4')
    self.f5 = snt.Linear(output_size=2, name='f5')

  def _build(self, s1, s2):
    e1 = self.enc(s1)
    e2 = self.enc(s2)
    e1 = tf.reshape(e1, [-1, 3*3*128])
    e2 = tf.reshape(e2, [-1, 3*3*128])
    e1 = tf.concat([e1, e2], 1)
    emb1 = tf.nn.relu(self.f1(e1))
    emb2 = tf.nn.relu(self.f2(emb1))
    emb3 = tf.nn.relu(self.f3(emb2))
    emb4 = tf.nn.relu(self.f4(emb3))
    emb5 = self.f5(emb4)
    return emb5
