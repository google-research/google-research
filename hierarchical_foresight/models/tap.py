# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Time Agnostic Prediction Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf


class TAP(snt.AbstractModule):
  """Time Agnostic Prediction Model."""

  def __init__(self, latentsize, name='itmsc', width=64):
    super(TAP, self).__init__(name=name)
    self.width = width
    if self.width == 48:
      self.lsz = 2
    else:
      self.lsz = 3
    self.latentsize = latentsize
    self.enc = snt.nets.ConvNet2D([16, 32, 64, 128], [3, 3, 3, 3],
                                  [2, 2, 2, 2], ['VALID'])

    self.dec = self.enc.transpose()
    self.lin1 = snt.Linear(output_size=512, name='lin1')
    self.lin2 = snt.Linear(output_size=self.latentsize*1,
                           name='lin2')
    self.lin3 = snt.Linear(output_size=self.lsz *3*128, name='lin3')

  def _build(self, bs, s1, s2):
    c1 = self.enc(s1)
    c2 = self.enc(s2)
    e1 = tf.reshape(c1, [-1, self.lsz *3*128])
    e2 = tf.reshape(c2, [-1, self.lsz *3*128])
    e = tf.concat([e1, e2], 1)

    l1 = tf.nn.relu(self.lin1(e))
    l2 = self.lin2(l1)
    self.z = l2
    ll = self.lin3(self.z)
    ll = tf.reshape(ll, [-1, self.lsz, 3, 128])
    dec1_3 = self.dec(ll)
    rec = tf.nn.sigmoid(dec1_3)
    rec = tf.clip_by_value(rec, 1e-3, 1 - 1e-3)
    return rec
