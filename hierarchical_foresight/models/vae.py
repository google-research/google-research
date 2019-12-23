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

"""Variational Autoencoder Models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow.compat.v1 as tf


class ImageTransformSC(snt.AbstractModule):
  """VAE for the Maze Environment."""

  def __init__(self, latentsize, name='itmsc', width=64):
    super(ImageTransformSC, self).__init__(name=name)
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
    self.lin2 = snt.Linear(output_size=self.latentsize*2, name='lin2')
    self.lin3 = snt.Linear(output_size=self.lsz *3*128, name='lin3')

    self.f1 = snt.Linear(output_size=512, name='f1')
    self.f2 = snt.Linear(output_size=512, name='f2')
    self.f3 = snt.Linear(output_size=256, name='f3')

    self.fc1 = snt.Linear(output_size=256, name='fc')
    self.fc2 = snt.Linear(output_size=256, name='fc')
    self.fc3 = snt.Linear(output_size=256, name='fc')

  def _build(self, bs):
    self.s1 = tf.placeholder(tf.float32, shape=[None, self.width, 64, 3])
    self.s2 = tf.placeholder(tf.float32, shape=[None, self.width, 64, 3])

    c1 = self.enc(self.s1)
    c2 = self.enc(self.s2)
    e1 = tf.reshape(c1, [-1, self.lsz *3*128])
    e2 = tf.reshape(c2, [-1, self.lsz *3*128])
    e = tf.concat([e1, e2], 1)

    l1 = tf.nn.relu(self.lin1(e))
    l2 = self.lin2(l1)
    mu, std = l2[:, :self.latentsize], tf.nn.relu(l2[:, self.latentsize:])
    n = tf.distributions.Normal(loc=[0.]*self.latentsize,
                                scale=[1.]*self.latentsize)
    a = n.sample(bs)
    self.z = mu +  std * a

    emb1 = tf.nn.relu(self.f1(e1))
    emb2 = tf.nn.relu(self.f2(emb1))
    emb3 = self.f3(emb2)

    s2emb = tf.nn.relu(self.fc1(tf.concat([emb3, self.z], 1)))
    s2emb = tf.nn.relu(self.fc2(s2emb))
    s2emb = self.fc3(s2emb)

    ll = self.lin3(emb3)
    ll = tf.reshape(ll, [-1, self.lsz, 3, 128])
    dec1_3 = self.dec(ll+c1)
    rec = tf.nn.sigmoid(dec1_3)
    rec = tf.clip_by_value(rec, 1e-5, 1 - 1e-5)

    l3 = self.lin3(s2emb)
    l3 = tf.reshape(l3, [-1, self.lsz, 3, 128])
    dec2_3 = self.dec(l3+c1)
    o = tf.nn.sigmoid(dec2_3)
    o = tf.clip_by_value(o, 1e-5, 1 - 1e-5)
    return o, rec, mu, std**2
