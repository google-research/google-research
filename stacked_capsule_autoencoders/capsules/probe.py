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

"""Tools for model introspection."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import sonnet as snt
import tensorflow.compat.v1 as tf


def classification_probe(features, labels, n_classes, labeled=None):
  """Classification probe with stopped gradient on features."""

  def _classification_probe(features):
    logits = snt.Linear(n_classes)(tf.stop_gradient(features))
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=labels)
    if labeled is not None:
      xe = xe * tf.to_float(labeled)
    xe = tf.reduce_mean(xe)
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, axis=1),
                                              labels)))
    return xe, acc

  return snt.Module(_classification_probe)(features)
