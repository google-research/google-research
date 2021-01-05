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

r"""Supervised training when per-frame labels are available."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tcc.algos.algorithm import Algorithm
from tcc.config import CONFIG
from tcc.dataset_splits import DATASET_TO_NUM_CLASSES
from tcc.models import Classifier


class Classification(Algorithm):
  """Performs classification using labels."""

  def __init__(self, model=None):
    super(Classification, self).__init__(model)
    if len(CONFIG.DATASETS) > 1:
      raise ValueError('Classification does not support multiple datasets yet.')

    self._num_classes = DATASET_TO_NUM_CLASSES[CONFIG.DATASETS[0]]
    fc_layers = [(self._num_classes, False)]
    classifier = Classifier(fc_layers, CONFIG.CLASSIFICATION.DROPOUT_RATE)
    self.model['classifier'] = classifier

  def get_algo_variables(self):
    return self.model['classifier'].variables

  def compute_loss(self, embs, steps, seq_lens, global_step, training,
                   frame_labels, seq_labels):

    if training:
      num_steps = CONFIG.TRAIN.NUM_FRAMES
    else:
      num_steps = CONFIG.EVAL.NUM_FRAMES

    embs = tf.squeeze(tf.concat(tf.split(embs, num_steps, axis=1),
                                axis=0),
                      axis=1)
    logits = self.model['classifier'](embs)
    num_frames_per_step = CONFIG.DATA.NUM_STEPS
    labels = frame_labels[:, num_frames_per_step - 1::num_frames_per_step]

    labels = tf.squeeze(tf.concat(tf.split(labels, num_steps, axis=1),
                                  axis=0),
                        axis=1)
    labels = tf.one_hot(labels, self._num_classes)

    loss = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(
            y_true=labels,
            y_pred=logits,
            from_logits=True,
            label_smoothing=CONFIG.CLASSIFICATION.LABEL_SMOOTHING))
    return loss
