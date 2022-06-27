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

"""Model definitions for dsprites and 3dident experiments.
"""

from absl import flags

import tensorflow.compat.v2 as tf


FLAGS = flags.FLAGS


class LinearLayerOverPretrainedSimclrModel(tf.keras.Model):
  """Trainable linear evaluation layer over a pretrained SimCLR model.
  """

  def __init__(self, path, optimizer, num_classes):
    super().__init__()
    self.saved_model = tf.saved_model.load(path)
    self.dense_layer = tf.keras.layers.Dense(
        units=num_classes, name='affine_transform')
    self.optimizer = optimizer

  def call(self, x):
    outputs = self.saved_model(x, trainable=False)
    pred_t = self.dense_layer(outputs['final_avg_pool'])
    return pred_t

