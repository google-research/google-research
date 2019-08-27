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

"""Evaluates a single trained model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import os

from absl import app
from absl import flags

import numpy as np

import tensorflow as tf  # tf

from evanet import model_dna

flags.DEFINE_string('checkpoints', '',
                    'Comma separated list of model checkpoints.')

FLAGS = flags.FLAGS

# encoded protobuf strings representing the final RGB and optical flow networks
_RGB_NETWORKS = [
    'CAMQARgCIAEoATAJOhQIAhAAGggIARAAGAEgARoECAMYBzomCAQQABoECAMYAxoCCAAaCAgBEAAYASABGgIIABoICAEQABgFIAE6HggFEAAaBAgDGAcaAggAGggIARABGAsgARoECAMYCzoeCAMQABoECAMYCxoICAEQABgBIAEaCAgBEAEYAyAB',
    'CAMQARgCIAEoAzAJOioIAhAAGgQIAxgBGgQIAxgDGg4IAhABGAEgASgJMAE4AhoECAMYARoCCAA6RAgDEAAaBAgDGAEaBAgDGAcaDggCEAIYBSABKAEwATgAGg4IAhABGAEgASgLMAE4AhoCCAAaDggCEAIYCyABKAEwATgAOhQIBRAAGgQIAxgFGgIIABoECAMYCToUCAQQABoICAEQAhgBIAEaBAgDGAU='
]
_FLOW_NETWORKS = [
    'CAMQARgCIAEoATAJOhQIAhAAGggIARAAGAEgARoECAMYBzomCAQQABoECAMYAxoCCAAaCAgBEAAYASABGgIIABoICAEQABgFIAE6HggFEAAaBAgDGAcaAggAGggIARABGAsgARoECAMYCzoeCAMQABoECAMYCxoICAEQABgBIAEaCAgBEAEYAyAB',
    'CAMQARgCIAEoAzAJOioIAhAAGgQIAxgBGgQIAxgDGg4IAhABGAEgASgJMAE4AhoECAMYARoCCAA6RAgDEAAaBAgDGAEaBAgDGAcaDggCEAIYBSABKAEwATgAGg4IAhABGAEgASgLMAE4AhoCCAAaDggCEAIYCyABKAEwATgAOhQIBRAAGgQIAxgFGgIIABoECAMYCToUCAQQABoICAEQAhgBIAEaBAgDGAU='
]


def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()


def main(_):
  weights = [x for x in FLAGS.checkpoints.split(',') if len(x)]
  videos = [
      'evanet/data/v_CricketShot_g04_c01_rgb.npy',
      'evanet/data/v_CricketShot_g04_c01_rgb.npy',
      'evanet/data/v_CricketShot_g04_c01_flow.npy',
      'evanet/data/v_CricketShot_g04_c01_flow.npy'
  ]

  label_map = 'evanet/data/label_map.txt'
  kinetics_classes = [x.strip() for x in open(label_map)]

  # create model
  final_logits = np.zeros((400,), np.float32)
  for i, model_str in enumerate(_RGB_NETWORKS + _FLOW_NETWORKS):
    tf.reset_default_graph()
    vid = np.load(videos[i])

    vid_placeholder = tf.placeholder(tf.float32, shape=vid.shape)

    model = model_dna.ModelDNA(base64.b64decode(model_str), num_classes=400)
    logits = model.model(vid_placeholder, mode='eval')

    variable_map = {}
    for var in tf.global_variables():
      variable_map[var.name.replace(':0', '').replace('Conv/', 'Conv3d/')] = var

    saver = tf.train.Saver(var_list=variable_map, reshape=True)
    with tf.Session() as sess:
      if i <= len(weights) - 1:
        saver.restore(sess,
                      os.path.join('evanet', 'data', 'checkpoints', weights[i]))
      else:
        print('Warning, model has no pretrained weights')
        sess.run(tf.global_variables_initializer())
      out_logits = sess.run([logits], feed_dict={vid_placeholder: vid})
      final_logits += out_logits[0][0]

  # average prediction
  final_logits /= float(len(_RGB_NETWORKS + _FLOW_NETWORKS))
  predictions = softmax(final_logits)
  sorted_indices = np.argsort(final_logits)[::-1]

  print('\nTop classes and probabilities')
  for index in sorted_indices[:20]:
    print(predictions[index], final_logits[index], kinetics_classes[index])


if __name__ == '__main__':
  app.run(main)
