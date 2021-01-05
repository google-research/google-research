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

"""EM Capsule Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import tensorflow.compat.v1 as tf
from capsule_em import em_layers
from capsule_em import simple_model
from capsule_em import utils
FLAGS = tf.app.flags.FLAGS


def _build_capsule(input_tensor, input_atom, position_grid, num_classes):
  """Stack capsule layers."""
  # input_tensor: [x, ch, atom, c1, c2] (64, 5x5, 2 conv1)
  print('hidden: ')
  print(input_tensor.get_shape())
  conv_caps_act, conv_caps_center = em_layers.primary_caps(
      input_tensor,
      input_atom,
      FLAGS.num_prime_capsules,
      FLAGS.num_primary_atoms,
  )
  with tf.name_scope('primary_act'):
    utils.activation_summary(conv_caps_act)
  with tf.name_scope('primary_center'):
    utils.activation_summary(conv_caps_center)
  last_dim = FLAGS.num_prime_capsules
  if FLAGS.extra_caps > 0:
    for i in range(FLAGS.extra_caps):
      if FLAGS.fast:
        conv_function = em_layers.conv_capsule_mat_fast
      else:
        conv_function = em_layers.conv_capsule_mat

      conv_caps_act, conv_caps_center = conv_function(
          conv_caps_center,
          conv_caps_act,
          last_dim,
          int(FLAGS.caps_dims.split(',')[i]),
          'convCaps{}'.format(i),
          FLAGS.routing_iteration,
          num_in_atoms=int(math.sqrt(FLAGS.num_primary_atoms)),
          num_out_atoms=int(math.sqrt(FLAGS.num_primary_atoms)),
          stride=int(FLAGS.caps_strides.split(',')[i]),
          kernel_size=int(FLAGS.caps_kernels.split(',')[i]),
          final_beta=FLAGS.final_beta,
      )

      position_grid = simple_model.conv_pos(
          position_grid, int(FLAGS.caps_kernels.split(',')[i]),
          int(FLAGS.caps_strides.split(',')[i]), 'VALID')
      last_dim = int(FLAGS.caps_dims.split(',')[i])
      print(conv_caps_center.get_shape())
      print(conv_caps_act.get_shape())

  capsule1_act = tf.layers.flatten(conv_caps_act)

  position_grid = tf.squeeze(position_grid, axis=[0])
  position_grid = tf.transpose(position_grid, [1, 2, 0])
  return em_layers.connector_capsule_mat(
      input_tensor=conv_caps_center,
      position_grid=position_grid,
      input_activation=capsule1_act,
      input_dim=last_dim,
      output_dim=num_classes,
      layer_name='capsule2',
      num_routing=FLAGS.routing_iteration,
      num_in_atoms=int(math.sqrt(FLAGS.num_primary_atoms)),
      num_out_atoms=int(math.sqrt(FLAGS.num_primary_atoms)),
      leaky=FLAGS.leaky,
      final_beta=FLAGS.final_beta,
  ), conv_caps_act


def inference(features):
  """Inference for EM Capsules: Conv+Caps."""
  num_classes = features['num_classes']
  conv, conv_dim, position_grid = simple_model.add_convs(features)

  final_capsule, mid_act = _build_capsule(
      conv,
      input_atom=conv_dim,
      position_grid=position_grid,
      num_classes=num_classes)
  capsule_activation, _ = final_capsule
  recons = None
  recons_2 = None

  return capsule_activation, recons, recons_2, mid_act
