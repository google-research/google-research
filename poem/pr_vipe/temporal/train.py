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

"""Temporal pose embedding model training with TFRecord inputs."""

import functools

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

from poem.core import common
from poem.core import input_generator
from poem.core import keypoint_profiles
from poem.core import keypoint_utils
from poem.core import models
from poem.core import tfse_input_layer
from poem.pr_vipe import train_base

tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.adopt_module_key_flags(train_base)

FLAGS.set_default('input_keypoint_profile_name_3d', '3DSTD16')
FLAGS.set_default('input_keypoint_profile_name_2d', '2DSTD13')
FLAGS.set_default('base_model_type', 'TEMPORAL_SIMPLE_LATE_FUSE')
FLAGS.set_default('input_shuffle_buffer_size', 262144)
# CSV of 3-tuple probability (probability_to_apply, probability_to_drop,
# probability_to_use_sequence_dropout) for performing stratified keypoint
# dropout. The third probability is optional and for the chance that dropout is
# applied to a sequence input.
FLAGS.set_default('keypoint_dropout_probs', ['0.0', '0.0', '0.0'])

flags.DEFINE_integer('input_sequence_length', 7, 'Length of input sequences.')

flags.DEFINE_integer(
    'num_late_fusion_preprojection_nodes', 0,
    'The dimension to project each frame features to before late fusion. No '
    'preprojection will be added if non-positive.')

flags.DEFINE_string(
    'late_fusion_preprojection_activation_fn', 'NONE',
    'Name of the activation function of the preprojection layer. Supported '
    'activation functions include `RELU` and `NONE` (no activation function). '
    'Only used if preprojection layer is added.')

flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')


def _create_keypoint_distance_config_override():
  """Creates keypoint distance configuration override."""
  if FLAGS.keypoint_distance_type == common.KEYPOINT_DISTANCE_TYPE_MPJPE:
    return {
        'keypoint_distance_fn':
            keypoint_utils.compute_temporal_procrustes_aligned_mpjpes,
        'min_negative_keypoint_distance':
            FLAGS.min_negative_keypoint_mpjpe,
    }
  raise ValueError('Unsupported keypoint distance type: `%s`.' %
                   str(FLAGS.keypoint_distance_type))


def main(_):
  input_example_parser_creator = functools.partial(
      tfse_input_layer.create_tfse_parser,
      sequence_length=FLAGS.input_sequence_length)

  create_model_input_fn_kwargs = {'sequential_inputs': True}

  train_base.run(
      master=FLAGS.master,
      input_dataset_class=tf.data.TFRecordDataset,
      common_module=common,
      keypoint_profiles_module=keypoint_profiles,
      models_module=models,
      input_example_parser_creator=input_example_parser_creator,
      keypoint_preprocessor_3d=input_generator.preprocess_keypoints_3d,
      keypoint_distance_config_override=(
          _create_keypoint_distance_config_override()),
      create_model_input_fn_kwargs=create_model_input_fn_kwargs,
      embedder_fn_kwargs={
          'num_late_fusion_preprojection_nodes':
              FLAGS.num_late_fusion_preprojection_nodes,
          'late_fusion_preprojection_activation_fn':
              FLAGS.late_fusion_preprojection_activation_fn,
      })


if __name__ == '__main__':
  app.run(main)
