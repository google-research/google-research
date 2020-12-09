# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Pose embedding model training with TFRecord inputs."""

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

from poem import train_base
from poem.core import common
from poem.core import input_generator
from poem.core import keypoint_profiles
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.adopt_module_key_flags(common)
flags.adopt_module_key_flags(train_base)

flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')


def main(_):
  train_base.run(
      master=FLAGS.master,
      input_dataset_class=tf.data.TFRecordDataset,
      common_module=common,
      keypoint_profiles_module=keypoint_profiles,
      input_example_parser_creator=None,
      keypoint_preprocessor_3d=input_generator.preprocess_keypoints_3d,
      create_model_input_fn=input_generator.create_model_input,
      keypoint_distance_config_override={})


if __name__ == '__main__':
  app.run(main)
