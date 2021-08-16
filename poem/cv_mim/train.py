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

"""Pose representation training with TFRecord inputs."""

from absl import app
from absl import flags
import tensorflow as tf

from poem.core import common
from poem.core import input_generator
from poem.core import keypoint_profiles
from poem.core import tfe_input_layer
from poem.cv_mim import train_base

FLAGS = flags.FLAGS

flags.adopt_module_key_flags(train_base)


def main(_):
  train_base.run(
      input_dataset_class=tf.data.TFRecordDataset,
      common_module=common,
      keypoint_profiles_module=keypoint_profiles,
      input_example_parser_creator=tfe_input_layer.create_tfe_parser,
      keypoint_preprocessor_3d=input_generator.preprocess_keypoints_3d)


if __name__ == '__main__':
  app.run(main)
