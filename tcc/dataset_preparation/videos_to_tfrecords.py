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

r"""Convert list of videos to tfrecords based on SequenceExample."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from tcc.dataset_preparation.dataset_utils import create_tfrecords

flags.DEFINE_string('input_dir', None, 'Path to videos.')
flags.DEFINE_string('name', None, 'Name of the dataset being created. This will'
                    'be used as a prefix.')
flags.DEFINE_string('file_pattern', '*.mp4', 'Pattern used to searh for files'
                    'in the given directory.')
flags.DEFINE_string('label_file', None, 'Provide a corresponding labels file'
                    'that stores per-frame or per-sequence labels. This info'
                    'will get stored.')
flags.DEFINE_string('output_dir', '/tmp/tfrecords/', 'Output directory where'
                    'tfrecords will be stored.')
flags.DEFINE_integer('files_per_shard', 1, 'Number of videos to store in a'
                     'shard.')
flags.DEFINE_boolean('rotate', False, 'Rotate videos by 90 degrees before'
                     'creating tfrecords')
flags.DEFINE_boolean('resize', True, 'Resize videos to a given size.')
flags.DEFINE_integer('width', 224, 'Width of frames in the TFRecord.')
flags.DEFINE_integer('height', 224, 'Height of frames in the TFRecord.')
flags.DEFINE_list(
    'frame_labels', '', 'Comma separated list of descriptions '
    'for labels given on a per frame basis. For example: '
    'winding_up,early_cocking,acclerating,follow_through')
flags.DEFINE_integer('action_label', -1, 'Action label of all videos.')
flags.DEFINE_integer('expected_segments', -1, 'Expected number of segments.')
flags.DEFINE_integer('fps', 0, 'Frames per second of video. If 0, fps will be '
                     'read from metadata of video.')
FLAGS = flags.FLAGS


def main(_):
  create_tfrecords(FLAGS.name, FLAGS.output_dir, FLAGS.input_dir,
                   FLAGS.label_file, FLAGS.file_pattern, FLAGS.files_per_shard,
                   FLAGS.action_label, FLAGS.frame_labels,
                   FLAGS.expected_segments, FLAGS.fps, FLAGS.rotate,
                   FLAGS.resize, FLAGS.width, FLAGS.height)


if __name__ == '__main__':
  app.run(main)
