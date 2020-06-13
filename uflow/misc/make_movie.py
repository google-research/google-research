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

# pylint:disable=g-docstring-has-escape
"""Produces videos of flow predictions.

To generate a video on (for example) sintel, use:

python3 uflow.make_movie.py -c opt -- \
--alsologtostderr \
--plot_dir=<location to save video> \
--make_movie_on=sintel-clean:<path to dataset> \
--checkpoint_dir=<path to checkpoint to load> \
--height=448 --width=1024 \
--num_frames=1000

Note that the dataset passed to --make_movie must have been written in order,
otherwise the frames will appear scrambled.
"""

import sys

from absl import app
from absl import flags
import gin
import tensorflow as tf

from uflow import uflow_data
from uflow import uflow_evaluator
# pylint:disable=unused-import
from uflow import uflow_flags
from uflow import uflow_plotting


flags.DEFINE_integer('num_frames', 1000, 'How many frames to plot.')
flags.DEFINE_string('make_movie_on', '',
                    '"format0:path0;format1:path1", e.g. "kitti:/usr/..."')

FLAGS = flags.FLAGS


def main(unused_argv):
  if not FLAGS.plot_dir:
    raise ValueError('make_movie needs plot directory.')
  if not tf.io.gfile.exists(FLAGS.plot_dir):
    print('Making new plot directory', FLAGS.plot_dir)
    tf.io.gfile.makedirs(FLAGS.plot_dir)
  gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)
  uflow = uflow_evaluator.build_network(batch_size=1)
  uflow.update_checkpoint_dir(FLAGS.checkpoint_dir)
  uflow.restore()
  train_it = uflow_data.make_train_iterator(
      FLAGS.make_movie_on,
      height=None,
      width=None,
      shuffle_buffer_size=0,
      batch_size=1,
      seq_len=2,
      crop_instead_of_resize=False,
      apply_augmentation=False,
      include_ground_truth=False,
      include_occlusions=False,
      mode='video',
  )
  for i, (batch, _) in enumerate(train_it):
    if i > FLAGS.num_frames:
      break
    sys.stdout.write(':')
    sys.stdout.flush()
    flow = uflow.infer(batch[0, 0], batch[0, 1], input_height=FLAGS.height,
                       input_width=FLAGS.width)
    uflow_plotting.plot_movie_frame(FLAGS.plot_dir, i, batch[0, 0].numpy(),
                                    flow.numpy())


if __name__ == '__main__':
  app.run(main)
