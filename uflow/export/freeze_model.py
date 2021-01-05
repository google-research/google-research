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

"""Script for freezing UFlow models."""

import os

from absl import app
from absl import flags
import cv2
import tensorflow as tf

from uflow import uflow_plotting
from uflow.uflow_net import UFlow

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir', '',
                    'Path to directory for saving and restoring checkpoints.')
flags.DEFINE_integer('height', 512, 'Image height for training and evaluation.')
flags.DEFINE_integer('width', 640, 'Image height for training and evaluation.')
flags.DEFINE_integer('resize_to_height', 512, 'Height to resize to')
flags.DEFINE_integer('resize_to_width', 640, 'Width to resize to')
flags.DEFINE_bool(
    'output_row_col', False, 'Whether to output flow in row, col '
    'format. If False, will output in col, row format.')
flags.DEFINE_string('image1_path', 'uflow/files/00000.jpg',
                    'Path to image1 for quality control.')
flags.DEFINE_string('image2_path', 'uflow/files/00001.jpg',
                    'Path to image2 for quality control.')
flags.DEFINE_float(
    'channel_multiplier', 1.,
    'Globally multiply the number of model convolution channels'
    'by this factor.')
flags.DEFINE_integer('num_levels', 5, 'The number of feature pyramid levels to '
                     'use.')

tf.compat.v1.disable_eager_execution()


def convert_graph(checkpoint_dir, height, width, output_row_col):
  """Converts uflow to a frozen model."""

  print('Building the model.')
  tf.compat.v1.reset_default_graph()
  uflow = UFlow(
      checkpoint_dir=checkpoint_dir,
      channel_multiplier=FLAGS.channel_multiplier,
      num_levels=FLAGS.num_levels)
  image1 = tf.compat.v1.placeholder(
      tf.float32, [height, width, 3], name='first_image')
  image2 = tf.compat.v1.placeholder(
      tf.float32, [height, width, 3], name='second_image')

  flow = uflow.infer_no_tf_function(
      image1,
      image2,
      input_height=FLAGS.resize_to_height,
      input_width=FLAGS.resize_to_width)
  if output_row_col:
    flow = tf.identity(flow, 'flow')
  else:
    flow = tf.identity(flow[Ellipsis, ::-1], 'flow')
  print('Loading the checkpoint.')
  saver = tf.compat.v1.train.Saver()
  sess = tf.compat.v1.Session()
  sess.run(tf.compat.v1.global_variables_initializer())
  sess.run(tf.compat.v1.local_variables_initializer())
  # Apparently, uflow.restore() does not work here in graph mode.
  saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

  # Load images.
  image1_np = cv2.imread(FLAGS.image1_path)[:, :, ::-1].astype('float32') / 255.
  image2_np = cv2.imread(FLAGS.image2_path)[:, :, ::-1].astype('float32') / 255.
  image1_np = cv2.resize(image1_np, (FLAGS.width, FLAGS.height))
  image2_np = cv2.resize(image2_np, (FLAGS.width, FLAGS.height))

  # Compute test flow and save.
  flow_np = sess.run(flow, feed_dict={image1: image1_np, image2: image2_np})
  uflow_plotting.plot_flow(image1_np, image2_np, flow_np, 'test_result.png',
                           checkpoint_dir)

  print('Freezing and optimizing the graph.')
  dg = tf.compat.v1.get_default_graph()
  frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
      sess,  # The session is used to retrieve the weights.
      dg.as_graph_def(),  # The graph_def is used to retrieve the nodes.
      ['flow']  # The output node names are used to select the useful nodes.
  )

  print('Writing the result to', checkpoint_dir)
  filename = os.path.join(checkpoint_dir, 'uflow.pb')
  with tf.io.gfile.GFile(filename, 'wb') as f:
    f.write(frozen_graph_def.SerializeToString())


def main(unused_argv):
  convert_graph(FLAGS.checkpoint_dir, FLAGS.height, FLAGS.width,
                FLAGS.output_row_col)


if __name__ == '__main__':
  app.run(main)
