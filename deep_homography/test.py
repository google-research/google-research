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

"""Functions to test neural networks on real-world images.

"""

from absl import app
from absl import flags

import cv2
import numpy as np

import tensorflow as tf
from deep_homography import hmg_util
from deep_homography import models

slim = tf.contrib.slim
logging = tf.logging

flags.DEFINE_string('image1', None, 'filename of the first input image')
flags.DEFINE_string('image2', None, 'filename of the second input image')
flags.DEFINE_string('model_path', None,
                    'Where to find the checkpoints for eval')
flags.DEFINE_string('out_dir', None, 'the output path')
flags.DEFINE_integer('train_height', 128,
                     'Height of images used when training the model')
flags.DEFINE_integer('train_width', 128,
                     'Width of images used when training the model')
flags.DEFINE_integer('num_level', 3, 'Number of hierarchical levels')
flags.DEFINE_integer('num_layer', 6,
                     'Number of layers in the motion feature network')
flags.DEFINE_enum('mode', 'test', ['test'], 'Mode of this run')
flags.DEFINE_enum('network_id', 'fmask_sem', ['hier', 'fmask_sem'],
                  'Type of network')

FLAGS = flags.FLAGS


def run_test():
  """Estimates the homography between two input images.
  """
  image1 = cv2.imread(FLAGS.image1)
  image2 = cv2.imread(FLAGS.image2)
  image_list = [image1, image2]
  image_norm_list = []
  for i in range(2):
    if FLAGS.network_id == 'fmask_sem':
      image_scale = cv2.resize(image_list[i],
                               (FLAGS.train_width, FLAGS.train_height),
                               cv2.INTER_LANCZOS4)
    else:
      image_gray = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2GRAY)
      image_scale = cv2.resize(image_gray,
                               (FLAGS.train_width, FLAGS.train_height),
                               cv2.INTER_LANCZOS4)

    image_norm = image_scale / 256.0 - 0.5
    image_norm_list.append(image_norm)
  if FLAGS.network_id == 'fmask_sem':
    norm_image_pair = np.expand_dims(np.concatenate(image_norm_list, 2), axis=0)
    num_channel = 3
  else:
    norm_image_pair = np.expand_dims(np.stack(image_norm_list, -1), axis=0)
    num_channel = 1

  batch_pairs = tf.placeholder(tf.float32,
                               [1, FLAGS.train_height, FLAGS.train_width,
                                2 * num_channel])
  with slim.arg_scope(models.homography_arg_scope()):
    if FLAGS.network_id == 'fmask_sem':
      batch_hmg_prediction, _ = models.hier_homography_fmask_estimator(
          batch_pairs, num_param=8, num_layer=FLAGS.num_layer,
          num_level=FLAGS.num_level, is_training=False)
    else:
      batch_hmg_prediction, _ = models.hier_homography_estimator(
          batch_pairs, num_param=8, num_layer=FLAGS.num_layer,
          num_level=FLAGS.num_level, is_training=False)

  batch_warped_result, _ = hmg_util.homography_warp_per_batch(
      batch_pairs[Ellipsis, 0 : num_channel],
      batch_hmg_prediction[FLAGS.num_level - 1])

  saver = tf.Saver()
  with tf.Session() as sess:
    saver.restore(sess, FLAGS.model_path)
    image_warp, homography_list = sess.run(
        [batch_warped_result, batch_hmg_prediction],
        feed_dict={batch_pairs: norm_image_pair})
    for i in range(8):
      logging.info('%f ', homography_list[FLAGS.num_level - 1][0][i])
    cv2.imwrite('%s/input0.jpg' % FLAGS.out_dir,
                (image_norm_list[0] + 0.5) * 256)
    cv2.imwrite('%s/input1.jpg' % FLAGS.out_dir,
                (image_norm_list[1] + 0.5) * 256)
    cv2.imwrite('%s/result.jpg' % FLAGS.out_dir, (image_warp[0] + 0.5) * 256)


def main(_):
  if FLAGS.mode == 'test':
    flags.mark_flag_as_required('image1')
    flags.mark_flag_as_required('image2')
    run_test()
  else:
    raise ValueError('Unknown mode: %s' % FLAGS.mode)


if __name__ == '__main__':
  flags.mark_flag_as_required('out_dir')
  flags.mark_flag_as_required('model_path')
  app.run(main)
