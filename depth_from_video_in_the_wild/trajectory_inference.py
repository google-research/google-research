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

"""A binary for generating odometry trajectories given a checkpoint."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import threading
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from depth_from_video_in_the_wild import model
import cv2


WAIT_TIME = 20  # Wait time in seconds before checking for new checkpoint.
NUM_THREADS = 16  # Number of threads in which eval is calculated.

ODOMETRY_SETS = ['09-image_2', '10-image_2']


flags.DEFINE_string('output_dir', None, 'Directory to store predictions. '
                    'Subdirectories will be created for each checkpoint.')
flags.DEFINE_string('odometry_test_set_dir', None,
                    'Directory where the odomotry test sets are.')
flags.DEFINE_string('checkpoint_path', None, 'Directory containing checkpoints '
                    'to evaluate.')
flags.DEFINE_integer('img_height', 128, 'Input frame height.')
flags.DEFINE_integer('img_width', 416, 'Input frame width.')


FLAGS = flags.FLAGS


def trajectory_inference():
  """Generates trajectories from the KITTI odometry sets and a checkpoint."""

  # Note that the struct2depth code only works at batch_size=1, because it uses
  # the training mode of batchnorm at inference.
  inference_model = model.Model(
      is_training=False,
      batch_size=1,
      img_height=FLAGS.img_height,
      img_width=FLAGS.img_width)
  saver = tf.train.Saver()
  sess = tf.Session()

  def infer_egomotion(image1, image2):
    return inference_model.inference_egomotion(image1, image2, sess)

  saver.restore(sess, FLAGS.checkpoint_path)
  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  for odo_set in ODOMETRY_SETS:
    logging.info('Evaluating odometry on %s', odo_set)
    test_file_dir = os.path.join(FLAGS.odometry_test_set_dir, odo_set)
    output_file = os.path.join(FLAGS.output_dir, 'odometry_%s.txt' % odo_set)
    odometry_inference(test_file_dir, output_file, infer_egomotion)


def get_egomotion(im_files, results, infer_egomotion):
  for im_file in im_files:
    im = load_image(im_file)
    # Each image is a sequence of 3 frames. We use only the first 2.
    rot, trans = infer_egomotion(
        [im[:, :FLAGS.img_width, :]],
        [im[:, FLAGS.img_width:2 * FLAGS.img_width, :]])
    results[im_file] = (rot, trans)


def odometry_inference(image_sequence_dir, output_file, infer_egomotion):
  """Calculates egomotion inference and accumulates results into a trajectory.

  Args:
    image_sequence_dir: A string, directory where the odometry test sets reside.
    output_file: A string, file path where the trajectory is to be written.
    infer_egomotion: A callable that receives image1, image2 and outputs a
      rotation matrix and a translation vector connecting them in terms of
      egomotion.
  """
  im_files = sorted(tf.gfile.ListDirectory(image_sequence_dir))
  im_files = [
      os.path.join(image_sequence_dir, f)
      for f in im_files
      if 'png' in f and 'seg' not in f
  ]
  num_images = len(im_files)

  # Divide the work to NUM_THREADS threads
  results = [None] * NUM_THREADS
  group_size = int(math.ceil(num_images / NUM_THREADS))
  threads = []
  for tid in range(NUM_THREADS):
    results[tid] = {}
    group_start = group_size * tid
    group_end = min(group_size * (tid + 1), num_images)
    im_group = im_files[group_start:group_end]
    threads.append(
        threading.Thread(
            target=get_egomotion,
            args=(im_group, results[tid], infer_egomotion)))

  def processed_images():
    return sum([len(results[th]) for th in range(NUM_THREADS)])

  threads.append(
      threading.Thread(
          target=_logger, args=(num_images, processed_images)))
  for th in threads:
    th.start()
  for th in threads:
    th.join()

  combined_results = {}
  for result in results:
    combined_results.update(result)

  # Accumulate the position and the orientation, to generate a trajectory.
  position = np.zeros(3)
  orientation = np.eye(3)
  with tf.gfile.Open(output_file, 'w') as f:
    logging.info('Writing results to %s', output_file)
    f.write('0.0 0.0 0.0\n')  # Initial position
    for im_file in sorted(im_files):
      rot, trans = combined_results[im_file]
      orientation = orientation.dot(rot[0])
      position += orientation.dot(trans[0])
      f.write(' '.join([str(p) for p in position]) + '\n')


def _logger(total_images, processed_images):
  """A helper function to log the progress of all eval threads."""
  p = 0
  prev_p = -1
  while p < total_images:
    p = processed_images()
    if p > prev_p:
      logging.info('Processed %d out of %d.', p, total_images)
      prev_p = p
    time.sleep(WAIT_TIME)


def load_image(img_file, resize=None, interpolation='linear'):
  """Load image from disk. Output value range: [0,1]."""
  with tf.gfile.Open(img_file, 'rb') as f:
    im_data = np.fromstring(f.read(), np.uint8)
  im = cv2.imdecode(im_data, cv2.IMREAD_COLOR)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  if resize and resize != im.shape[:2]:
    ip = cv2.INTER_LINEAR if interpolation == 'linear' else cv2.INTER_NEAREST
    im = cv2.resize(im, resize, interpolation=ip)
  return im.astype(np.float32) / 255.0


def main(_):
  trajectory_inference()


if __name__ == '__main__':
  app.run(main)

