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

"""Evaluate embeddings on downstream tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v2 as tf

from tcc.algorithms import get_algo
from tcc.config import CONFIG
from tcc.datasets import create_one_epoch_dataset
from tcc.utils import get_embeddings_dataset
from tcc.utils import get_lr_opt_global_step
from tcc.utils import restore_ckpt
from tcc.utils import setup_eval_dir

gfile = tf.gfile
layers = tf.keras.layers

flags.DEFINE_boolean('defun', True, 'Defun everything!')
flags.DEFINE_float('gpu_fraction', 1.0, 'Fraction of GPU to use.')
flags.DEFINE_string('save_path', '/tmp/embeddings.npy',
                    'where to store embeddings')
flags.DEFINE_string('dataset', None, 'dataset')
flags.DEFINE_string('split', None, 'split')
flags.DEFINE_string('logdir', None, 'Log dir for checkpoint.')
flags.DEFINE_string('path_to_tfrecords', '/tmp/%s_tfrecords', 'Path to '
                    'TFRecords.')
flags.DEFINE_integer('frames_per_batch', 30, 'frames_per_batch')
flags.DEFINE_boolean('visualize', False, 'Visualize images. Switched off by '
                     'for default to speed traininig up and take less memory.')
flags.DEFINE_boolean('keep_data', False, 'Keep frames of video with '
                     'embeddings.')
flags.DEFINE_boolean('keep_labels', True, 'Keep per-frame labels with '
                     'embeddings.')
flags.DEFINE_integer('sample_all_stride', 1, 'Stride between frames that will '
                     'be embedded.')
flags.DEFINE_integer(
    'max_embs', 0, 'Max number of videos to embed. 0 or less '
    'means embed all videos in dataset.')

FLAGS = flags.FLAGS

flags.mark_flag_as_required('dataset')
flags.mark_flag_as_required('split')
flags.mark_flag_as_required('logdir')


def evaluate():
  """Extract embeddings."""

  logdir = FLAGS.logdir
  setup_eval_dir(logdir)
  # Can ignore frame labels if dataset doesn't have per-frame labels.
  CONFIG.DATA.FRAME_LABELS = FLAGS.keep_labels
  # Subsample frames in case videos are long or fps is high to save memory.
  CONFIG.DATA.SAMPLE_ALL_STRIDE = FLAGS.sample_all_stride

  algo = get_algo(CONFIG.TRAINING_ALGO)
  _, optimizer, _ = get_lr_opt_global_step()
  restore_ckpt(logdir=logdir, optimizer=optimizer, **algo.model)

  if FLAGS.defun:
    algo.call = tf.function(algo.call)
    algo.compute_loss = tf.function(algo.compute_loss)

  iterator = create_one_epoch_dataset(FLAGS.dataset, FLAGS.split, mode='eval',
                                      path_to_tfrecords=FLAGS.path_to_tfrecords)

  max_embs = None if FLAGS.max_embs <= 0 else FLAGS.max_embs
  embeddings = get_embeddings_dataset(
      algo.model,
      iterator,
      frames_per_batch=FLAGS.frames_per_batch,
      keep_data=FLAGS.keep_data,
      keep_labels=FLAGS.keep_labels,
      max_embs=max_embs)
  np.save(gfile.Open(FLAGS.save_path, 'w'), embeddings)


def main(_):
  # Executing eagerly.
  gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction)
  tf.enable_eager_execution(config=tf.ConfigProto(gpu_options=gpu_opt))
  tf.keras.backend.set_learning_phase(0)
  evaluate()


if __name__ == '__main__':
  app.run(main)
