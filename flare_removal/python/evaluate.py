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

"""Evaluation script for flare removal."""

import os.path

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from flare_removal.python import data_provider
from flare_removal.python import losses
from flare_removal.python import models
from flare_removal.python import synthesis


flags.DEFINE_string(
    'eval_dir', '/tmp/eval',
    'Directory where evaluation summaries and outputs are written.')
flags.DEFINE_string(
    'train_dir', '/tmp/train',
    'Directory where training checkpoints are written. This script will '
    'repeatedly poll and evaluate the latest checkpoint.')
flags.DEFINE_string('scene_dir', None,
                    'Full path to the directory containing scene images.')
flags.DEFINE_string('flare_dir', None,
                    'Full path to the directory containing flare images.')
flags.DEFINE_enum(
    'data_source', 'jpg', ['tfrecord', 'jpg'],
    'Source of training data. Use "jpg" for individual image files, such as '
    'JPG and PNG images. Use "tfrecord" for pre-baked sharded TFRecord files.')
flags.DEFINE_string('model', 'unet', 'the name of the training model')
flags.DEFINE_string('loss', 'percep', 'the name of the loss for training')
flags.DEFINE_integer('batch_size', 2, 'Evaluation batch size.')
flags.DEFINE_float(
    'learning_rate', 1e-4,
    'Unused placeholder. The flag has to be defined to satisfy parameter sweep '
    'requirements.')
flags.DEFINE_float(
    'scene_noise', 0.01,
    'Gaussian noise sigma added in the scene in synthetic data. The actual '
    'Gaussian variance for each image will be drawn from a Chi-squared '
    'distribution with a scale of scene_noise.')
flags.DEFINE_float(
    'flare_max_gain', 10.0,
    'Max digital gain applied to the flare patterns during synthesis.')
flags.DEFINE_float('flare_loss_weight', 1.0,
                   'Weight added on the flare loss (scene loss is 1).')
flags.DEFINE_integer('training_res', 512,
                     'Image resolution at which the network is trained.')
FLAGS = flags.FLAGS


def main(_):
  eval_dir = FLAGS.eval_dir
  assert eval_dir, 'Flag --eval_dir must not be empty.'
  train_dir = FLAGS.train_dir
  assert train_dir, 'Flag --train_dir must not be empty.'
  summary_dir = os.path.join(eval_dir, 'summary')

  # Load data.
  scenes = data_provider.get_scene_dataset(
      FLAGS.scene_dir, FLAGS.data_source, FLAGS.batch_size, repeat=0)
  flares = data_provider.get_flare_dataset(FLAGS.flare_dir, FLAGS.data_source,
                                           FLAGS.batch_size)

  # Make a model.
  model = models.build_model(FLAGS.model, FLAGS.batch_size)
  loss_fn = losses.get_loss(FLAGS.loss)

  ckpt = tf.train.Checkpoint(
      step=tf.Variable(0, dtype=tf.int64),
      training_finished=tf.Variable(False, dtype=tf.bool),
      model=model)

  summary_writer = tf.summary.create_file_writer(summary_dir)

  # The checkpoints_iterator keeps polling the latest training checkpoints,
  # until:
  #   1) `timeout` seconds have passed waiting for a new checkpoint; and
  #   2) `timeout_fn` (in this case, the flag indicating the last training
  #      checkpoint) evaluates to true.
  for ckpt_path in tf.train.checkpoints_iterator(
      train_dir, timeout=30, timeout_fn=lambda: ckpt.training_finished):
    try:
      status = ckpt.restore(ckpt_path)
      # Assert that all model variables are restored, but allow extra unmatched
      # variables in the checkpoint. (For example, optimizer states are not
      # needed for evaluation.)
      status.assert_existing_objects_matched()
      # Suppress warnings about unmatched variables.
      status.expect_partial()
      logging.info('Restored checkpoint %s @ step %d.', ckpt_path, ckpt.step)
    except (tf.errors.NotFoundError, AssertionError):
      logging.exception('Failed to restore checkpoint from %s.', ckpt_path)
      continue

    for scene, flare in tf.data.Dataset.zip((scenes, flares)):
      loss_value, summary = synthesis.run_step(
          scene,
          flare,
          model,
          loss_fn,
          noise=FLAGS.scene_noise,
          flare_max_gain=FLAGS.flare_max_gain,
          flare_loss_weight=FLAGS.flare_loss_weight,
          training_res=FLAGS.training_res)
    with summary_writer.as_default():
      tf.summary.image('prediction', summary, max_outputs=1, step=ckpt.step)
      tf.summary.scalar('loss', loss_value, step=ckpt.step)

  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)
