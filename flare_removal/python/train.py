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

r"""Training script for flare removal.

This script trains a model that outputs a flare-free image from a flare-polluted
image.
"""
import os.path
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from flare_removal.python import data_provider
from flare_removal.python import losses
from flare_removal.python import models
from flare_removal.python import synthesis


flags.DEFINE_string(
    'train_dir', '/tmp/train',
    'Directory where training checkpoints and summaries are written.')
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
flags.DEFINE_integer('batch_size', 2, 'Training batch size.')
flags.DEFINE_integer('epochs', 100, 'Training config: epochs.')
flags.DEFINE_integer(
    'ckpt_period', 1000,
    'Write model checkpoint and summary to disk every ckpt_period steps.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
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
flags.DEFINE_integer('training_res', 512, 'Training resolution.')
FLAGS = flags.FLAGS


@tf.function
def train_step(model, scene, flare, loss_fn, optimizer):
  """Executes one step of gradient descent."""
  with tf.GradientTape() as tape:
    loss_value, summary = synthesis.run_step(
        scene,
        flare,
        model,
        loss_fn,
        noise=FLAGS.scene_noise,
        flare_max_gain=FLAGS.flare_max_gain,
        flare_loss_weight=FLAGS.flare_loss_weight,
        training_res=FLAGS.training_res)
  grads = tape.gradient(loss_value, model.trainable_weights)
  grads, _ = tf.clip_by_global_norm(grads, 5.0)
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
  return loss_value, summary


def main(_):
  train_dir = FLAGS.train_dir
  assert train_dir, 'Flag --train_dir must not be empty.'
  summary_dir = os.path.join(train_dir, 'summary')
  model_dir = os.path.join(train_dir, 'model')

  # Load data.
  scenes = data_provider.get_scene_dataset(
      FLAGS.scene_dir, FLAGS.data_source, FLAGS.batch_size, repeat=FLAGS.epochs)
  flares = data_provider.get_flare_dataset(FLAGS.flare_dir, FLAGS.data_source,
                                           FLAGS.batch_size)

  # Make a model.
  model = models.build_model(FLAGS.model, FLAGS.batch_size)
  optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
  loss_fn = losses.get_loss(FLAGS.loss)

  # Model checkpoints. Checkpoints don't contain model architecture, but
  # weights only. We use checkpoints to keep track of the training progress.
  ckpt = tf.train.Checkpoint(
      step=tf.Variable(0, dtype=tf.int64),
      training_finished=tf.Variable(False, dtype=tf.bool),
      optimizer=optimizer,
      model=model)
  ckpt_mgr = tf.train.CheckpointManager(
      ckpt, train_dir, max_to_keep=3, keep_checkpoint_every_n_hours=3)

  # Restore the latest checkpoint (model weights), if any. This is helpful if
  # the training job gets restarted from an unexpected termination.
  latest_ckpt = ckpt_mgr.latest_checkpoint
  restore_status = None
  if latest_ckpt is not None:
    # Note that due to lazy initialization, not all checkpointed variables can
    # be restored at this point. Hence 'expect_partial()'. Full restoration is
    # checked in the first training step below.
    restore_status = ckpt.restore(latest_ckpt).expect_partial()
    logging.info('Restoring latest checkpoint @ step %d from: %s', ckpt.step,
                 latest_ckpt)
  else:
    logging.info('Previous checkpoints not found. Starting afresh.')

  summary_writer = tf.summary.create_file_writer(summary_dir)

  step_time_metric = tf.keras.metrics.Mean('step_time')
  step_start_time = time.time()
  for scene, flare in tf.data.Dataset.zip((scenes, flares)):
    # Perform one training step.
    loss_value, summary = train_step(model, scene, flare, loss_fn, optimizer)

    # By this point, all lazily initialized variables should have been
    # restored by the checkpoint if one was available.
    if restore_status is not None:
      restore_status.assert_consumed()
      restore_status = None

    # Write training summaries and checkpoints to disk.
    ckpt.step.assign_add(1)
    if ckpt.step % FLAGS.ckpt_period == 0:
      # Write model checkpoint to disk.
      ckpt_mgr.save()

      # Also save the full model using the latest weights. To restore previous
      # weights, you'd have to load the model and restore a previously saved
      # checkpoint.
      tf.keras.models.save_model(model, model_dir, save_format='tf')

      # Write summaries to disk, which can be visualized with TensorBoard.
      with summary_writer.as_default():
        tf.summary.image('prediction', summary, max_outputs=1, step=ckpt.step)
        tf.summary.scalar('loss', loss_value, step=ckpt.step)
        tf.summary.scalar(
            'step_time', step_time_metric.result(), step=ckpt.step)
        step_time_metric.reset_state()

    # Record elapsed time in this training step.
    step_end_time = time.time()
    step_time_metric.update_state(step_end_time - step_start_time)
    step_start_time = step_end_time

  ckpt.training_finished.assign(True)
  ckpt_mgr.save()
  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)
