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

r"""Train variations of DirectionNet for relative camera pose estimation."""
import collections
import time

from absl import app
from absl import flags
import dataset_loader
import losses
import model
import tensorflow.compat.v1 as tf
import util

tf.compat.v1.disable_eager_execution()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'master', 'local', 'BNS name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
flags.DEFINE_integer(
    'ps_tasks', 0, 'Number of tasks in the ps job. If 0 no ps job is used.')
flags.DEFINE_string(
    'checkpoint_dir', '', 'The directory to save checkpoints and summaries.')
flags.DEFINE_string(
    'data_dir', '', 'The training data directory.')
flags.DEFINE_string(
    'model', '9D',
    '9D (rotation), 6D (rotation), T (translation), Single (no derotation)')
flags.DEFINE_integer('batch', 20, 'The size of mini-batches.')
flags.DEFINE_integer('n_epoch', -1, 'Number of training epochs.')
flags.DEFINE_integer(
    'distribution_height', 64, 'The height dimension of output distributions.')
flags.DEFINE_integer(
    'distribution_width', 64, 'The width dimension of output distributions.')
flags.DEFINE_integer(
    'transformed_height', 344,
    'The height dimension of input images after derotation transformation.')
flags.DEFINE_integer(
    'transformed_width', 344,
    'The width dimension of input images after derotation transformation.')
flags.DEFINE_float('lr', 1e-3, 'The learning rate.')
flags.DEFINE_float('alpha', 8e7,
                   'The weight of the distribution loss.')
flags.DEFINE_float('beta', 0.1,
                   'The weight of the spread loss.')
flags.DEFINE_float('kappa', 10.,
                   'A coefficient multiplied by the concentration loss.')
flags.DEFINE_float(
    'transformed_fov', 105.,
    'The field of view of input images after derotation transformation.')
flags.DEFINE_bool('derotate_both', True,
                  'Derotate both input images when training DirectionNet-T')

Computation = collections.namedtuple('Computation',
                                     ['train_op', 'loss', 'global_step'])


def direction_net_rotation(src_img,
                           trt_img,
                           rotation_gt,
                           n_output_distributions=3):
  """Build the computation graph to train the DirectionNet-R.

  Args:
    src_img: [BATCH, HEIGHT, WIDTH, 3] input source images.
    trt_img: [BATCH, HEIGHT, WIDTH, 3] input target images.
    rotation_gt: [BATCH, 3, 3] ground truth rotation matrices.
    n_output_distributions: (int) number of output distributions. (either two or
    three) The model uses 9D representation for rotations when it is 3 and the
    model uses 6D representation when it is 2.

  Returns:
    A collection of tensors including training ops, loss, and global step count.

  Raises:
    ValueError: 'n_output_distributions' must be either 2 or 3.
  """
  if n_output_distributions != 3 and n_output_distributions != 2:
    raise ValueError("'n_output_distributions' must be either 2 or 3.")

  net = model.DirectionNet(n_output_distributions)
  global_step = tf.train.get_or_create_global_step()
  directions_gt = rotation_gt[:, :n_output_distributions]
  distribution_gt = util.spherical_normalization(util.von_mises_fisher(
      directions_gt,
      tf.constant(FLAGS.kappa, tf.float32),
      [FLAGS.distribution_height, FLAGS.distribution_width]), rectify=False)

  pred = net(src_img, trt_img, training=True)
  directions, expectation, distribution_pred = util.distributions_to_directions(
      pred)
  if n_output_distributions == 3:
    rotation_estimated = util.svd_orthogonalize(directions)
  elif n_output_distributions == 2:
    rotation_estimated = util.gram_schmidt(directions)

  direction_loss = losses.direction_loss(directions, directions_gt)
  distribution_loss = tf.constant(
      FLAGS.alpha, tf.float32) * losses.distribution_loss(
          distribution_pred, distribution_gt)
  spread_loss = tf.cast(
      FLAGS.beta, tf.float32) * losses.spread_loss(expectation)
  rotation_error = tf.reduce_mean(util.rotation_geodesic(
      rotation_estimated, rotation_gt))
  direction_error = tf.reduce_mean(tf.acos(tf.clip_by_value(
      tf.reduce_sum(directions * directions_gt, -1), -1., 1.)))

  loss = direction_loss + distribution_loss + spread_loss

  tf.summary.scalar('loss', loss)
  tf.summary.scalar('distribution_loss', distribution_loss)
  tf.summary.scalar('spread_loss', spread_loss)
  tf.summary.scalar('direction_error',
                    util.radians_to_degrees(direction_error))
  tf.summary.scalar('rotation_error',
                    util.radians_to_degrees(rotation_error))

  for i in range(n_output_distributions):
    tf.summary.image('distribution/rotation/ground_truth_%d'%(i+1),
                     distribution_gt[:, :, :, i:i+1],
                     max_outputs=4)
    tf.summary.image('distribution/rotation/prediction_%d'%(i+1),
                     distribution_pred[:, :, :, i:i+1],
                     max_outputs=4)

  tf.summary.image('source_image', src_img, max_outputs=4)
  tf.summary.image('target_image', trt_img, max_outputs=4)

  optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr)
  train_op = optimizer.minimize(
      loss, global_step=global_step, name='train')
  update_op = net.updates
  return Computation(tf.group([train_op, update_op]), loss, global_step)


def direction_net_translation(src_img,
                              trt_img,
                              rotation_gt,
                              translation_gt,
                              fov_gt,
                              rotation_pred,
                              derotate_both=False):
  """Build the computation graph to train the DirectionNet-T.

  Args:
    src_img: [BATCH, HEIGHT, WIDTH, 3] input source images.
    trt_img: [BATCH, HEIGHT, WIDTH, 3] input target images.
    rotation_gt: [BATCH, 3, 3] ground truth rotation matrices.
    translation_gt: [BATCH, 3] ground truth translation directions.
    fov_gt: [BATCH] the ground truth field of view (degrees) of input images.
    rotation_pred: [BATCH, 3, 3] estimated rotations from DirectionNet-R.
    derotate_both: (bool) transform both input images to a middle frame by half
      the relative rotation between them to cancel out the rotation if true.
      Otherwise, only derotate the target image to the source image's frame.

  Returns:
    A collection of tensors including training ops, loss, and global step count.
  """
  net = model.DirectionNet(1)
  global_step = tf.train.get_or_create_global_step()
  perturbed_rotation = tf.cond(
      tf.less(tf.random_uniform([], 0, 1.0), 0.5),
      lambda: util.perturb_rotation(rotation_gt, [10., 5., 10.]),
      lambda: rotation_pred)

  (transformed_src, transformed_trt) = util.derotation(
      src_img,
      trt_img,
      perturbed_rotation,
      fov_gt,
      FLAGS.transformed_fov,
      [FLAGS.transformed_height, FLAGS.transformed_width],
      derotate_both)

  (transformed_src_gt, transformed_trt_gt) = util.derotation(
      src_img,
      trt_img,
      rotation_gt,
      fov_gt,
      FLAGS.transformed_fov,
      [FLAGS.transformed_height, FLAGS.transformed_width],
      derotate_both)

  half_derotation = util.half_rotation(perturbed_rotation)
  translation_gt = tf.squeeze(tf.matmul(
      half_derotation, tf.expand_dims(translation_gt, -1),
      transpose_a=True), -1)
  translation_gt = tf.expand_dims(translation_gt, 1)
  distribution_gt = util.spherical_normalization(util.von_mises_fisher(
      translation_gt,
      tf.constant(FLAGS.kappa, tf.float32),
      [FLAGS.distribution_height, FLAGS.distribution_width]), rectify=False)

  pred = net(transformed_src, transformed_trt, training=True)
  directions, expectation, distribution_pred = util.distributions_to_directions(
      pred)

  direction_loss = losses.direction_loss(directions, translation_gt)
  distribution_loss = tf.constant(
      FLAGS.alpha, tf.float32) * losses.distribution_loss(
          distribution_pred, distribution_gt)
  spread_loss = tf.cast(
      FLAGS.beta, tf.float32) * losses.spread_loss(expectation)
  direction_error = tf.reduce_mean(tf.acos(tf.clip_by_value(
      tf.reduce_sum(directions * translation_gt, -1), -1., 1.)))

  loss = direction_loss + distribution_loss + spread_loss

  tf.summary.scalar('loss', loss)
  tf.summary.scalar('distribution_loss', distribution_loss)
  tf.summary.scalar('spread_loss', spread_loss)
  tf.summary.scalar('direction_error',
                    util.radians_to_degrees(direction_error))

  tf.summary.image('distribution/translation/ground_truth',
                   distribution_gt,
                   max_outputs=4)
  tf.summary.image('distribution/translation/prediction',
                   distribution_pred,
                   max_outputs=4)

  tf.summary.image('source_image', src_img, max_outputs=4)
  tf.summary.image('target_image', trt_img, max_outputs=4)
  tf.summary.image('transformed_source_image', transformed_src, max_outputs=4)
  tf.summary.image('transformed_target_image', transformed_trt, max_outputs=4)
  tf.summary.image(
      'transformed_source_image_gt', transformed_src_gt, max_outputs=4)
  tf.summary.image(
      'transformed_target_image_gt', transformed_trt_gt, max_outputs=4)

  optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr)
  train_op = optimizer.minimize(
      loss, global_step=global_step, name='train')
  update_op = net.updates
  return Computation(tf.group([train_op, update_op]), loss, global_step)


def direction_net_single(src_img, trt_img, rotation_gt, translation_gt):
  """Build the computation graph to train the DirectionNet-Single.

  Args:
    src_img: [BATCH, HEIGHT, WIDTH, 3] input source images.
    trt_img: [BATCH, HEIGHT, WIDTH, 3] input target images.
    rotation_gt: [BATCH, 3, 3] ground truth rotation matrices.
    translation_gt: [BATCH, 3] ground truth translation directions.

  Returns:
    A collection of tensors including training ops, loss, and global step count.
  """
  net = model.DirectionNet(4)
  global_step = tf.train.get_or_create_global_step()
  directions_gt = tf.concat([rotation_gt, translation_gt], 1)
  distribution_gt = util.spherical_normalization(util.von_mises_fisher(
      directions_gt,
      tf.constant(FLAGS.kappa, tf.float32),
      [FLAGS.distribution_height, FLAGS.distribution_width]), rectify=False)

  pred = net(src_img, trt_img, training=True)
  directions, expectation, distribution_pred = util.distributions_to_directions(
      pred)
  rotation_estimated = util.svd_orthogonalize(
      directions[:, :3])

  direction_loss = losses.direction_loss(directions, directions_gt)
  distribution_loss = tf.constant(
      FLAGS.alpha, tf.float32) * losses.distribution_loss(
          distribution_pred, distribution_gt)
  spread_loss = tf.cast(
      FLAGS.beta, tf.float32) * losses.spread_loss(expectation)
  rotation_error = tf.reduce_mean(util.rotation_geodesic(
      rotation_estimated, rotation_gt))
  translation_error = tf.reduce_mean(tf.acos(tf.clip_by_value(
      tf.reduce_sum(directions[:, -1] * directions_gt[:, -1], -1)
      , -1., 1.)))
  direction_error = tf.reduce_mean(tf.acos(tf.clip_by_value(
      tf.reduce_sum(directions * directions_gt, -1), -1., 1.)))

  loss = direction_loss + distribution_loss + spread_loss

  tf.summary.scalar('loss', loss)
  tf.summary.scalar('distribution_loss', distribution_loss)
  tf.summary.scalar('spread_loss', spread_loss)
  tf.summary.scalar('direction_error',
                    util.radians_to_degrees(direction_error))
  tf.summary.scalar('rotation_error',
                    util.radians_to_degrees(rotation_error))
  tf.summary.scalar('translation_error',
                    util.radians_to_degrees(translation_error))

  for i in range(3):
    tf.summary.image('distribution/rotation/ground_truth_%d'%(i+1),
                     distribution_gt[:, :, :, i:i+1],
                     max_outputs=4)
    tf.summary.image('distribution/rotation/prediction_%d'%(i+1),
                     distribution_pred[:, :, :, i:i+1],
                     max_outputs=4)

  tf.summary.image('distribution/translation/ground_truth',
                   distribution_gt[:, :, :, -1:],
                   max_outputs=4)
  tf.summary.image('distribution/translation/prediction',
                   distribution_pred[:, :, :, -1:],
                   max_outputs=4)

  tf.summary.image('source_image', src_img, max_outputs=4)
  tf.summary.image('target_image', trt_img, max_outputs=4)

  optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr)
  train_op = optimizer.minimize(
      loss, global_step=global_step, name='train')
  update_op = net.updates
  return Computation(tf.group([train_op, update_op]), loss, global_step)


class TimingHook(tf.train.SessionRunHook):

  def begin(self):
    self.timing_log = []

  def before_run(self, run_context):
    self.start = time.time()

  def after_run(self, run_context, run_values):
    self.timing_log.append(time.time() - self.start)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with tf.device(tf.train.replica_device_setter(ps_tasks=FLAGS.ps_tasks)):
    ds = dataset_loader.data_loader(data_path=FLAGS.data_dir,
                                    epochs=FLAGS.n_epoch,
                                    batch_size=FLAGS.batch,
                                    training=True,
                                    load_estimated_rot=FLAGS.model == 'T')
    elements = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    src_img, trt_img = elements.src_image, elements.trt_image
    rotation_gt = elements.rotation
    translation_gt = elements.translation

    print('Create computation graph.')
    if FLAGS.model == '9D':
      computation = direction_net_rotation(src_img, trt_img, rotation_gt, 3)
    elif FLAGS.model == '6D':
      computation = direction_net_rotation(src_img, trt_img, rotation_gt, 2)
    elif FLAGS.model == 'T':
      fov_gt = tf.squeeze(elements.fov, -1)
      rotation_pred = elements.rotation_pred
      computation = direction_net_translation(
          src_img, trt_img, rotation_gt, translation_gt, fov_gt, rotation_pred,
          derotate_both=FLAGS.derotate_both)
    elif FLAGS.model == 'Single':
      computation = direction_net_single(
          src_img, trt_img, rotation_gt, translation_gt)

    timing_hook = TimingHook()
    print('Create a monitored training session.')
    with tf.train.MonitoredTrainingSession(
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        hooks=[timing_hook,
               tf.train.StepCounterHook(),
               tf.train.NanTensorHook(computation.loss)],
        checkpoint_dir=FLAGS.checkpoint_dir,
        save_checkpoint_steps=2000,
        save_summaries_secs=180) as sess:
      while not sess.should_stop():
        _, loss, step = sess.run(
            [computation.train_op, computation.loss, computation.global_step])
        if step % 10 == 0:
          tf.logging.info('step = {0}, loss = {1}, time = {2}'.format(
              step, loss, timing_hook.timing_log[-1]))

if __name__ == '__main__':
  app.run(main)
