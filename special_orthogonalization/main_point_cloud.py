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

"""Training and evaluation for the point cloud alignment experiment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import utils
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
FLAGS = flags.FLAGS

# General flags.
flags.DEFINE_string('method', 'svd',
                    'Specifies the method to use for predicting rotations. '
                    'Choices are "svd", "svd-inf", or "gs".')
flags.DEFINE_string('checkpoint_dir', '',
                    'Locations for checkpoints, summaries, etc.')
flags.DEFINE_integer('train_steps', 2600000, 'Number of training iterations.')
flags.DEFINE_integer('save_checkpoints_steps', 10000,
                     'How often to save checkpoints')
flags.DEFINE_integer('log_step_count', 500, 'How often to log the step count')
flags.DEFINE_integer('save_summaries_steps', 5000,
                     'How often to save summaries.')
flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate')
flags.DEFINE_boolean('lr_decay', False, 'Decay the learning rate if True.')
flags.DEFINE_integer('lr_decay_steps', 35000,
                     'Learning rate decays steps.')
flags.DEFINE_float('lr_decay_rate', 0.95,
                   'Learning rate decays rate.')
flags.DEFINE_boolean('predict_all_test', False,
                     'If true, runs an eval job on latest checkpoint and '
                     'prints the error for each input.')
flags.DEFINE_integer('eval_examples', 0, 'Number of test examples.')
flags.DEFINE_boolean('print_variable_names', False,
                     'Print model variable names.')

# Flags only used in the point cloud alignment experiment.
flags.DEFINE_integer('num_train_augmentations', 10,
                     'Number of random rotations for augmenting each input '
                     'point cloud.')
flags.DEFINE_string('pt_cloud_train_files', '',
                    'Expression matching all training point files, e.g. '
                    '/path/to/files/pc_plane/points/*.pts')
flags.DEFINE_string('pt_cloud_test_files', '',
                    'Expression matching all modified test point files, e.g. '
                    '/path/to/files/pc_plane/points_test/*.pts')
flags.DEFINE_boolean('random_rotation_axang', True,
                     'If true, samples random rotations using the method '
                     'from the original benchmark code. Otherwise samples '
                     'by Haar measure.')


def pt_features(batch_pts):
  """Input shape: [B, N, 3], output shape: [B, 1024]."""
  with tf.variable_scope('ptenc', reuse=tf.AUTO_REUSE):
    f1 = tf.layers.conv1d(inputs=batch_pts, filters=64, kernel_size=1)
    f1 = tf.nn.leaky_relu(f1)
    f2 = tf.layers.conv1d(inputs=f1, filters=128, kernel_size=1)
    f2 = tf.nn.leaky_relu(f2)
    f3 = tf.layers.conv1d(inputs=f2, filters=1024, kernel_size=1)

  f = tf.reduce_max(f3, axis=1, keep_dims=False)
  return f


def regress_from_features(batch_features, out_dim):
  """Regress to a rotation representation from point cloud encodings.

  In Zhou et al, CVPR19, the paper describes this regression network as an MLP
  mapping 2048->512->512->out_dim, but the associated code implements it with
  one less layer: 2048->512->out_dim. We mimic the code.

  Args:
    batch_features: [batch_size, in_dim].
    out_dim: desired output dimensionality.

  Returns:
    A [batch_size, out_dim] tensor.
  """
  f1 = tf.layers.dense(batch_features, 512)
  f1 = tf.nn.leaky_relu(f1)
  f2 = tf.layers.dense(f1, out_dim)
  return f2


def net_point_cloud(points1, points2, mode):
  """Predict a relative rotation given two point clouds.

  Args:
    points1: [batch_size, N, 3] float tensor.
    points2: [batch_size, N, 3] float tensor.
    mode: tf.estimator.ModeKeys.

  Returns:
    [batch_size, 3, 3] matrices.
  """
  f1 = pt_features(points1)
  f2 = pt_features(points2)
  f = tf.concat([f1, f2], axis=-1)

  if FLAGS.method == 'svd':
    p = regress_from_features(f, 9)
    return utils.symmetric_orthogonalization(p)

  if FLAGS.method == 'svd-inf':
    p = regress_from_features(f, 9)
    if mode == tf.estimator.ModeKeys.TRAIN:
      return tf.reshape(p, (-1, 3, 3))
    else:
      return utils.symmetric_orthogonalization(p)

  if FLAGS.method == 'gs':
    p = regress_from_features(f, 6)
    return utils.gs_orthogonalization(p)


def model_fn(features, labels, mode, params):
  """The model_fn used to construct the tf.Estimator."""
  del labels, params  # Unused.
  if mode == tf.estimator.ModeKeys.TRAIN:
    # Training data has point cloud of size [1, N, 3] and random rotations
    # of size [1, FLAGS.num_train_augmentations, 3, 3]
    rot = features['rot'][0]
    num_rot = FLAGS.num_train_augmentations
    batch_pts1 = tf.tile(features['data'], [num_rot, 1, 1])
    # In this experiment it does not matter if we pre or post-multiply the
    # rotation as long as we are consistent between training and eval.
    batch_pts2 = tf.matmul(batch_pts1, rot)  # post-multiplying!
  else:
    # Test data has point cloud of size [1, N, 3] and a single random
    # rotation of size [1, 3, 3]
    batch_pts1 = features['data']
    rot = features['rot']
    batch_pts2 = tf.matmul(batch_pts1, rot)
  rot = tf.reshape(rot, (-1, 3, 3))

  # Predict the rotation.
  r = net_point_cloud(batch_pts1, batch_pts2, mode)

  # Compute the loss.
  loss = tf.nn.l2_loss(rot - r)

  # Compute the relative angle in radians.
  theta = utils.relative_angle(r, rot)

  # Mean angle error over the batch.
  mean_theta = tf.reduce_mean(theta)
  mean_theta_deg = mean_theta * 180.0 / np.pi

  # Train, eval, or predict depending on mode.
  if mode == tf.estimator.ModeKeys.TRAIN:
    tf.summary.scalar('train/loss', loss)
    tf.summary.scalar('train/theta', mean_theta_deg)
    global_step = tf.train.get_or_create_global_step()

    if FLAGS.lr_decay:
      learning_rate = tf.train.exponential_decay(
          FLAGS.learning_rate,
          global_step,
          FLAGS.lr_decay_steps,
          FLAGS.lr_decay_rate)
    else:
      learning_rate = FLAGS.learning_rate

    tf.summary.scalar('train/learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)

  if mode == tf.estimator.ModeKeys.EVAL:
    if FLAGS.predict_all_test:
      print_error_op = tf.print('error:', mean_theta_deg)
      with tf.control_dependencies([print_error_op]):
        eval_metric_ops = {
            'mean_degree_err': tf.metrics.mean(mean_theta_deg),
        }
    else:
      eval_metric_ops = {
          'mean_degree_err': tf.metrics.mean(mean_theta_deg),
      }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)

  if mode == tf.estimator.ModeKeys.PREDICT:
    pred = {'error': mean_theta_deg}
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred)


def train_input_fn():
  """Generate training data iterator from the .pts files."""
  def _file_to_matrix(pts_path):
    """Read Nx3 point cloud from a .pts file."""
    file_buffer = tf.read_file(pts_path)
    lines = tf.string_split([file_buffer], delimiter='\n')
    values = tf.stack(tf.decode_csv(lines.values,
                                    record_defaults=[[0.0], [0.0], [0.0]],
                                    field_delim=' '))
    values = tf.transpose(values)  # 3xN --> Nx3.
    # The experiment code in
    # github.com/papagina/RotationContinuity/.../shapenet/code/train_pointnet.py
    # only used the first half of the points in each file.
    return values[:(tf.shape(values)[0] // 2), :]

  def _random_rotation(pts):
    """Attach N random rotations to a point cloud."""
    if FLAGS.random_rotation_axang:
      rotations = utils.random_rotation_benchmark(FLAGS.num_train_augmentations)
    else:
      rotations = utils.random_rotation(FLAGS.num_train_augmentations)
    return pts, rotations

  pts_paths = tf.gfile.Glob(FLAGS.pt_cloud_train_files)
  dataset = tf.data.Dataset.from_tensor_slices(pts_paths)
  dataset = dataset.map(_file_to_matrix)
  dataset = dataset.cache()  # Comment out if memory cannot hold all the data.
  dataset = dataset.shuffle(buffer_size=50, reshuffle_each_iteration=True)
  dataset = dataset.repeat()
  dataset = dataset.map(_random_rotation)
  dataset = dataset.batch(1)
  iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
  batch_data, batch_rot = iterator.get_next()
  features_dict = {'data': batch_data, 'rot': batch_rot}
  batch_size = tf.shape(batch_data)[0]
  batch_labels_dummy = tf.zeros(shape=(batch_size, 1))
  return (features_dict, batch_labels_dummy)


def eval_input_fn():
  """Generate test data from *modified* .pts files.

  See README and comments below for details on how the data is modified.

  Returns:
    A tuple of features and associated labels.
  """
  def _file_to_matrix(pts_path):
    """Read Nx3 point cloud and 3x3 rotation matrix from a .pts file.

    The test data is a modified version of the original files. For each .pts
    file we have (1) added a 3x3 rotation matrix for testing, and (2) removed
    the second half of the point cloud since it is not used at all.

    Args:
      pts_path: path to a .pts file.

    Returns:
      A Nx3 point cloud.
      A 3x3 rotation matrix.
    """
    file_buffer = tf.read_file(pts_path)
    lines = tf.string_split([file_buffer], delimiter='\n')
    values = tf.stack(tf.decode_csv(lines.values,
                                    record_defaults=[[0.0], [0.0], [0.0]],
                                    field_delim=' '))
    values = tf.transpose(values)  # 3xN --> Nx3.
    # First three rows are the rotation matrix, remaining rows the point cloud.
    rot = values[:3, :]
    return values[4:, :], rot

  pts_paths = tf.gfile.Glob(FLAGS.pt_cloud_test_files)
  dataset = tf.data.Dataset.from_tensor_slices(pts_paths)
  dataset = dataset.map(_file_to_matrix)
  dataset = dataset.batch(1)
  iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
  batch_data, batch_rot = iterator.get_next()
  features_dict = {'data': batch_data, 'rot': batch_rot}
  batch_size = tf.shape(batch_data)[0]
  batch_labels_dummy = tf.zeros(shape=(batch_size, 1))
  return (features_dict, batch_labels_dummy)


def print_variable_names():
  """Print variable names in a model."""
  params = {'dummy': 0}
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=FLAGS.checkpoint_dir,
      params=params)
  names = estimator.get_variable_names()
  for name in names:
    print(name)


def predict_all_test():
  """Print error statistics for the test dataset."""
  params = {'dummy': 0}
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=FLAGS.checkpoint_dir,
      params=params)
  evals = estimator.predict(input_fn=eval_input_fn, yield_single_examples=False)

  # Print error statistics.
  all_errors = [x['error'] for x in evals]
  errors = np.array(all_errors)
  print('Evaluated %d examples'%np.size(errors))
  print('Mean error: %f degrees', np.mean(errors))
  print('Median error: %f degrees', np.median(errors))
  print('Std: %f degrees', np.std(errors))
  sorted_errors = np.sort(errors)
  n = np.size(sorted_errors)
  print('\nPercentiles:')
  for perc in range(1, 101):
    index = np.int32(np.float32(n * perc) / 100.0) - 1
    print('%3d%%: %f'%(perc, sorted_errors[index]))


def train_and_eval():
  """Train and evaluate a model."""
  save_summary_steps = FLAGS.save_summaries_steps
  save_checkpoints_steps = FLAGS.save_checkpoints_steps
  log_step_count = FLAGS.log_step_count

  config = tf.estimator.RunConfig(
      save_summary_steps=save_summary_steps,
      save_checkpoints_steps=save_checkpoints_steps,
      log_step_count_steps=log_step_count,
      keep_checkpoint_max=None)

  params = {'dummy': 0}
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=FLAGS.checkpoint_dir,
      config=config,
      params=params)

  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      max_steps=FLAGS.train_steps)

  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                    start_delay_secs=60,
                                    steps=FLAGS.eval_examples,
                                    throttle_secs=60)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.print_variable_names:
    print_variable_names()
    return

  if FLAGS.predict_all_test:
    predict_all_test()
  else:
    train_and_eval()


if __name__ == '__main__':
  tf.compat.v1.app.run()
