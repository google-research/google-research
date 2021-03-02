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

# Lint as: python3
"""Training code for learning policies using imitation learning on Jumpy World."""

import collections
import os
import os.path as osp

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

from pse.jumping_task import data_helpers
from pse.jumping_task import evaluation_helpers
from pse.jumping_task import model_helpers
from pse.jumping_task import training_helpers


FLAGS = flags.FLAGS
flags.DEFINE_integer('training_epochs', None,
                     'The number of epochs to run training for.')
# Hyperparameters
flags.DEFINE_integer('batch_size', 256, 'Hyperparameter: batch size.')
flags.DEFINE_float('learning_rate', 1e-2, 'Hyperparameter: learning rate.')
flags.DEFINE_float('l2_reg', 0.0, 'Hyperparameter: L2 regularization')
flags.DEFINE_float('decay_rate', 0.999, 'Hyperparameter: LR decay rate')
flags.DEFINE_float('dropout', 0.0, 'Hyperparameter: Dropout')
# Contrastive Loss related hyperparameters
flags.DEFINE_float('alpha', 1.0, 'Hyperparameter: Alignment loss coefficient')
flags.DEFINE_float(
    'temperature', 1.0, 'Hyperparameter: Temperature used in '
    'contrastive (NT-Xent) loss')
flags.DEFINE_float(
    'soft_coupling_temperature', 1.0, 'Hyperparameter: Temperature used in '
    'computing the soft coupling values')
flags.DEFINE_integer(
    'use_coupling_weights', 0, 'Whether to use the coupling '
    'weights for weighting the positive and negative examples')
# Only useful for finding the best solution with additional information
flags.DEFINE_bool(
    'ground_truth_coupling', False, 'Whether to use the ground '
    'truth coupling or not when computing the contrastive loss.')
# Imitation data helpers
flags.DEFINE_integer(
    'min_obstacle_grid', 20, 'Minimum obstacle position in'
    'JumpyWorld environments. Must be >= 14.')
flags.DEFINE_integer(
    'max_obstacle_grid', 45, 'Minimum obstacle position in'
    'JumpyWorld environments. Must be <= 48')
flags.DEFINE_integer(
    'min_floor_grid', 10, 'Minimum floor height in'
    'JumpyWorld environments. Must be >= 0.')
flags.DEFINE_integer(
    'max_floor_grid', 20, 'Maximum floor height in'
    'JumpyWorld environments. Must be <= 41.')
flags.DEFINE_integer(
    'min_obstacle_position', 20, 'Minimum obstacle position in'
    'JumpyWorld seen during training. Must be >= 14.')
flags.DEFINE_integer(
    'max_obstacle_position', 45, 'Minimum obstacle position in'
    'JumpyWorld seen during training. Must be <= 48')
flags.DEFINE_integer(
    'min_floor_height', 10, 'Minimum floor height in'
    'JumpyWorld seen during training. Must be >= 0.')
flags.DEFINE_integer(
    'max_floor_height', 20, 'Maximum floor height in'
    'JumpyWorld seen during training. Must be <= 41.')
flags.DEFINE_integer(
    'positions_train_diff', 5, 'Number of obstacle '
    'positions seen during training. The positions uniformly '
    'divide [min_obstacle_position, max_obstacle_position].')
flags.DEFINE_integer(
    'heights_train_diff', 5, 'Number of obstacle '
    'positions seen during training. The positions uniformly '
    'divide [min_obstacle_position, max_obstacle_position].')
# Additional flags
flags.DEFINE_string(
    'train_dir', None, 'Directory in which the training '
    'checkpoints and tensorboard summaries are saved')
flags.DEFINE_integer(
    'max_checkpoints_to_keep', 1, 'Indicates the maximum '
    'number of recent checkpoint files to keep.')
flags.DEFINE_integer(
    'save_checkpoint_every_n_epochs', 40, 'Indicates the '
    'number of epochs after which an checkpoint is saved')
flags.DEFINE_integer(
    'evaluate_every_n_epochs', 20, 'Indicates the number of epochs'
    'after which the agent is evaluated.')
flags.DEFINE_bool('debugging', False,
                  'Turns on additional logging for debugging.')
flags.DEFINE_bool('show_alignment_loss_image', False,
                  'Turns on tensorboard summaries displaying the coupling '
                  'and the cost matrix for the alignment loss.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('exp_num', 1, 'Experiment number for sweeps.')
flags.DEFINE_bool('random_tasks', False,
                  'Whether to use random tasks for training or not.')
flags.DEFINE_bool('rand_conv', False, 'Whether to use rand conv or not.')
flags.DEFINE_bool('use_colors', False, 'Whether to use colored obstacles. Red '
                  'blocks behave like white obstacles while green obstacles '
                  'behave differently.')
flags.DEFINE_bool('no_validation', False, 'Validation set or not.')
flags.DEFINE_bool('projection', True, 'Whether to project the embedding before'
                  'learning representations or not.')
flags.DEFINE_bool('use_l2_loss', False, 'Whether to use l2 loss for learning '
                  'embeddings or the contrastive loss.')
flags.DEFINE_bool('use_bisim', False, 'Whether to use pi*-bisimulation metric'
                  'instead of the action similarity metric.')


@flags.multi_flags_validator(['use_l2_loss', 'projection'],
                             message='L2 loss for embedding should not be used '
                             'in conjunction with projection.')
def l2_loss_and_projection(flags_dict):
  return not (flags_dict['use_l2_loss'] and flags_dict['projection'])




def train_step(nn_model,
               x,
               y,
               optimizer,
               optimal_data_tuple=None,
               alpha=0.0,
               l2_reg=0.0,
               ground_truth_coupling=False,
               use_coupling_weights=False,
               temperature=1.0,
               debugging=False):
  """Take a training step."""
  total_loss, losses = 0.0, {}
  with tf.GradientTape() as tape:
    tape.watch(nn_model.trainable_variables)
    if not debugging:
      cross_entropy_loss = training_helpers.cross_entropy_loss(
          nn_model, x, y, training=True)
      losses['cross_entropy_loss'] = cross_entropy_loss
      total_loss += cross_entropy_loss
    if l2_reg > 0:
      l2_regularization_loss = training_helpers.weight_decay(nn_model)
      losses['l2_regularization_loss'] = l2_regularization_loss
      total_loss += l2_reg * l2_regularization_loss
    if alpha > 0:
      alignment_loss, _, _ = training_helpers.representation_alignment_loss(
          nn_model,
          optimal_data_tuple=optimal_data_tuple,
          use_bisim=FLAGS.use_bisim,
          ground_truth=ground_truth_coupling,
          gamma=0.999,
          use_l2_loss=FLAGS.use_l2_loss,
          use_coupling_weights=use_coupling_weights,
          coupling_temperature=FLAGS.soft_coupling_temperature,
          temperature=temperature)
      losses['alignment_loss'] = alignment_loss
      total_loss += alpha * alignment_loss
  losses['total_loss'] = total_loss
  grads = tape.gradient(total_loss, nn_model.trainable_variables)
  optimizer.apply_gradients(zip(grads, nn_model.trainable_variables))
  return losses


def train_agent(train_dir, measurements=None):
  """Training Loop."""
  nn_model = model_helpers.JumpyWorldNetwork(
      num_actions=2, dropout=float(FLAGS.dropout), rand_conv=FLAGS.rand_conv,
      projection=FLAGS.projection)
  learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
  optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
  # Imitation Data Generation
  imitation_data = data_helpers.generate_imitation_data(
      min_obstacle_position=FLAGS.min_obstacle_grid,
      max_obstacle_position=FLAGS.max_obstacle_grid,
      min_floor_height=FLAGS.min_floor_grid,
      max_floor_height=FLAGS.max_floor_grid,
      use_colors=FLAGS.use_colors)

  training_positions = data_helpers.generate_training_positions(
      min_obstacle_position=FLAGS.min_obstacle_position,
      max_obstacle_position=FLAGS.max_obstacle_position,
      min_floor_height=FLAGS.min_floor_height,
      max_floor_height=FLAGS.max_floor_height,
      positions_train_diff=FLAGS.positions_train_diff,
      heights_train_diff=FLAGS.heights_train_diff,
      random_tasks=FLAGS.random_tasks,
      seed=FLAGS.seed)

  if FLAGS.no_validation:
    validation_positions = []  # Pass an empty list of positions
  else:
    # Generate validation positions depending on grid configuration.
    num_positions = FLAGS.max_obstacle_grid - FLAGS.min_obstacle_grid + 1
    num_heights = FLAGS.max_floor_grid - FLAGS.min_floor_grid + 1
    position_span = FLAGS.max_obstacle_position - FLAGS.min_obstacle_position
    is_tight_grid = (not FLAGS.random_tasks) and (position_span <= 12)
    if is_tight_grid:
      extra_training_positions = data_helpers.generate_training_positions(
          min_obstacle_position=FLAGS.min_obstacle_position -
          FLAGS.positions_train_diff,
          max_obstacle_position=FLAGS.max_obstacle_position,
          min_floor_height=FLAGS.min_floor_height,
          max_floor_height=FLAGS.max_floor_height,
          positions_train_diff=FLAGS.positions_train_diff,
          heights_train_diff=FLAGS.heights_train_diff,
          random_tasks=FLAGS.random_tasks,
          seed=FLAGS.seed)
      validation_positions = data_helpers.generate_validation_tight_grid(
          extra_training_positions,
          pos_diff=FLAGS.positions_train_diff,
          height_diff=FLAGS.heights_train_diff,
          min_obstacle_position=FLAGS.min_obstacle_grid,
          max_obstacle_position=FLAGS.max_obstacle_grid,
          min_floor_height=FLAGS.min_floor_grid,
          max_floor_height=FLAGS.max_floor_grid)
    else:
      validation_positions = evaluation_helpers.generate_validation_positions(
          training_positions, FLAGS.min_obstacle_grid,
          FLAGS.min_floor_grid, num_positions, num_heights)

  x_train, y_train = data_helpers.training_data(imitation_data,
                                                training_positions)
  ds_tensors = training_helpers.create_balanced_dataset(x_train, y_train,
                                                        FLAGS.batch_size)
  # tf.config.experimental_run_functions_eagerly(True)
  if FLAGS.rand_conv:
    _ = nn_model.rand_conv.rand_output(x_train[:1])

  ckpt_manager = model_helpers.create_checkpoint_manager(
      nn_model,
      ckpt_dir=osp.join(train_dir, 'model'),
      step=tf.Variable(1, trainable=False),
      optimizer=optimizer,
      restore=True)
  # Log summaries for the training and validation results
  summary_writer = tf.summary.create_file_writer(
      osp.join(train_dir, 'tb_log'), flush_millis=5000)
  avg_losses = {
      name: tf.keras.metrics.Mean(name=name, dtype=tf.float32) for name in [
          'total_loss', 'cross_entropy_loss', 'l2_regularization_loss',
          'alignment_loss'
      ]
  }

  num_iters_per_epoch = (len(x_train) // FLAGS.batch_size) + 1
  save_ckpt_iters = FLAGS.save_checkpoint_every_n_epochs * num_iters_per_epoch
  eval_iters = FLAGS.evaluate_every_n_epochs * num_iters_per_epoch
  alpha, l2_reg = float(FLAGS.alpha), float(FLAGS.l2_reg)
  # Monte-Carlo averaging for RandConv
  eval_mc_samples = 5 if FLAGS.rand_conv else 1
  if FLAGS.use_colors:
    data_for_tuple_generation = imitation_data['RED']
  else:
    data_for_tuple_generation = imitation_data['WHITE']
  with summary_writer.as_default():
    for x, y in ds_tensors:
      if FLAGS.alpha > 0:
        optimal_data_tuple = data_helpers.generate_optimal_data_tuple(
            data_for_tuple_generation, training_positions, print_log=False)
      else:
        optimal_data_tuple = None

      losses = train_step(
          nn_model,
          x,
          y,
          optimizer,
          optimal_data_tuple=optimal_data_tuple,
          l2_reg=l2_reg,
          alpha=alpha,
          ground_truth_coupling=FLAGS.ground_truth_coupling,
          use_coupling_weights=FLAGS.use_coupling_weights,
          temperature=FLAGS.temperature,
          debugging=FLAGS.debugging)
      # Log summaries
      for loss_name, loss_val in losses.items():
        avg_losses[loss_name].update_state(loss_val)
      if optimizer.iterations % num_iters_per_epoch == 0:
        learning_rate.assign(learning_rate * FLAGS.decay_rate)
        tf.summary.scalar(
            'learning_rate', learning_rate, step=optimizer.iterations)
        for loss_name in losses:
          tf.summary.scalar(
              'loss/{}'.format(loss_name),
              avg_losses[loss_name].result(),
              step=optimizer.iterations)
          avg_losses[loss_name].reset_states()

        if optimizer.iterations % save_ckpt_iters == 0:
          ckpt_manager.save()
        if optimizer.iterations % eval_iters == 0:
          logging.info('Epoch: %d', optimizer.iterations // num_iters_per_epoch)

          solved_envs = collections.defaultdict(int)
          for color_name, imitation_color_data in imitation_data.items():
            eval_grid = evaluation_helpers.create_evaluation_grid(
                nn_model, imitation_color_data, mc_samples=eval_mc_samples,
                color_name=color_name)

            eval_grid_plot = evaluation_helpers.plot_evaluation_grid(
                eval_grid, training_positions, FLAGS.min_obstacle_grid,
                FLAGS.min_floor_grid)
            eval_grid_image = evaluation_helpers.plot_to_image(eval_grid_plot)
            tf.summary.image(
                f'Grid/Evaluation/{color_name}',
                eval_grid_image,
                step=optimizer.iterations)

            solved_envs_color = evaluation_helpers.num_solved_tasks(
                eval_grid, training_positions, validation_positions,
                FLAGS.min_obstacle_grid, FLAGS.min_floor_grid)
            for split_name, num_solved in solved_envs_color.items():
              solved_envs[split_name] += num_solved
              color_key = f'{split_name}_{color_name}'
              if color_name != 'WHITE':
                tf.summary.scalar(
                    name=f'eval/{color_key}_solved',
                    data=num_solved,
                    step=optimizer.iterations)
                if measurements and color_key in measurements:
                  measurements[color_key].create_measurement(
                      objective_value=num_solved,
                      step=optimizer.iterations.numpy() // num_iters_per_epoch)

          for key, num_solved in solved_envs.items():
            tf.summary.scalar(
                name='eval/{}_solved'.format(key),
                data=num_solved,
                step=optimizer.iterations)
            if measurements and key in measurements:
              measurements[key].create_measurement(
                  objective_value=num_solved,
                  step=optimizer.iterations.numpy() // num_iters_per_epoch)

          if FLAGS.alpha > 0 and FLAGS.show_alignment_loss_image:
            # Log the coupling and the cost matrix
            _, coupling_cost, similarity_matrix = training_helpers.representation_alignment_loss(
                nn_model,
                optimal_data_tuple=optimal_data_tuple,
                ground_truth=FLAGS.ground_truth_coupling,
                use_bisim=FLAGS.use_bisim,
                gamma=0.999,
                use_l2_loss=FLAGS.use_l2_loss,
                coupling_temperature=FLAGS.soft_coupling_temperature,
                temperature=FLAGS.temperature)
            tf.summary.image(
                name='align/coupling_cost',
                data=evaluation_helpers.np_array_figure(coupling_cost.numpy()),
                step=optimizer.iterations)
            tf.summary.image(
                name='align/similarity_matrix',
                data=evaluation_helpers.np_array_figure(
                    similarity_matrix.numpy()),
                step=optimizer.iterations)
            if FLAGS.debugging:
              learned_coupling = evaluation_helpers.induced_coupling(
                  similarity_matrix)
              tf.summary.image(
                  name='align/learned_coupling',
                  data=evaluation_helpers.np_array_figure(
                      learned_coupling.numpy()),
                  step=optimizer.iterations)

      if optimizer.iterations > (num_iters_per_epoch * FLAGS.training_epochs):
        break


def limit_gpu_memory_growth():
  """Allocate a subset of the available GPU memory to tensorflow."""
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)


def set_random_seed():
  """Set random seed for reproducibility."""
  os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)


def main(argv):
  _ = argv
  set_random_seed()
  train_dir, measurements = FLAGS.train_dir, None
  limit_gpu_memory_growth()
  train_agent(train_dir, measurements)


if __name__ == '__main__':
  flags.mark_flag_as_required('train_dir')
  flags.mark_flag_as_required('training_epochs')
  app.run(main)
