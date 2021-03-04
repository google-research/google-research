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

"""Cycle consistency across sets for Shapes3D factor isolation experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import datetime
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
from shapes3d import evaluation
from shapes3d import loss_fns
import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS


#################################### I/O #######################################
flags.DEFINE_string('outdir',
                    'shapes3d_output/',
                    'The directory in which to save results and images.')
flags.DEFINE_integer('eval_frequency', 500,
                     'The number of iterations between tensorboard outputs.')
flags.DEFINE_integer('save_frequency', 1000,
                     'The number of iterations between saving checkpoints.')
flags.DEFINE_bool('save_pngs', False, 'Whether to save the eval pngs.')
flags.DEFINE_bool('save_model', False, 'Whether to save the model at the end.')
flags.DEFINE_integer('visualization_size', 256,
                     'The number of example embeddings to display in outputs.')
################################ Model Specs ###################################
flags.DEFINE_integer('num_latent_dims', 2,
                     'The size of the latent space for the embeddings. '
                     'The mutual information and visualization outputs are '
                     'only for num_latent_dims = 2, but training can be with '
                     'any dimension.')
#################################### Data ######################################
flags.DEFINE_integer('stack_size',
                     32,
                     'The number of examples per stack.')
flags.DEFINE_string('inactive_vars', '1',
                    'A string with any subset of 012345, one for each of the '
                    '6 generative factors of shapes3d.  E.g. \'04\' holds the '
                    'wall hue and shape inactive.  Can be empty for '
                    'unconstrained sets.')
flags.DEFINE_bool('curate_both_stacks', True,
                  'As in Figure 3 of the manuscript, the default case (True) is'
                  ' where both sets for each training batch have the same '
                  'inactive factors of variation.  If False, the second set is'
                  ' unconstrained.')
flags.DEFINE_bool('run_augmentation_experiment', False, 'Whether to run the '
                  'double augmentation baseline, where each image is augmented '
                  'twice by hue and training is through a contrastive learning '
                  'loss.')
################################# Training #####################################
flags.DEFINE_integer('num_iters',
                     2000,
                     'The number of iterations to train.')
flags.DEFINE_float('learning_rate',
                   3e-5,
                   'The learning rate for the optimizer.')
flags.DEFINE_string('optimizer', 'Adam', 'The keras optimizer to use.')
flags.DEFINE_float('temperature', 1.,
                   'The temperature to use in the cycle consistency loss, both'
                   ' for the soft nearest neighbor calculation and the cross-'
                   'entropy loss on the way back.')
flags.DEFINE_string('similarity_type', 'l2sq',
                    'The metric for computing distances in embedding space.'
                    ' Implemented: l1, l2, l2sq, linf, cosine.')
################################################################################
################################################################################


def main(_):
  ##############################################################################
  # Parse flags
  ##############################################################################
  outdir = FLAGS.outdir
  stack_size = FLAGS.stack_size
  num_iters = FLAGS.num_iters
  learning_rate = FLAGS.learning_rate
  optimizer_name = FLAGS.optimizer
  inactive_vars = FLAGS.inactive_vars
  inactive_vars = [int(digit) for digit in inactive_vars]
  visualization_size = FLAGS.visualization_size
  temperature = FLAGS.temperature
  similarity_type = FLAGS.similarity_type
  num_curated_stacks = int(FLAGS.curate_both_stacks) + 1

  num_latent_dims = FLAGS.num_latent_dims
  eval_frequency = FLAGS.eval_frequency
  save_frequency = FLAGS.save_frequency
  save_pngs = FLAGS.save_pngs
  save_model = FLAGS.save_model

  run_augmentation_experiment = FLAGS.run_augmentation_experiment
  if run_augmentation_experiment:
    inactive_vars = []
    num_curated_stacks = 1

  ##############################################################################
  # Load the data
  ##############################################################################
  inactive_vars_num = [
      evaluation.GENERATIVE_FACTORS[var_id][1] for var_id in inactive_vars
  ]

  dataset_full = tfds.load('shapes3d', split='train')

  def parse_example(example):
    image = tf.image.convert_image_dtype(example['image'], tf.float32)
    gen_factors = [example['label_' + factor_name] for
                   factor_name, _ in evaluation.GENERATIVE_FACTORS]
    gen_factors = tf.stack(gen_factors, 0)
    return image, gen_factors

  dataset_full = dataset_full.map(
      parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # These will be used for tensorboard output images showing sample embeddings
  visualization_images, visualization_labels = [[], []]
  for image_stack, labels in dataset_full.batch(32).take(visualization_size //
                                                         32):
    visualization_images.append(image_stack)
    visualization_labels.append(labels)
  visualization_images = tf.concat(visualization_images, 0)
  visualization_labels = tf.concat(visualization_labels, 0)

  ##############################################################################
  # Build the model and optimizer
  ##############################################################################
  model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(64, 64, 3)),
      tf.keras.layers.Conv2D(32, 3, activation='relu', strides=1),
      tf.keras.layers.Conv2D(32, 3, activation='relu', strides=1),
      tf.keras.layers.Conv2D(64, 3, activation='relu', strides=2),
      tf.keras.layers.Conv2D(64, 3, activation='relu', strides=1),
      tf.keras.layers.Conv2D(128, 3, activation='relu', strides=1),
      tf.keras.layers.Conv2D(128, 3, activation='relu', strides=2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, 'relu'),
      tf.keras.layers.Dense(num_latent_dims),
  ])

  optimizer = tf.keras.optimizers.get(optimizer_name)
  optimizer.learning_rate = learning_rate

  ##############################################################################
  # Set up logging and checkpointing
  ##############################################################################
  base_log_dir = os.path.join(outdir, 'logs')
  if os.path.exists(base_log_dir):
    train_log_dir = os.path.join(base_log_dir,
                                 os.listdir(base_log_dir)[0])
  else:
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = os.path.join(outdir, 'logs', current_time)
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)

  checkpoint_dir = os.path.join(outdir, 'checkpoints')
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint, directory=checkpoint_dir, max_to_keep=1)

  ##############################################################################
  # Train
  ##############################################################################
  train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
  logging.info('Started training with [%s] '
               'inactive', ', '.join([
                   evaluation.GENERATIVE_FACTORS[i][0] for i in inactive_vars
               ]))
  dataset_full_iterator = iter(
      dataset_full.repeat().shuffle(1000).batch(stack_size))

  if inactive_vars:
    def matching_inactive_vars(labels, label_values):
      return tf.reduce_all(
          tf.equal(tf.gather(labels, inactive_vars), label_values))

  for _ in range(num_iters):
    iteration = optimizer.iterations.numpy()
    if iteration > num_iters:
      break

    # Forge a new stack each step with random values for the inactive factors
    image_stacks = []
    for _ in range(num_curated_stacks):
      label_values = [
          np.random.randint(low=0, high=val) for val in inactive_vars_num
      ]
      if inactive_vars:
        dataset_partial = dataset_full.filter(
            lambda _, labels: matching_inactive_vars(labels, label_values))  # pylint: disable=cell-var-from-loop
        dataset_partial = iter(dataset_partial.shuffle(1000).batch(stack_size))
      else:
        dataset_partial = dataset_full_iterator
      image_stacks.append(next(dataset_partial)[0])
    if num_curated_stacks == 1:
      image_stacks.append(next(dataset_full_iterator)[0])
    if run_augmentation_experiment:
      image_stacks = [
          tf.image.random_hue(image_stacks[0], 0.5),
          tf.image.random_hue(image_stacks[0], 0.5)
      ]

    with tf.GradientTape() as tape:
      embeddings1 = model(image_stacks[0], training=True)
      embeddings2 = model(image_stacks[1], training=True)

      if run_augmentation_experiment:
        logits = loss_fns.get_scaled_similarity(embeddings1, embeddings2,
                                                similarity_type, temperature)
        labels = tf.one_hot(tf.range(stack_size), stack_size)
        loss = loss_fns.classification_loss(logits, labels)
      else:
        # Standard cycle consistency across sets
        logits, labels = loss_fns.quantify_unambiguous_cycles(
            embeddings1, embeddings2, similarity_type, temperature)
        loss = loss_fns.classification_loss(logits, labels)
        logits, labels = loss_fns.quantify_unambiguous_cycles(
            embeddings2, embeddings1, similarity_type, temperature)
        loss += loss_fns.classification_loss(logits, labels)

    grads = tape.gradient(loss, model.trainable_variables)
    train_loss(loss)
    # Eval before updating the variables to eval the randomly initialized model
    if iteration % eval_frequency == 0:
      logging.info('Started eval, step %d', iteration)
      avg_loss = train_loss.result()
      # Everything but loss is implemented only for 2-dimensional latent space
      if num_latent_dims == 2:
        entropy_y, mutual_infos_all = evaluation.compute_mutual_info(
            model, dataset_full)

        csv_outfile = os.path.join(outdir, 'mutual_info_results.csv')
        to_write = [str(iteration), '{:.4f}'.format(entropy_y)] + [
            '{:.4f}'.format(mutual_info) for mutual_info in mutual_infos_all
        ]
        with open(csv_outfile, 'a') as csvfile:
          results_writer = csv.writer(csvfile, delimiter=',')
          results_writer.writerow(to_write)

        if save_pngs:
          out_fname = os.path.join(outdir, '{}.png'.format(iteration))
          visualization_embeddings = model(visualization_images, training=False)
          evaluation.visualize_embeddings(mutual_infos_all,
                                          visualization_embeddings,
                                          visualization_labels, out_fname)
      with train_summary_writer.as_default():
        tf.summary.scalar('loss', avg_loss, step=iteration)
        if num_latent_dims == 2:
          tf.summary.scalar('entropy_y', entropy_y, step=iteration)
          for i in range(len(mutual_infos_all)):
            tf.summary.scalar(
                'mutual_info_{}'.format(evaluation.GENERATIVE_FACTORS[i][0]),
                mutual_infos_all[i],
                step=iteration)

      train_loss.reset_states()

    if iteration and iteration % save_frequency == 0:
      checkpoint_manager.save()

    # Finally, update the variables
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

  logging.info('Finished training.')
  if save_model:
    model_out_fname = os.path.join(outdir, 'model_shapes3d')
    model.save(model_out_fname)

if __name__ == '__main__':
  app.run(main)



