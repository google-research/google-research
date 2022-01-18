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

"""Runs a pose estimation experiment on the symmetric_solids dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
from implicit_pdf import data
from implicit_pdf import evaluation
from implicit_pdf import models

tfkl = tf.keras.layers
FLAGS = flags.FLAGS

#################################### I/O #######################################
flags.DEFINE_string('outdir',
                    'ipdf_output/',
                    'The directory in which to save results and images.')
flags.DEFINE_bool('save_models', True, 'Whether to save the vision and IPDF'
                  ' models at the end of training.')
################################ Model Specs ###################################
flags.DEFINE_multi_integer('head_network_specs',
                           [256]*2,
                           'The sizes of the dense layers in the head network.')
#################################### Data ######################################
flags.DEFINE_multi_string('symsol_shapes', ['tet'],
                          'Can be any subset of the 8 shapes of SYMSOL I & II: '
                          'tet, cube, icosa, cyl, cone, tetX, cylO, sphereX, or'
                          ' \'symsol1\' for the first five.')
flags.DEFINE_integer('downsample_continuous_gt', 0,
                     'Whether, and how much, to downsample the cone and '
                     'cylinder ground truth rotations, which can make '
                     'evaluation slow.')
################################# Training #####################################
flags.DEFINE_integer('number_training_iterations',
                     10_000,
                     'The number of iterations to train.')
flags.DEFINE_integer('number_eval_iterations', None,
                     'The number of iterations to eval.')

flags.DEFINE_float('learning_rate', 1e-4, 'The learning rate.')
flags.DEFINE_integer('batch_size', 32, 'The batch size.')
flags.DEFINE_integer('test_batch_size', 32, 'The batch size for evaluation, '
                     'where it may be helpful to evaluate with a larger grid '
                     'and reduce this batch size if memory issues arise.')
flags.DEFINE_string('optimizer', 'Adam', 'The name of the optimizer to use.')
flags.DEFINE_integer('number_train_queries', 2**12,
                     'The number of sampled points on SO(3) for each loss '
                     'evaluation.')
flags.DEFINE_integer('number_eval_queries', 2**16,
                     'The number of sampled points on SO(3) to use for '
                     'evaluation.')
flags.DEFINE_enum('so3_sampling_mode', 'random',
                  ['random', 'grid'],
                  'How to sample from SO(3): \'random\' samples rotations '
                  'uniformly and \'grid\' creates an equivolumetric grid based '
                  'off Yershova et al. (2010).')
flags.DEFINE_integer('number_fourier_components', 1,
                     'The number of components in the positional encoding '
                     'for the implicit model.')
flags.DEFINE_integer('eval_every', -1, 'How often to evaluate.  If -1, evaluate'
                     '  100 times during training.')
flags.DEFINE_bool('skip_spread_evaluation', False, 'Whether to skip the '
                  'evaluation of the spread metric, which can be slow for '
                  'shapes with many ground truths.')
################################################################################
################################################################################

flags.DEFINE_bool('mock', False,
                  'Skip download of dataset and pre-trained weights. '
                  'Useful for testing.')


def main(_):

  outdir = FLAGS.outdir
  checkpoint_dir = os.path.join(outdir, 'checkpoints')

  number_training_iterations = FLAGS.number_training_iterations
  learning_rate = FLAGS.learning_rate

  if FLAGS.eval_every == -1:
    eval_every = number_training_iterations // 100
  else:
    eval_every = FLAGS.eval_every

  symsol_shapes = FLAGS.symsol_shapes
  if 'symsol1' in symsol_shapes:
    symsol_shapes = data.SHAPE_NAMES[:5]
  ######################   Create the models   ###############################
  vision_model, len_visual_description = models.create_vision_model(
      weights=None if FLAGS.mock else 'imagenet')

  model_head = models.ImplicitSO3(len_visual_description,
                                  FLAGS.number_fourier_components,
                                  FLAGS.head_network_specs,
                                  FLAGS.so3_sampling_mode,
                                  FLAGS.number_train_queries,
                                  FLAGS.number_eval_queries)
  ######################   Load the datasets   ###############################
  dset_train = data.load_symsol(symsol_shapes, mode='train', mock=FLAGS.mock)
  dset_train = dset_train.repeat().shuffle(1000).batch(FLAGS.batch_size)

  dset_val_list = []
  for symsol_shape in symsol_shapes:
    dset_val_list.append(
        data.load_symsol(
            [symsol_shape],
            mode='test',
            downsample_continuous_gt=FLAGS.downsample_continuous_gt,
            mock=FLAGS.mock))
  dset_val_tags = symsol_shapes

  visualization_images, visualization_rotations_gt = [[], []]
  for image, rotations_gt in tf.data.experimental.sample_from_datasets(
      dset_val_list).take(8):
    visualization_images.append(image)
    visualization_rotations_gt.append(rotations_gt)

  measurement_labels = []
  for tag in dset_val_tags:
    measurement_labels += [f'gt_log_likelihood_{tag}', f'spread_{tag}']
  measurements = {}
  ##########################  Optimizer  #####################################
  optimizer = tf.keras.optimizers.get(FLAGS.optimizer)
  optimizer.learning_rate = learning_rate
  #########################  Logging setup  ##################################
  train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
  log_dir = os.path.join(outdir, 'logs')
  train_summary_writer = tf.summary.create_file_writer(log_dir)
  #########################  Checkpointing  ##################################
  checkpoint = tf.train.Checkpoint(
      vision_model=vision_model, model_head=model_head, optimizer=optimizer)

  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  max_ckpts_to_keep = 1
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint, directory=checkpoint_dir, max_to_keep=max_ckpts_to_keep)

  def cosine_decay(step, warmup_steps=1000):
    warmup_factor = min(step, warmup_steps) / warmup_steps
    decay_step = max(step - warmup_steps, 0) / (
        number_training_iterations - warmup_steps)
    return learning_rate * warmup_factor * (1 + tf.cos(decay_step * np.pi)) / 2

  @tf.function
  def train_step(vision_model, model_head, optimizer, images, rotations_gt):
    with tf.GradientTape() as tape:
      vision_description = vision_model(images, training=True)
      loss = model_head.compute_loss(vision_description, rotations_gt)
    grads = tape.gradient(
        loss,
        vision_model.trainable_variables + model_head.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, vision_model.trainable_variables +
            model_head.trainable_variables))
    return loss

  logging.info('Started training.')
  for batch_data in dset_train:
    step_num = optimizer.iterations.numpy()
    if step_num > number_training_iterations:
      break

    tf.keras.backend.set_value(optimizer.learning_rate,
                               cosine_decay(step_num))
    images, rotations_gt = batch_data
    loss = train_step(vision_model, model_head, optimizer, images, rotations_gt)

    train_loss(loss)

    if (step_num) % 100 == 0:
      avg_loss = train_loss.result()
      train_loss.reset_states()
      with train_summary_writer.as_default():
        tf.summary.scalar('loss', avg_loss, step=step_num)
        tf.summary.scalar(
            'learning_rate', optimizer.learning_rate, step=step_num)
      logging.info('Step %d, training loss=%.2f', step_num, avg_loss)

    if (step_num+1) % eval_every == 0:
      measurements = eval_step(
          vision_model,
          model_head,
          dset_val_list,
          dset_val_tags)

      with train_summary_writer.as_default():
        logline = f'Step {step_num}: '
        for k, v in measurements.items():
          tf.summary.scalar(k, v, step=step_num)
          logline += f'{k}={v:.2f} '
        logging.info(logline)
        logging.info('Started visualize_so3.')
        distribution_images = evaluation.visualize_model_output(
            vision_model, model_head, visualization_images,
            visualization_rotations_gt)
        tf.summary.image('output_distribution', distribution_images,
                         step=step_num)

      logging.info('Saving checkpoint.')
      checkpoint_manager.save()

  logging.info('Finished training.')
  #######################  Save the models  ##################################
  if FLAGS.save_models:
    visual_input = tfkl.Input(shape=(len_visual_description,))
    query_input = tfkl.Input(shape=(None, model_head.len_query,))
    inp = [visual_input, query_input]
    saveable_head_model = tf.keras.Model(inp, model_head(inp))
    models_saved = False
    if not models_saved:
      vision_model.save(os.path.join(outdir, 'base_vision_model'))
      saveable_head_model.save(os.path.join(outdir, 'ipdf_head_model'))

    logging.info('Saved models.')


def eval_step(vision_model, model_head, dataset_list,
              dataset_measurement_tags):
  """Evaluate distribution-based metrics.

  Args:
    vision_model: The model which produces a feature vector to hand to IPDF.
    model_head: The IPDF model.
    dataset_list: A list of datasets, to evaluate separately.
    dataset_measurement_tags: The names to associate with each dataset in
      dataset_list, when outputting metrics.
  Returns:
    A dictionary of values indexed by their associated descriptor tag.
  """
  assert len(dataset_list) == len(dataset_measurement_tags)
  measurements = {}
  logging.info('Started eval_step.')
  for dataset, tag in zip(dataset_list, dataset_measurement_tags):
    avg_log_likelihood, spread = evaluation.eval_spread_and_loglikelihood(
        vision_model,
        model_head,
        dataset,
        batch_size=FLAGS.test_batch_size,
        skip_spread_evaluation=FLAGS.skip_spread_evaluation,
        number_eval_iterations=FLAGS.number_eval_iterations)
    measurements[f'gt_log_likelihood_{tag}'] = avg_log_likelihood
    measurements[f'spread_{tag}'] = spread

  return measurements


if __name__ == '__main__':
  app.run(main)
