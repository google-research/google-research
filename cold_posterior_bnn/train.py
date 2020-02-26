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

# Lint as: python3
"""Runs MCMC methods for ResNet and LSTM models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib

from absl import app
from absl import flags
from absl import logging
import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from cold_posterior_bnn import datasets
from cold_posterior_bnn import models
from cold_posterior_bnn.core import ensemble
from cold_posterior_bnn.core import keras_utils
from cold_posterior_bnn.core import priorfactory
from cold_posterior_bnn.core import sgmcmc
from cold_posterior_bnn.core import statistics as stats


# FLAGS experiment
flags.DEFINE_string('output_dir', '/tmp/bnn/experiment/',
                    'Output directory.')
flags.DEFINE_integer('experiment_id', 0, 'ID of this run.')
flags.DEFINE_bool(
    'write_experiment_metadata_to_csv', False,
    'Write hyperparamters to csv file, useful for hyperparameter sweeps.')

# FLAGS train
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('train_epochs', 1000, 'Number of training epochs.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('pretend_batch_size', -1,
                     'Batch size used for cycle/epoch computations.')

flags.DEFINE_float('init_learning_rate', 0.1, 'Learning rate.')

# FLAGS dataset
flags.DEFINE_string('dataset', 'cifar10', 'Dataset from: cifar10, imdb.')
flags.DEFINE_bool('cifar_data_augmentation', True,
                  'Whether to use basic data augmentation for CIFAR-10 data.')
flags.DEFINE_integer('subsample_train_size', 0,
                     'Sub-sample training set to given number of samples.')

# FLAGS model
flags.DEFINE_string('model', 'resnet',
                    'Model to train, one of: cnnlstm, resnet.')
flags.DEFINE_bool('resnet_use_frn', False,
                  'Use filter response normalization instead of batchnorm.')
flags.DEFINE_bool('resnet_bias', True,
                  'Use biases in ResNet Conv2D layers.')

# Priors
flags.DEFINE_string('pfac', 'default',
                    'Use "default", "gaussian" prior factory.')

# FLAGS optimizer
flags.DEFINE_string('method', 'sgmcmc', 'MCMC method, one of: sgmcmc, baoab.')
flags.DEFINE_float('momentum_decay', 0.9,
                   'Momentum decay (used for sgmcmc).')

# FLAGS preconditioning
flags.DEFINE_bool('use_preconditioner', True,
                  'Use preconditioning of gradients (updated every epoch).')

# FLAGS cyclical learning rate ensemble
flags.DEFINE_integer('cycle_start_sampling', 10,
                     'Start sampling phase after x epoch.')
flags.DEFINE_integer('cycle_length', 5, 'Length of one cycle (in epochs).')
flags.DEFINE_string('cycle_schedule', 'cosine',
                    'Time stepping schedule ("cosine", "glide", or "flat").')

# FLAGS MCMC
flags.DEFINE_float('temperature', 1.,
                   'Temperature used in MCMC scheme (used for sgmcmc and hmc).')

FLAGS = flags.FLAGS
DATASET_SEED = 124


# Custom gradient function for SG-MCMC methods
def gradest_train_fn():
  """Function providing a step function for gradient estimation."""

  @tf.function
  def gest_step(grad_est, model, images, labels):
    """Custom gradient of log prior + log likelihood."""
    with tf.GradientTape(persistent=True) as tape:
      labels = tf.squeeze(labels)
      logits = model(images)
      ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=labels)
      ce = tf.reduce_mean(ce)
      prior = sum(model.losses)
      obj = ce + prior

    gradients = tape.gradient(obj, model.trainable_variables)
    grad_est.apply_gradients(zip(gradients, model.trainable_variables))

  def train_step(grad_est, model, data):
    images, labels = data
    gest_step(grad_est, model, images, labels)

  return train_step


def main(argv):
  del argv  # unused arg

  tf.io.gfile.makedirs(FLAGS.output_dir)

  # Load data
  tf.random.set_seed(DATASET_SEED)

  if FLAGS.dataset == 'cifar10':
    dataset_train, ds_info = datasets.load_cifar10(
        tfds.Split.TRAIN, with_info=True,
        data_augmentation=FLAGS.cifar_data_augmentation,
        subsample_n=FLAGS.subsample_train_size)
    dataset_test = datasets.load_cifar10(tfds.Split.TEST)
    logging.info('CIFAR10 dataset loaded.')

  elif FLAGS.dataset == 'imdb':
    dataset, ds_info = datasets.load_imdb(
        with_info=True, subsample_n=FLAGS.subsample_train_size)
    dataset_train, dataset_test = datasets.get_generators_from_ds(dataset)
    logging.info('IMDB dataset loaded.')

  else:
    raise ValueError('Unknown dataset {}.'.format(FLAGS.dataset))

  # Prepare data for SG-MCMC methods
  dataset_size = ds_info['train_num_examples']
  dataset_size_orig = ds_info.get('train_num_examples_orig', dataset_size)
  dataset_train = dataset_train.repeat().shuffle(10 * FLAGS.batch_size).batch(
      FLAGS.batch_size)
  test_batch_size = 100
  validation_steps = ds_info['test_num_examples'] // test_batch_size
  dataset_test_single = dataset_test.batch(FLAGS.batch_size)
  dataset_test = dataset_test.repeat().batch(test_batch_size)

  # If --pretend_batch_size flag is provided any cycle/epoch-length computation
  # is done using this pretend_batch_size.  Real batches are all still
  # FLAGS.batch_size of length.  This feature is used in the batch size ablation
  # study.
  #
  # Also, always determine number of iterations from original data set size
  if FLAGS.pretend_batch_size >= 1:
    steps_per_epoch = dataset_size_orig // FLAGS.pretend_batch_size
  else:
    steps_per_epoch = dataset_size_orig // FLAGS.batch_size

  # Set seed for the experiment
  tf.random.set_seed(FLAGS.seed)

  # Build model using pfac for proper priors
  reg_weight = 1.0 / float(dataset_size)
  if FLAGS.pfac == 'default':
    pfac = priorfactory.DefaultPriorFactory(weight=reg_weight)
  elif FLAGS.pfac == 'gaussian':
    pfac = priorfactory.GaussianPriorFactory(prior_stddev=1.0,
                                             weight=reg_weight)
  else:
    raise ValueError('Choose pfac from: default, gaussian.')

  input_shape = ds_info['input_shape']

  if FLAGS.model == 'cnnlstm':
    assert FLAGS.dataset == 'imdb'
    model = models.build_cnnlstm(ds_info['num_words'],
                                 ds_info['sequence_length'],
                                 pfac)

  elif FLAGS.model == 'resnet':
    assert FLAGS.dataset == 'cifar10'
    model = models.build_resnet_v1(
        input_shape=input_shape,
        depth=20,
        num_classes=ds_info['num_classes'],
        pfac=pfac,
        use_frn=FLAGS.resnet_use_frn,
        use_internal_bias=FLAGS.resnet_bias)
  else:
    raise ValueError('Choose model from: cnnlstm, resnet.')

  model.summary()

  # Setup callbacks executed in keras.compile loop
  callbacks = []

  # Setup preconditioner
  precond_dict = dict()

  if FLAGS.use_preconditioner:
    precond_dict['preconditioner'] = 'fixed'
    logging.info('Use fixed preconditioner.')
  else:
    logging.info('No preconditioner is used.')

  # Always append preconditioner callback to compute ctemp statistics
  precond_estimator_cb = keras_utils.EstimatePreconditionerCallback(
      gradest_train_fn,
      iter(dataset_train),
      every_nth_epoch=1,
      batch_count=32,
      raw_second_moment=True,
      update_precond=FLAGS.use_preconditioner)
  callbacks.append(precond_estimator_cb)

  # Setup MCMC method
  if FLAGS.method == 'sgmcmc':
    # SG-MCMC optimizer, first-order symplectic Euler integrator
    optimizer = sgmcmc.NaiveSymplecticEulerMCMC(
        total_sample_size=dataset_size,
        learning_rate=FLAGS.init_learning_rate,
        momentum_decay=FLAGS.momentum_decay,
        temp=FLAGS.temperature,
        **precond_dict)
    logging.info('Use symplectic Euler integrator.')

  elif FLAGS.method == 'baoab':
     # SG-MCMC optimizer, second-order accurate BAOAB integrator
    optimizer = sgmcmc.BAOABMCMC(
        total_sample_size=dataset_size,
        learning_rate=FLAGS.init_learning_rate,
        momentum_decay=FLAGS.momentum_decay,
        temp=FLAGS.temperature,
        **precond_dict)
    logging.info('Use BAOAB integrator.')
  else:
    raise ValueError('Choose method from: sgmcmc, baoab.')

  # Statistics for evaluation of ensemble performance
  perf_stats = {
      'ens_gce': stats.MeanStatistic(stats.ClassificationGibbsCrossEntropy()),
      'ens_ce': stats.MeanStatistic(stats.ClassificationCrossEntropy()),
      'ens_ce_sem': stats.StandardError(stats.ClassificationCrossEntropy()),
      'ens_brier': stats.MeanStatistic(stats.BrierScore()),
      'ens_brier_unc': stats.BrierUncertainty(),
      'ens_brier_res': stats.BrierResolution(),
      'ens_brier_reliab': stats.BrierReliability(),
      'ens_ece': stats.ECE(10),
      'ens_gacc': stats.MeanStatistic(stats.GibbsAccuracy()),
      'ens_acc': stats.MeanStatistic(stats.Accuracy()),
      'ens_acc_sem': stats.StandardError(stats.Accuracy()),
  }

  perf_stats_l, perf_stats_s = zip(*(perf_stats.items()))

  # Setup ensemble
  ens = ensemble.EmpiricalEnsemble(model, input_shape)
  last_ens_eval = {'size': 0}  # ensemble size from last evaluation

  def cycle_ens_eval_maybe():
    """Ensemble evaluation callback, only evaluate at end of cycle."""

    if len(ens) > last_ens_eval['size']:
      last_ens_eval['size'] = len(ens)
      logging.info('... evaluate ensemble on %d members', len(ens))
      return ens.evaluate_ensemble(
          dataset=dataset_test_single, statistics=perf_stats_s)
    else:
      return None

  ensemble_eval_cb = keras_utils.EvaluateEnsemblePartial(
      cycle_ens_eval_maybe, perf_stats_l)
  callbacks.append(ensemble_eval_cb)

  # Setup cyclical learning rate and temperature schedule for sgmcmc
  if FLAGS.method == 'sgmcmc' or FLAGS.method == 'baoab':
    # setup cyclical learning rate schedule
    cyclic_sampler_cb = keras_utils.CyclicSamplerCallback(
        ens,
        FLAGS.cycle_length * steps_per_epoch,  # number of iterations per cycle
        FLAGS.cycle_start_sampling,  # sampling phase start epoch
        schedule=FLAGS.cycle_schedule,
        min_value=0.0)  # timestep_factor min value
    callbacks.append(cyclic_sampler_cb)

    # Setup temperature ramp-up schedule
    begin_ramp_epoch = FLAGS.cycle_start_sampling - FLAGS.cycle_length
    if begin_ramp_epoch < 0:
      raise ValueError(
          'cycle_start_sampling must be greater equal than cycle_length.')
    ramp_iterations = FLAGS.cycle_length
    tempramp_cb = keras_utils.TemperatureRampScheduler(
        0.0, FLAGS.temperature, begin_ramp_epoch * steps_per_epoch,
        ramp_iterations * steps_per_epoch)
    # T0, Tf, begin_iter, ramp_epochs
    callbacks.append(tempramp_cb)

  # Additional callbacks
  # Plot additional logs
  def plot_logs(epoch, logs):
    del epoch  # unused
    logs['lr'] = optimizer.get_config()['learning_rate']
    if FLAGS.method == 'sgmcmc':
      logs['timestep_factor'] = optimizer.get_config()['timestep_factor']
    logs['ens_size'] = len(ens)
  plot_logs_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=plot_logs)

  # Write logs to tensorboard
  tensorboard_cb = tf.keras.callbacks.TensorBoard(
      log_dir=FLAGS.output_dir, write_graph=False)

  # Output ktemp
  diag_cb = keras_utils.PrintDiagnosticsCallback(10)

  callbacks.extend([
      diag_cb,
      plot_logs_cb,
      keras_utils.TemperatureMetric(),
      keras_utils.SamplerTemperatureMetric(),
      tensorboard_cb,  # Should be after all callbacks that write logs
      tf.keras.callbacks.CSVLogger(os.path.join(FLAGS.output_dir, 'logs.csv'))
  ])

  # Keras train model
  metrics = [
      tf.keras.metrics.SparseCategoricalCrossentropy(
          name='negative_log_likelihood',
          from_logits=True),
      tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
  model.compile(
      optimizer,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=metrics)
  logging.info('Model input shape: %s', model.input_shape)
  logging.info('Model output shape: %s', model.output_shape)
  logging.info('Model number of weights: %s', model.count_params())

  model.fit(
      dataset_train,
      steps_per_epoch=steps_per_epoch,
      epochs=FLAGS.train_epochs,
      validation_data=dataset_test,
      validation_steps=validation_steps,
      callbacks=callbacks)

  # Evaluate final ensemble performance
  logging.info('Ensemble has %d members, computing final performance metrics.',
               len(ens))

  if ens.weights_list:
    ens_perf_stats = ens.evaluate_ensemble(dataset_test_single, perf_stats_s)
    print('Test set metrics:')
    for label, stat_value in zip(perf_stats_l, ens_perf_stats):
      stat_value = float(stat_value)
      logging.info('%s: %.5f', label, stat_value)
      print('%s: %.5f' % (label, stat_value))

  # Add experiment info to experiment metadata csv file in *parent folder*
  if FLAGS.write_experiment_metadata_to_csv:
    csv_path = pathlib.Path.joinpath(
        pathlib.Path(FLAGS.output_dir).parent, 'run_sweeps.csv')
    data = {
        'id': [FLAGS.experiment_id],
        'seed': [FLAGS.seed],
        'temperature': [FLAGS.temperature],
        'dir': ['run_{}'.format(FLAGS.experiment_id)]
    }
    if tf.io.gfile.exists(csv_path):
      sweeps_df = pd.read_csv(csv_path)
      sweeps_df = sweeps_df.append(
          pd.DataFrame.from_dict(data), ignore_index=True).set_index('id')
    else:
      sweeps_df = pd.DataFrame.from_dict(data).set_index('id')

    # save experiment metadata csv file
    sweeps_df.to_csv(csv_path)


if __name__ == '__main__':

  # Print logging.info directly in shell
  def log_print(msg, *args):
    print(msg % args)
  logging.info = log_print

  tf.enable_v2_behavior()
  app.run(main)
