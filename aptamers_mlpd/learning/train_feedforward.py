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

# Lint as: python3
r"""Train a simple feedforward net to predict aptamer read counts.

Sample usage:

xx/learning/train_feedforward \
  --epochs=1 --mbsz=64 --save_base=/tmp/ --dataset=xxx --run_name=debug
"""

import logging
import os
import time


import numpy as np
import six
import xarray

import tensorflow.compat.v1 as tf
from tensorflow.contrib import labeled_tensor as lt

# Google internal
import shards
import slim
import study_configuration as study_config
import vizier_pb2
import vizier_client
import text_format
import app
import flags
import gfile
import StopWatch
import generationinfo
import sstable
import deprecated as contrib_deprecated
import pybar


from ..util import io_utils
from ..util import selection_pb2
from ..learning import config
from ..learning import data
from ..learning import dataset_stats
from ..learning import eval_feedforward as eval_ff
from ..learning import feedforward as ff
from ..learning import output_layers
from ..learning import utils


_VALID_INPUT_FEATURES = frozenset({
    data.SEQUENCE_ONE_HOT,
    data.SEQUENCE_KMER_COUNT,
})

TUNER_LOSS_LOSS = 'loss'
TUNER_LOSS_AUC = 'auc/true_top_1p'
TUNER_GOAL_MAX = 'MAXIMIZE'
TUNER_GOAL_MIN = 'MINIMIZE'
TUNER_LOSS_TO_GOAL = {
    TUNER_LOSS_LOSS: TUNER_GOAL_MIN,
    TUNER_LOSS_AUC: TUNER_GOAL_MAX,
}

flags.DEFINE_integer('task', 0, 'Task id when running online')
flags.DEFINE_string('master', '', 'TensorFlow master to use')
flags.DEFINE_string('input_dir', None, 'Path to input data.')
flags.DEFINE_string(
    'affinity_target_map', '',
    'Name of the affinity map from count values to affinity values. '
    'Needed only if using input_dir and running inference or using '
    'microarray values.')
flags.DEFINE_enum(
    'dataset', None,
    sorted(config.INPUT_DATA_DIRS),
    'Name of dataset with known input_dir on which to train. Either input_dir '
    'or dataset is required.')
flags.DEFINE_integer('val_fold', 0, 'Fold to use for validation.')
flags.DEFINE_string('save_base', None,
                    'Base path to save any output or weights.')
flags.DEFINE_string('run_name', None, 'Name of folder created in save_base.')
flags.DEFINE_bool(
    'interactive_display', False,
    'Scale displayed pandas DataFrames to the active terminal window?')
flags.DEFINE_boolean(
    'autotune', False,
    'If true, use automated hyperparameter optimization via Vizier.')
flags.DEFINE_string('tuner_target', 'mean',
                    'Target count(s) for use for tuner optimization.')
flags.DEFINE_enum('tuner_loss', 'auc/true_top_1p',
                  [TUNER_LOSS_LOSS, TUNER_LOSS_AUC],
                  'Loss function to use for tuner optimization.')
flags.DEFINE_enum('tuner_algorithm', 'RANDOM_SEARCH',
                  ['RANDOM_SEARCH', 'DEFAULT_ALGORITHM'],
                  'Algorithm to use for searching with Vizier')
# FLAGS.study_name is defined by Vizier. By default, we use run_name to set it.
flags.DEFINE_enum('output_layer', 'FULLY_OBSERVED', [
    output_layers.OUTPUT_FULLY_OBSERVED, output_layers.OUTPUT_LATENT_AFFINITY,
    output_layers.OUTPUT_LATENT_WITH_DEPS,
    output_layers.OUTPUT_LATENT_WITH_PRED_DEPS,
    output_layers.OUTPUT_LATENT_WITH_CROSS_DEPS
], 'Name of the output layer class to use.')
flags.DEFINE_enum('loss_name', 'SQUARED_ERROR', [
    output_layers.LOSS_SQUARED_ERROR,
    output_layers.LOSS_CROSS_ENTROPY,
    output_layers.LOSS_POISSON_LOSS,
    output_layers.LOSS_ZERO_TRUNCATED_POISSON_LOSS,
], 'Name of the loss class to use.')
flags.DEFINE_enum('dependency_norm', 'STANDARDIZE', [
    output_layers.NORM_STANDARDIZE,
    output_layers.NORM_BINARIZE,
    output_layers.NORM_TOTAL_COUNTS,
    output_layers.NORM_SKIP,
], 'Method to use for normalization in the dependency model.')
flags.DEFINE_enum('loss_norm', 'STANDARDIZE', [
    output_layers.NORM_STANDARDIZE,
    output_layers.NORM_BINARIZE,
    output_layers.NORM_TOTAL_COUNTS,
    output_layers.NORM_SKIP,
], 'Method to use for normalization in the loss.')
flags.DEFINE_bool('standardize_log_transform', True,
                  'Log transform sequencing counts before standardizing.')
flags.DEFINE_integer('binarize_threshold', 1,
                     'Threshold for binarizing sequencing counts.')
flags.DEFINE_integer('epochs', 5, 'Number of epochs for training.')
flags.DEFINE_integer(
    'epoch_size', 0,
    'Number of examples in an epoch (will be rounded down to the nearest '
    'mini-batch size), defaults to size of full training dataset.')
flags.DEFINE_integer(
    'eval_size',
    int(1e6), 'Maximum number of examples to evaluate in each evaluation loop.')
flags.DEFINE_integer('mbsz', 128, 'Minibatch size during training.')
flags.DEFINE_enum('optimizer', 'momentum', ['adam', 'momentum'],
                  'Optimizer name')
flags.DEFINE_integer('num_conv_layers', 0,
                     'Number of convolutional hidden layers.')
flags.DEFINE_integer('max_strides', 1,
                     'When using convolutional layers and parameter tuning, '
                     'the maximum stride to test.')
flags.DEFINE_integer('max_rates', 1,
                     'When using convolutional layers and parameter tuning, '
                     'the maximum dilation rate to test.')
flags.DEFINE_integer('num_fc_layers', None,
                     'Number of fully connected hidden layers.')
flags.DEFINE_list(
    'target_names', [output_layers.TARGETS_ALL_OUTPUTS],
    'List of count targets to train against. By default, train against all '
    'counts.')
flags.DEFINE_enum('preprocess_mode', 'PREPROCESS_SKIP_ALL_ZERO_COUNTS', [
    data.PREPROCESS_SKIP_ALL_ZERO_COUNTS,
    data.PREPROCESS_INJECT_RANDOM_SEQUENCES,
    data.PREPROCESS_ALL_COUNTS
], 'How to preprocess input data for training purposes.')
flags.DEFINE_list('input_features', [
    'SEQUENCE_ONE_HOT'
], 'List of features to use as inputs to the model. Valid choices: %r' %
                  _VALID_INPUT_FEATURES)
flags.DEFINE_integer(
    'kmer_k_max', 4,
    'Maximum k-mer size for which to calculate counts if using '
    'SEQUENCE_KMER_COUNT as a feature.')
flags.DEFINE_float(
    'ratio_random_dna', 1.0,
    'Ratio of random sequences to inject if using preprocess_mode == '
    'PREPROCESS_INJECT_RANDOM_SEQUENCES. Used to scale the default epoch_size.')
flags.DEFINE_integer(
    'total_reads_defining_positive', 0,
    'Number of reads required to be seen across all conditions to classify '
    'an example as positive.')
# TODO(mdimon): add a discounted cumulative gain when they are implemented
flags.DEFINE_list('metrics_measures',
                  ['auc/true_top_1p', 'spearman_correlation/score_top_1p'],
                  'Metric measurements to report to Vizier')
flags.DEFINE_string('hpconfig', '',
                    """A comma separated list of hyperparameters for the model.
    Format is hp1=value1,hp2=value2,etc. If this FLAG is set and
    there is a tuner, the tuner will train the model
    with the specified hyperparameters, filling in
    missing hyperparameters from the default_values in
    the list of hyper_params and only receiving
    hyperparameters in the study definition from the
    tuner service.""")
flags.DEFINE_string('validation_fold_template', '',
                    'A template for the filename of the validation fold.')
flags.DEFINE_bool('verbose_eval', False,
                  'True will print more evaluation metrics.')
flags.DEFINE_bool('train_on_array', True,
                  'True will train the model on binding array data.')
flags.DEFINE_bool('save_stats', False,
                  'True will save the computed statistics.')
flags.DEFINE_integer('epoch_interval_to_save_best', 5,
                     'The epoch interval at which to check if the current model'
                     'has lower mean loss than the best so far, and if so'
                     'update the saved best model')
flags.DEFINE_string('additional_output', '', 'A comma-delimited string'
                    'indicating which feature to predict in addition to counts')
FLAGS = flags.FLAGS

# These hparams are directly copied from flags:
HPARAM_FLAGS = [
    'val_fold',
    'run_name',
    'tuner_target',
    'tuner_loss',
    'output_layer',
    'dependency_norm',
    'loss_name',
    'loss_norm',
    'standardize_log_transform',
    'binarize_threshold',
    'epochs',
    'epoch_size',
    'eval_size',
    'mbsz',
    'num_conv_layers',
    'num_fc_layers',
    'target_names',
    'preprocess_mode',
    'input_features',
    'kmer_k_max',
    'ratio_random_dna',
    'total_reads_defining_positive',
    'train_on_array',
    'save_stats',
    'epoch_interval_to_save_best',
    'additional_output',
]

# If training error ever exceeds MAX_COST, we assume training has
# diverged and raise eval_ff.TrainingDivergedException to stop training.
MAX_COST = 5e3

logger = logging.getLogger(__name__)


class Error(Exception):
  pass


def _default_study_hyperparams(num_conv_layers, num_fc_layers):
  """Create the default hyperparameter values given a number of layers.

  Args:
    num_conv_layers: non-negative integer number of convolutional layers
    num_fc_layers: non-negative integer number of hidden layers

  Returns:
    A tf.HParams instance holding default hyperparameters.
  """
  hypers = {
      'nonlinearity': 'tanh',
      'learn_rate': 0.005,
      'momentum': 0.9,
      'output_init_factor': 1.0,
      'dropouts': [0.0] * (num_fc_layers + 1)
  }
  if num_fc_layers:
    hypers.update({
        'fc_hid_sizes': [256] * num_fc_layers,
        'fc_init_factors': [1.0] * num_fc_layers
    })
  if num_conv_layers:
    # HParams cannot handle empty lists
    hypers.update({
        'conv_widths': [16] * num_conv_layers,
        'conv_depths': [32] * num_conv_layers,
        'conv_strides': [1] * num_conv_layers,
        'conv_rates': [1] * num_conv_layers,
        'conv_init_factors': [1.0] * num_conv_layers
    })

  return tf.HParams(**hypers)


# TODO(gdahl): specify the study params with flags, config file, or something
# Hard-coding them is unacceptable in the long run.
def _get_study(study_name, tuner_goal, tuner_algorithm, num_conv_layers,
               max_strides, max_rates, num_fc_layers, hpconfig):
  """Return a dict of study params."""
  study = study_config.StudyConfiguration(
      goal=getattr(study_config, tuner_goal),
      algorithm=getattr(study_config, tuner_algorithm),
      study_name=study_name,)
  study.AddCategoricalParam('nonlinearity', ['tanh', 'sigmoid', 'relu', 'elu'])
  study.AddFloatParam('learn_rate', 0.0001, 0.01, scale=study_config.LOG)
  study.AddFloatParam('momentum', 0.0, 0.99)
  study.AddFloatParam('output_init_factor', 1.0, 1.0)
  study.AddFloatParam('dropouts', 0.0, 0.5, length=num_fc_layers + 1)
  if num_fc_layers:
    study.AddIntParam('fc_hid_sizes', 64, 768, length=num_fc_layers)
    study.AddFloatParam('fc_init_factors', 1.0, 1.0, length=num_fc_layers)
  if num_conv_layers:
    study.AddIntParam('conv_widths', 4, 12, length=num_conv_layers)
    study.AddIntParam('conv_depths', 16, 64, length=num_conv_layers)
    study.AddIntParam('conv_strides', 1, max_strides, length=num_conv_layers)
    study.AddIntParam('conv_rates', 1, max_rates, length=num_conv_layers)
    study.AddFloatParam('conv_init_factors', 1.0, 1.0, length=num_conv_layers)
  study.FixSelectParameters(hpconfig)
  return study


def add_summary(summarizer, tag, value, global_step):
  s = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
  summarizer.add_summary(s, global_step)


class FeedForwardTrainer:
  """Class for training a simple feedforward net on aptamers data.

  This class constructs the TF graph we need to train a model on aptamer data
  and to evaluate the model on the validation data during training. It also
  provides a method to actually run the training and uses some helper methods
  to handle reporting training progress.

  During training, this class also measures validation error. Therefore it also
  builds the graph needed for making predictions on new data using feed_dict.
  To evaluate the model, we feed serialized example protos to self.val_strs.

  Attributes:
    hps: The configuration object holding values of training metaparameters.
      The trainer object just expects hps to have a learn_rate and momentum
      attribute, but if the network uses the same object more attributes are
      needed. Usually we expect hps to be an instance of tf.HParams and in
      order for restoring the model to work it will need to be.
    experiment_proto: selection_pb2.Experiment describing the experiment.
    net: The neural net we are training.
    train_dir: The folder we write checkpoints and other data to.
    global_step: TF variable holding the global step counter
  """

  def __init__(self, hps, net, output_layer, experiment_proto, input_paths):
    inputs, outputs = data.input_pipeline(
        input_paths, experiment_proto, hps.mbsz, hps=hps, num_threads=8)
    with tf.name_scope('neural_net'):
      logits = net.fprop(inputs, mode='train')
    with tf.name_scope('output_layer'):
      loss_per_target = output_layer.average_loss_per_target(
          logits, outputs, include_array=hps.train_on_array)
      loss = utils.reduce_nanmean(loss_per_target)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if hps.optimizer == 'momentum':
      optimizer = tf.MomentumOptimizer(hps.learn_rate, hps.momentum)
    elif hps.optimizer == 'adam':
      optimizer = tf.AdamOptimizer(hps.learn_rate)
    else:
      raise ValueError('invalid optimizer: %s' % hps.optimizer)
    optimizer = tf.MomentumOptimizer(hps.learn_rate, hps.momentum)
    grads = optimizer.compute_gradients(loss, net.params + output_layer.params)
    opt_op = optimizer.apply_gradients(grads, global_step=self.global_step)
    self.train_op = tf.with_dependencies([opt_op], loss)

    contrib_deprecated.scalar_summary('loss/mean', loss)
    for target in loss_per_target.axes['target'].labels:
      contrib_deprecated.scalar_summary(
          'loss/' + six.ensure_str(target),
          lt.select(loss_per_target, {'target': target}))
    with tf.name_scope('summarize_grads'):
      slim.learning.add_gradients_summaries(grads)

    tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, self.global_step)
    tf.add_to_collection('train_op', self.train_op)
    tf.add_to_collection('loss', loss)

    self.mbsz = hps.mbsz
    # The log Poisson loss implemented in TensorFlow may sometimes be negative.
    if (hps.loss_name == output_layers.LOSS_POISSON_LOSS or
        hps.loss_name == output_layers.LOSS_ZERO_TRUNCATED_POISSON_LOSS):
      self.min_cost = -float('inf')
      self.min_is_inclusive = False
    else:
      self.min_cost = 0
      self.min_is_inclusive = True

  def train(self, sess, num_epochs, num_batches_per_epoch):
    """Method to actually run training given a session, and data.

    The train method is responsible for the main job of the trainer class. It
    takes a supervisor, session, training and validation data, number of
    epochs, and epoch size as input. It will write summaries and print reports.

    Args:
      sess: TF session to use for running the graph
      num_epochs: integer number of epochs of training to perform
      num_batches_per_epoch: integer number of batches in a single epoch

    Yields:
      None, after completing each epoch.

    Raises:
      eval_ff.TrainingDivergedException: when training loss gets above MAX_COST.
        This might be caused by bad initial weights or a large learning rate
        and/or momentum.
    """
    gs = self.global_step.eval(sess)

    # if global_step > 0, resume training where we left off
    epoch_start, step_start = divmod(gs, num_batches_per_epoch)
    logger.info('Starting training at global_step=%d', gs)

    for epoch in range(epoch_start, num_epochs):
      sw = StopWatch()

      num_steps = num_batches_per_epoch - step_start
      for _ in pybar.BarGenerator(
          list(range(num_steps)), message='Epoch %d' % epoch):
        sw.start('step')
        gs, cur_cost = sess.run([self.global_step, self.train_op])
        sw.stop('step')

        if (cur_cost > MAX_COST or cur_cost < self.min_cost or
            (cur_cost <= self.min_cost and not self.min_is_inclusive)):
          lower_bracket = '[' if self.min_is_inclusive else '('
          msg = (
              'instantaneous training cost of %f is outside bounds %s%f, %f]' %
              (cur_cost, lower_bracket, self.min_cost, MAX_COST))
          raise eval_ff.TrainingDivergedException(msg)

      examples_per_sec = float(self.mbsz * num_steps) / sw.timervalue('step')
      logger.info('gs=%d, epoch=%d, examples/second=%f', gs, epoch,
                  examples_per_sec)

      yield epoch

      step_start = 0


def _expand_sharded_paths(basepaths):
  """Returns a list of all filepaths represented by the given basepaths.

  Args:
    basepaths: A list of base paths that may or may not be part of sharded
        paths.

  Returns:
    A list of all files corresponding to the input base paths. If the inputs
    are unsharded, the return value will be identical to the input.
  """
  retval = []
  for path in basepaths:
    try:
      num_shards = shards.DetectNumberOfShards(path)
    except shards.Error:
      # This is not a sharded file, just append it directly.
      retval.append(path)
    else:
      retval.extend(
          shards.GenerateShardedFilenames('%s@%d' % (path, num_shards)))
  return retval


def _setup_fold(input_dir, train_dir, val_fold, val_fold_template):
  """Read the experiment proto and setup input for this validation fold.

  Args:
    input_dir: string giving the path to input data.
    train_dir: string path to directory for writing TensorFlow stuff.
    val_fold: integer ID for this validation fold (e.g., between 0 and 4)
    val_fold_template: string to include in validation fold basename.

  Returns:
    experiment_proto: selection_pb2.Experiment proto for training.
    train_input_paths: list of strings giving paths to sstables with training
      data.
    val_input_paths: list of strings giving paths to sstable(s) with validation
      data.
  """
  train_pbtxt_file = config.wetlab_experiment_train_pbtxt_path[val_fold]
  train_pbtxt_file_w_stats = six.ensure_str(train_pbtxt_file) + '.wstats'
  if gfile.Exists(os.path.join(input_dir, train_pbtxt_file_w_stats)):
    logger.info('Load pbtxt file with statistics: %s',
                os.path.join(input_dir, train_pbtxt_file_w_stats))
    with gfile.GFile(os.path.join(input_dir, train_pbtxt_file_w_stats)) as f:
      experiment_proto = text_format.Parse(f.read(), selection_pb2.Experiment())
  else:
    logger.info('Load pbtxt file without statistics: %s',
                os.path.join(input_dir, train_pbtxt_file))
    with gfile.GFile(os.path.join(input_dir, train_pbtxt_file)) as f:
      experiment_proto = text_format.Parse(f.read(), selection_pb2.Experiment())

  val_pbtxt_file = config.get_wetlab_experiment_val_pbtxt_path(
      val_fold, val_fold_template)
  gfile.Copy(
      os.path.join(input_dir, val_pbtxt_file),
      os.path.join(train_dir, config.wetlab_experiment_val_name),
      overwrite=True)

  train_input_paths = _expand_sharded_paths([
      os.path.join(input_dir, p)
      for n, p in enumerate(config.example_sstable_paths) if n != val_fold
  ])
  val_input_paths = _expand_sharded_paths([
      os.path.join(input_dir,
                   config.get_example_sstable_path(val_fold, val_fold_template))
  ])

  return experiment_proto, train_input_paths, val_input_paths


def _num_valid_convolutions(n_positions, filter_width, stride, rate):  # pylint: disable=unused-argument
  """Returns the output length for a convolution.

  Args:
    n_positions: Integer, the number of positions to convolute.
    filter_width: Integer, the width of the filter.
    stride: Integer, the stride with which the filter is slid.
    rate: Integer, the dilation rate for expanding the filter.
  Returns:
    The integer number of valid convolutions.
  Raises:
    Error if both rate and stride are greater than 1 or if either stride
    or rate are less than 1.
  """
  # If we ever go back to padding='VALID', the equation for dilated conv is:
  # dil_filter_width = (filter_width * rate) - (rate - 1)
  # np.ceil(float(n_positions - dil_filter_width + 1) / float(stride))
  return np.ceil(float(n_positions) / float(stride))


def hps_is_infeasible(hps, sequence_length):
  """Checks the hyperparameter combination to make sure it looks feasible.

  Args:
    hps: tf.HParams with training parameters.
    sequence_length: The length of the sequence, which is the axis over which
      convolution occur.
  Returns:
    (is_infeasible, reason) as (boolean, string) where is_infeasible indicates
    whether the hps parameters are infeasible and, if they are infeasible,
    the reason string indicates the reason why the combination is infeasible.
  """
  # for convolutional layers, we need to check that the number of outputs
  # (legal convolutions) from one layer needs to meets or exceeds
  # the width of the next layer.
  if hps.num_conv_layers > 0:
    widths = hps.conv_widths + [1]  # add next layer as width=1
    input_width = sequence_length
    for i in range(hps.num_conv_layers):
      if hps.conv_strides[i] > 1 and hps.conv_rates[i] > 1:
        return True, ('Only stride or rate can be greater than 1, not both. '
                      'Rate=%d, Stride=%d' % (hps.conv_rates[i],
                                              hps.conv_strides[i]))
      cur_output = _num_valid_convolutions(input_width, hps.conv_widths[i],
                                           hps.conv_strides[i],
                                           hps.conv_rates[i])
      # check the width for this layer is less than the previous outputs
      if cur_output < widths[i + 1]:
        return True, ('There are %d outputs from layer %d, which is smaller '
                      'than the width of the next layer which is %d' %
                      (cur_output, i, widths[i + 1]))
      input_width = cur_output

  return False, ''


def run_training(hps,
                 experiment_proto,
                 train_dir,
                 train_input_paths,
                 val_input_paths,
                 tuner=None,
                 master='',
                 metrics_targets=None,
                 metrics_measures=None):
  """Main training function.

  Trains the model given a directory to write to and a logfile to write to.

  Args:
    hps: tf.HParams with training parameters.
    experiment_proto: selection_pb2.Experiment proto for training.
    train_dir: str path to train directory.
    train_input_paths: List[str] giving paths to input sstables for training.
    val_input_paths: List[str] giving paths to input sstable(s) for validation.
    tuner: optional hp_tuner.HPTuner.
    master: optional string to pass to a tf.Supervisor.
    metrics_targets: String list of network targets to report metrics for.
    metrics_measures: Measurements about the performance of the network to
        report, e.g. 'auc/top_1p'.

  Returns:
    None.

  Raises:
    Error: if the hyperparamter combination in hps is infeasible and there is
    no tuner. (If the hyperparameter combination is infeasible and there is
    a tuner then the params are reported back to the tuner as infeasible.)
  """
  hps_infeasible, infeasible_reason = hps_is_infeasible(
      hps, experiment_proto.sequence_length)
  if hps_infeasible:
    if tuner:
      tuner.report_done(True, infeasible_reason)
      logger.info('report_done(infeasible=%r)', hps_infeasible)
      return
    else:
      raise Error('Hyperparams are infeasible: %s', infeasible_reason)

  logger.info('Starting training.')
  if tuner:
    logger.info('Using tuner: loaded HParams from Vizier')
  else:
    logger.info('No tuner: using default HParams')
  logger.info('experiment_proto: %s', experiment_proto)
  logger.info('train_dir: %s', train_dir)
  logger.info('train_input_paths[0]: %s', train_input_paths[0])
  logger.info('val_input_paths[0]: %s', val_input_paths[0])
  logger.info('%r', list(hps.values()))
  generationinfo.to_file(os.path.join(train_dir, 'geninfo.pbtxt'))
  with gfile.Open(os.path.join(train_dir, config.hparams_name), 'w') as f:
    f.write(str(hps.to_proto()))

  eval_size = hps.eval_size or None

  def make_subdir(subdirectory_mame):
    path = os.path.join(train_dir, subdirectory_mame)
    gfile.MakeDirs(path)
    return path

  logger.info('Computing preprocessing statistics')
  # TODO(shoyer): move this over into preprocessing instead?
  experiment_proto = dataset_stats.compute_experiment_statistics(
      experiment_proto,
      train_input_paths,
      os.path.join(
          hps.input_dir,
          six.ensure_str(
              config.wetlab_experiment_train_pbtxt_path[hps.val_fold]) +
          '.wstats'),
      preprocess_mode=hps.preprocess_mode,
      max_size=eval_size,
      logdir=make_subdir('compute-statistics'),
      save_stats=hps.save_stats)

  logging.info('Saving experiment proto with statistics')
  with gfile.Open(
      os.path.join(train_dir, config.wetlab_experiment_train_name), 'w') as f:
    f.write(str(experiment_proto))

  logger.debug(str(hps.to_proto()))
  logger.debug(hps.run_name)

  tr_entries = len(sstable.MergedSSTable(train_input_paths))
  logger.info('Training sstable size: %d', tr_entries)
  val_entries = len(sstable.MergedSSTable(val_input_paths))
  logger.info('Validation sstable size: %d', val_entries)

  epoch_size = hps.epoch_size or int(tr_entries * (1 + hps.ratio_random_dna))
  num_batches_per_epoch = int(float(epoch_size) / hps.mbsz)

  eval_ff.config_pandas_display(FLAGS.interactive_display)
  tr_evaluator = eval_ff.Evaluator(
      hps,
      experiment_proto,
      train_input_paths,
      make_subdir(config.experiment_training_dir),
      verbose=FLAGS.verbose_eval)
  val_evaluator = eval_ff.Evaluator(
      hps,
      experiment_proto,
      val_input_paths,
      make_subdir(config.experiment_validation_dir),
      verbose=FLAGS.verbose_eval)

  with tf.Graph().as_default():
    # we need to use the registered key 'hparams'
    tf.add_to_collection('hparams', hps)

    # TODO(shoyer): collect these into a Model class:
    dummy_inputs = data.dummy_inputs(
        experiment_proto,
        input_features=hps.input_features,
        kmer_k_max=hps.kmer_k_max,
        additional_output=six.ensure_str(hps.additional_output).split(','))
    output_layer = output_layers.create_output_layer(experiment_proto, hps)
    net = ff.FeedForward(dummy_inputs, output_layer.logit_axis, hps)

    trainer = FeedForwardTrainer(hps, net, output_layer, experiment_proto,
                                 train_input_paths)

    summary_writer = tf.SummaryWriter(make_subdir('training'), flush_secs=30)

    # TODO(shoyer): file a bug to figure out why write_version=2 (now the
    # default) doesn't work.
    saver = tf.Saver(write_version=1)

    # We are always the chief since we do not do distributed training.
    # Every replica with a different task id is completely independent and all
    # must be their own chief.
    sv = tf.Supervisor(
        logdir=train_dir,
        is_chief=True,
        summary_writer=summary_writer,
        save_summaries_secs=10,
        save_model_secs=180,
        saver=saver)

    logger.info('Preparing session')

    train_report_dir = os.path.join(train_dir, config.experiment_training_dir)
    cur_train_report = os.path.join(train_report_dir,
                                    config.experiment_report_name)
    best_train_report = os.path.join(train_report_dir,
                                     config.experiment_best_report_name)

    valid_report_dir = os.path.join(train_dir, config.experiment_validation_dir)
    cur_valid_report = os.path.join(valid_report_dir,
                                    config.experiment_report_name)
    best_valid_report = os.path.join(valid_report_dir,
                                     config.experiment_best_report_name)

    best_checkpoint = os.path.join(train_dir, 'model.ckpt-lowest_val_loss')
    best_checkpoint_meta = best_checkpoint + '.meta'
    best_epoch_file = os.path.join(train_dir, 'best_epoch.txt')

    with sv.managed_session(master) as sess:

      logger.info('Starting queue runners')
      sv.start_queue_runners(sess)

      def save_and_evaluate():
        """Save and evaluate the current model.

        Returns:
          path: the path string to the checkpoint.
          summary_df: pandas.DataFrame storing the evaluation result on the
            validation dataset with rows for each output name and columns for
            each metric value
        """
        logger.info('Saving model checkpoint')
        path = sv.saver.save(
            sess,
            sv.save_path,
            global_step=sv.global_step,
            write_meta_graph=True)
        tr_evaluator.run(path, eval_size)
        summary_df, _ = val_evaluator.run_and_report(
            tuner,
            path,
            eval_size,
            metrics_targets=metrics_targets,
            metrics_measures=metrics_measures)
        return path, summary_df

      def update_best_model(path, cur_epoch):
        """Update the records of the model with the lowest validation error.

        Args:
          path: the path to the checkpoint of the current model.
          cur_epoch: a integer of the current epoch
        """

        cur_checkpoint = path
        cur_checkpoint_meta = six.ensure_str(cur_checkpoint) + '.meta'

        gfile.Copy(cur_train_report, best_train_report, overwrite=True)
        gfile.Copy(cur_valid_report, best_valid_report, overwrite=True)
        gfile.Copy(cur_checkpoint, best_checkpoint, overwrite=True)
        gfile.Copy(cur_checkpoint_meta, best_checkpoint_meta, overwrite=True)
        with gfile.Open(best_epoch_file, 'w') as f:
          f.write(str(cur_epoch)+'\n')

      def compare_with_best_model(checkpoint_path, summary_df, cur_epoch):
        logger.info('Comparing current val loss with the best model')

        if not gfile.Exists(best_train_report):
          logger.info('No best model saved. Adding current model...')
          update_best_model(checkpoint_path, cur_epoch)
        else:
          with gfile.GFile(best_valid_report) as f:
            with xarray.open_dataset(f) as best_ds:
              best_ds.load()
          cur_loss = summary_df['loss'].loc['mean']
          best_loss = best_ds['loss'].mean('output')
          logger.info('Current val loss:%f', cur_loss)
          logger.info('The best val loss:%f', best_loss)
          if cur_loss < best_loss:
            logger.info(
                'Current model has lower loss. Updating the best model.')
            update_best_model(checkpoint_path, cur_epoch)
          else:
            logger.info('The best model has lower loss.')

      logger.info('Running eval before starting training')
      save_and_evaluate()

      try:
        for cur_epoch in trainer.train(sess, hps.epochs, num_batches_per_epoch):
          checkpoint_path, val_summary_df = save_and_evaluate()
          if (cur_epoch+1) % hps.epoch_interval_to_save_best == 0:
            compare_with_best_model(checkpoint_path, val_summary_df, cur_epoch)
          if tuner and tuner.should_trial_stop():
            break
      except eval_ff.TrainingDivergedException as error:
        logger.error('Training diverged: %s', str(error))
        infeasible = True
      else:
        infeasible = False

      logger.info('Saving final checkpoint')
      sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

  if tuner:
    # should be at the very end of execution, to avoid possible race conditions
    tuner.report_done(infeasible=infeasible)
    logger.info('report_done(infeasible=%r)', infeasible)

  logger.info('Done.')


def run_training_with_default_inputs(hps,
                                     train_dir,
                                     tuner=None,
                                     master='',
                                     val_fold_template='',
                                     metrics_targets=None,
                                     metrics_measures=None):
  """Start a training run with default inputs.

  Args:
    hps: tf.HParams with training parameters.
    train_dir: str path to train directory.
    tuner: optional hp_tuner.HPTuner.
    master: optional string to pass to a tf.Supervisor.
    val_fold_template: optional string to use to change the name of the
      validation fold data from its default.
    metrics_targets: String list of network targets to report metrics for.
    metrics_measures: Measurements about the performance of the network to
        report, e.g. 'auc/top_1p'.

  Raises:
    ValueError: Proto inconsistent with input feature.

  Returns:
    None.
  """
  gfile.MakeDirs(train_dir)

  task_log_path = os.path.join(train_dir, 'train_feedforward.log')
  io_utils.log_to_stderr_and_file(
      task_log_path, loggers=[logger, eval_ff.logger])

  logger.info('Setting up fold')
  experiment_proto, train_input_paths, val_input_paths = _setup_fold(
      hps.input_dir, train_dir, hps.val_fold, val_fold_template)

  if hps.additional_output:
    existing_ao = [ao.name for ao in experiment_proto.additional_output]
    for idx, name in enumerate(
        six.ensure_str(hps.additional_output).split(',')):
      if name not in existing_ao and name:
        experiment_proto.additional_output.add(
            name=name,
            measurement_id=idx)
  if data.STRUCTURE_PARTITION_FUNCTION in hps.input_features:
    if not experiment_proto.has_partition_function:
      raise ValueError(
          'invalid input_feature for proto lacking partition function: %s' %
          (data.STRUCTURE_PARTITION_FUNCTION))

  try:
    run_training(
        hps,
        experiment_proto,
        train_dir,
        train_input_paths,
        val_input_paths,
        tuner=tuner,
        master=master,
        metrics_targets=metrics_targets,
        metrics_measures=metrics_measures)
  except Exception:  # pylint: disable=broad-except
    # ensure errors end up in the logs
    logger.exception('Error encountered in run_training:')
    # re-raise to fail the task
    raise


def _copy_flags_to_hparams(hps):
  for name in HPARAM_FLAGS:
    hps.add_hparam(name, getattr(FLAGS, name))

  if FLAGS.dataset is None:
    hps.add_hparam('input_dir', FLAGS.input_dir)
    hps.add_hparam('affinity_target_map', FLAGS.affinity_target_map)
  else:
    dataset = FLAGS.dataset
    hps.add_hparam('dataset', dataset)
    hps.add_hparam('input_dir', config.INPUT_DATA_DIRS[dataset])
    hps.add_hparam('affinity_target_map', FLAGS.dataset)


def _train_with_autotune(root_dir):
  """Starts training using a tuner (i.e. Vizier).

  Args:
    root_dir: String directory to save the training results.
  """
  study_name = 'aptamer_ff.%s' % (FLAGS.study_name or FLAGS.run_name)
  client_handle = '%s/%s' % (study_name, FLAGS.task)
  tuner = tf.training.HPTuner(client_handle)
  tuner_goal = TUNER_LOSS_TO_GOAL[FLAGS.tuner_loss]
  study = _get_study(study_name, tuner_goal, FLAGS.tuner_algorithm,
                     FLAGS.num_conv_layers, FLAGS.max_strides, FLAGS.max_rates,
                     FLAGS.num_fc_layers, FLAGS.hpconfig)
  tuner.create_study(study)

  # if we have a dataset defined, grab the targets so we can report on them
  # in Vizier
  if FLAGS.dataset:
    metrics_targets = set()
    for t_list in config.DEFAULT_AFFINITY_TARGET_MAPS[FLAGS.dataset].values():
      for target in t_list:
        metrics_targets.add(target)
    metrics_targets = list(metrics_targets)
  else:
    metrics_targets = None

  # The standard approach of tuner.next_trial() is currently broken if
  # some workers restart (see b/64980341). The code below is a
  # workaround.
  while get_pending_or_stopping_trial_workaround(tuner):
    train_dir = '%s/%s' % (root_dir, tuner.trial_handle())
    hps = tuner.hparams()
    _copy_flags_to_hparams(hps)
    run_training_with_default_inputs(hps, train_dir, tuner, FLAGS.master,
                                     FLAGS.validation_fold_template)


# Code below is from the colab for the workaround for b/64980341
# pylint:disable=missing-docstring
# pylint:disable=protected-access
# pylint:disable=invalid-name
# To implement the fix (only need to modify HPTuner.next_trial):
#   1) Remove call to StudyIsDone()
#   2) Try GetSuggestions and catch VizierClientError,
#         returning False on exception.
def _next_trial(tuner):  # HPTuner.next_trial
  if not tuner._vizier:
    raise RuntimeError('Tuner is not part of a study.  Call create_study().')

  try:
    trial_suggestions = tuner._vizier.GetSuggestions(1)
  except vizier_client.VizierClientError:
    return False

  if not trial_suggestions:
    logging.info('Tuner failed to generate new trial suggestion')
    return False
  assert len(trial_suggestions) == 1
  tuner._current_trial = trial_suggestions[0]
  tuner._config_handle = str(tuner._current_trial.id)
  return True


def _get_trial(tuner, status_list):  # HPTuner._get_trial
  while _next_trial(tuner):
    if tuner._current_trial.status in status_list:
      return True
    time.sleep(2)
  return False


def get_pending_or_stopping_trial_workaround(
    tuner):  # HPTuner.get_pending_or_stopping_trial
  return _get_trial(tuner,
                    [vizier_pb2.Trial.PENDING, vizier_pb2.Trial.STOPPING])


# End of the workaround code
# pylint:enable=missing-docstring
# pylint:enable=protected-access
# pylint:enable=invalid-name


def _train_without_autotune(root_dir):
  """Trains without using Vizier.

  Args:
    root_dir: String path to the training directory.
  """
  tuner = None
  train_dir = '%s/%s' % (root_dir, FLAGS.task)
  hps = _default_study_hyperparams(FLAGS.num_conv_layers, FLAGS.num_fc_layers)
  hps = hps.parse(FLAGS.hpconfig)
  _copy_flags_to_hparams(hps)

  run_training_with_default_inputs(hps, train_dir, tuner, FLAGS.master,
                                   FLAGS.validation_fold_template)


def main(unused_argv):
  if (FLAGS.input_dir is None) == (FLAGS.dataset is None):
    raise ValueError('exactly one of --input_dir or --dataset required')

  invalid_input_features = set(FLAGS.input_features) - _VALID_INPUT_FEATURES
  if invalid_input_features:
    raise ValueError('invalid input_features: %r' % invalid_input_features)

  # use r=3 subdirectory to fix empty events files (see b/28535367)
  root_dir = os.path.join(FLAGS.save_base, FLAGS.run_name, 'r=3')
  gfile.MakeDirs(root_dir)

  if FLAGS.autotune:
    _train_with_autotune(root_dir)
  else:
    _train_without_autotune(root_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('save_base')
  flags.mark_flag_as_required('run_name')
  flags.mark_flag_as_required('num_fc_layers')
  app.run()
