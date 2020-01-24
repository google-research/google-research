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

"""SQuAD experiment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pprint

import tensorflow as tf

from qanet import model_base  # pylint: disable=unused-import
from qanet import qanet_model  # pylint: disable=unused-import
from qanet.data import tf_data_pipeline  # pylint: disable=unused-import
from qanet.util import configurable
from tensorflow.contrib import training as contrib_training



def create_datasets(config, train_split, test_split, data_format='squad'):
  """Setup datasets.

  Args:
    config: a config describing the dataset
    train_split: name of train split
    test_split: name of test split
    data_format: 'squad' or 'squad2'

  Returns:
    A ConfigDict containing 4 keys:
      train_fn, test_fn - Input functions
      train_dataset, test_dataset - instantiated dataset objects
  """
  # Load dataset
  train_config = configurable.merge(config, split_name=train_split)
  eval_config = configurable.merge(config, split_name=test_split)
  dataset_class = configurable.Configurable.load(config)

  train = dataset_class.get_input_fn(
      mode='train', config=train_config, data_format=data_format)
  test = dataset_class.get_input_fn(
      mode='eval', config=eval_config, data_format=data_format)

  train_dataset = dataset_class(
      mode='train', config=train_config, data_format=data_format)
  test_dataset = dataset_class(
      mode='eval', config=eval_config, data_format=data_format)

  return dict(
      train_fn=train,
      test_fn=test,
      train_dataset=train_dataset,
      test_dataset=test_dataset)


class Experiment(object):
  """Container to help running experiments."""

  def __init__(self, estimator, train_spec, eval_spec, eval_frequency=None):
    self.estimator = estimator
    self.train_spec = train_spec
    self.eval_spec = eval_spec
    self.eval_frequency = eval_frequency

  def train(self, steps=None):
    max_steps = self.train_spec.max_steps
    if steps:
      max_steps = None
    else:
      steps = None
    tf.logging.info(max_steps)
    tf.logging.info(steps)
    return self.estimator.train(
        input_fn=self.train_spec.input_fn,
        max_steps=max_steps,
        steps=steps,
        hooks=self.train_spec.hooks)

  def predict(self):
    return self.estimator.predict(
        input_fn=self.eval_spec.input_fn,
        predict_keys=None,
        hooks=None,
        checkpoint_path=None,
        yield_single_examples=True)

  def evaluate(self):
    return self.estimator.evaluate(
        input_fn=self.eval_spec.input_fn,
        steps=self.eval_spec.steps,
        hooks=self.eval_spec.hooks,
        checkpoint_path=None,
        name=None)

  def train_and_evaluate(self):
    while True:
      self.train(steps=self.eval_frequency)
      metrics = self.evaluate()
      tf.logging.info(metrics)
      if metrics['global_step'] >= self.train_spec.max_steps:
        tf.logging.info('Hit max steps. Exiting train/eval loop.')
        break

  def get_distributed_spec(self):
    return tf.estimator.DistributedTrainingSpec(self.estimator, self.train_spec,
                                                self.eval_spec)


def create_experiment_fn(default_config,
                         return_distributed_spec=False,
                         run_config=None):
  """Create an experiment fn to pass to tf.Experiment.

  Args:
    default_config: The base config before any additional hparams from vizier
    return_distributed_spec: Whether to return
    run_config: A default run_config.

  Returns:
    A function that takes hparams to merge, and returns an Experiment.
  """

  def _create_config(hparams):
    """Create trial config and save to disk.

    Args:
      hparams: Nondefault params to merge

    Returns:
      A configurable object `spec` instantiated from the final config.
        spec.config will return the config.
    """
    hparams = hparams or contrib_training.HParams()
    tuned_config = configurable.unflatten_dict(hparams.values())
    pprinter = pprint.PrettyPrinter()
    tf.logging.info('Provided extra params:\n%s',
                    pprinter.pformat(tuned_config))
    try:
      merged_config = configurable.merge(default_config, tuned_config)
      tf.logging.info('Tuned default config:\n%s', merged_config)
    except TypeError as e:
      tf.logging.info(
          'Do not provide same config in both config string and vizier.'
          '  This may lead to type errors.')

      raise e

    # Instantiate a ConfigurableExperiment object.
    experiment_spec = configurable.Configurable.initialize(merged_config)
    tf.logging.info('Final config:\n%s', experiment_spec.config)

    return experiment_spec

  def _compute_steps(config, train_dataset, test_dataset):
    """Compute number of training and test steps based on config and data."""
    if config['train_steps']:
      # If train_steps is given, then ignore train_epochs and use it directly.
      train_steps = config['train_steps']
      tf.logging.info('Train dataset is %s examples', train_dataset.size)
      tf.logging.info('Test dataset is %s examples', test_dataset.size)
    else:
      # If train_steps is not given, compute it based on train_epochs * train
      # dataset size
      if not train_dataset.size:
        raise ValueError('Train dataset size not specified. '
                         'Manually specify steps.')
      train_steps = int(
          math.ceil(train_dataset.size / float(train_dataset.batch_size)) *
          config['train_epochs'])
      tf.logging.info('Train dataset is %s examples (%s batches)',
                      train_dataset.size,
                      train_dataset.size / float(train_dataset.batch_size))
      tf.logging.info('Test dataset is %s examples (%s batches)',
                      test_dataset.size,
                      test_dataset.size / float(test_dataset.batch_size))

    if config['eval_steps']:
      eval_steps = config['eval_steps']
    else:
      if not test_dataset.size:
        # If size is not specified, assumes that the dataset will throw an
        # exception when an epoch is done
        eval_steps = None
      else:
        eval_steps = int(math.ceil(test_dataset.size / test_dataset.batch_size))
    return train_steps, eval_steps

  def experiment_fn(hparams, model_dir=None):
    """Return an Experiment.

    Args:
      hparams: If not None, then these values are merged into default_config to
        create the final experiment spec.
      model_dir: Optional model directory.

    Returns:
      An Experiment instance.

    Raises:
      ValueError: If train_steps == 0 and the dataset does not specify its
        training split size to compute steps based on train_epochs
    """
    # Update config for Estimator
    inner_run_config = run_config or tf.estimator.RunConfig()
    inner_run_config = inner_run_config.replace(
        session_config=tf.ConfigProto(
            log_device_placement=default_config.get('log_device_placement',
                                                    False),
            allow_soft_placement=default_config.get('allow_soft_placement',
                                                    True)))

    # Create final trial config
    experiment_spec = _create_config(hparams=hparams)
    config = experiment_spec.config
    tf.logging.info('experiment_spec %s' % str(experiment_spec))
    tf.logging.info('config %s' % str(config))

    # Create input functions, along with dataset objects to provide metadata
    data = create_datasets(
        config['dataset'],
        config['train_split'],
        config['test_split'],
        data_format=config['model']['data_format'])
    train_steps, eval_steps = _compute_steps(config, data['train_dataset'],
                                             data['test_dataset'])
    ## Fetch model & create Estimator
    model = configurable.Configurable.load(config['model'])
    model_fn = model.get_model_fn(
        train_steps,
        # Provide dataset for access to vocabulary.
        dataset=data['train_dataset'],
        model_dir=inner_run_config.model_dir,
        use_estimator=True)

    ## Setup training time monitors
    train_monitors = [tf.train.StepCounterHook()]

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=inner_run_config,
        model_dir=model_dir,
        params=config['model'])

    train_spec = tf.estimator.TrainSpec(
        input_fn=data['train_fn'], hooks=train_monitors, max_steps=train_steps)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=data['test_fn'],
        # TODO(ddohan): Support exporting best
        steps=eval_steps,
        hooks=None,
        exporters=None)

    experiment = Experiment(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec,
        eval_frequency=config.eval_frequency)
    if return_distributed_spec:
      return experiment.get_distributed_spec()
    else:
      return experiment

  return experiment_fn


class ConfigurableExperiment(configurable.Configurable):
  """Run an experiment given a model and dataset."""

  @staticmethod
  def _config():
    return {
        'model': None,
        'dataset': None,
        'train_epochs': 1.0,
        'train_steps': 0,  # Overrides train_epochs
        'eval_steps': 0,  # if 0, do 1 epoch
        'eval_frequency': 0,  # if 0, eval only at end
        'train_split': 'train',
        'test_split': 'valid',
    }

  @property
  def metrics(self):
    """Get metrics for this experiment.

    Returns:
      A dictionary mapping keys to MetricSpec objects or a list of strings that
      are defined in the metrics.METRICS dict
    """
    raise NotImplementedError


class SQUADExperiment(ConfigurableExperiment):
  """Stanford Question Answering Dataset experiment."""

  @staticmethod
  def _config():
    config = ConfigurableExperiment._config()
    config.update({
        'model': qanet_model.QANet,
        'dataset': tf_data_pipeline.SQUADDatasetPipeline,
    })
    return config

  @property
  def metrics(self):
    return None
