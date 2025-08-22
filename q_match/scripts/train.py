# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Training code to train from scratch using optimal hyperparameters.

Trains one algorithm for any number of random trials and evaluates results on
downstream tasks.
"""
import copy
import json
import os
import random
from typing import Sequence

from absl import app
from absl import flags
from clu import metric_writers
import jax
from ml_collections.config_flags import config_flags
import numpy as np
import pudb
import tensorflow as tf

from q_match.configs.enums import EvalTask
from q_match.factories.algorithms import get_algorithms
from q_match.factories.datasets import get_dataset_class
from q_match.factories.models import get_models
from q_match.tasks.tasks import eval_test_dataset


_INTERNAL = False
_BASE_DIR = 'q_match'
_RELATIVE_DIR = 'q_match'

config_flags.DEFINE_config_file(
    'config',
    (
        os.path.join(_RELATIVE_DIR, 'configs/one_percent_experiments_config.py')
    ),
    'Training configuration.',
    lock_config=True,
)

flags.DEFINE_enum_class('eval_task', EvalTask.BOTH, EvalTask,
                        'Evaluation task.')
flags.DEFINE_integer('num_trials', None,
                     'Number of random experiments per configuration.')

flags.DEFINE_string('algo', None, 'Name of algorithm.')
flags.DEFINE_string('dataset', None, 'Name of dataset.')
flags.DEFINE_integer('batch_size', None, 'Training batch size.')

flags.DEFINE_float('learning_rate', None, 'Supervised Learning rate')
flags.DEFINE_float('pretext_learning_rate', None, 'Pretext learning rate')
flags.DEFINE_integer('pretext_epochs', None, 'Number of pretext epochs')
flags.DEFINE_integer('supervised_epochs', None, 'Number of supervised epochs')
flags.DEFINE_float('pretext_weight_decay', None,
                   'Weight decay for pretext params.')
flags.DEFINE_float('supervised_weight_decay', None,
                   'Weight decay for supervised training.')
flags.DEFINE_integer('top_k', None, 'Number of nearest neighbors.')
flags.DEFINE_float('corruption_p', None,
                   'Probability of corrupting the input.')
flags.DEFINE_float('query_corruption_p', None,
                   'Probability of corrupting the input to the query.')
flags.DEFINE_integer('support_set_size', None,
                     'Support set size.')
flags.DEFINE_float('student_temperature', None,
                   'Student temp for dist match loss.')
flags.DEFINE_float('temperature', None,
                   'Temp for NN loss.')
flags.DEFINE_float('tau', None,
                   'Exponentential moving average weight.')
flags.DEFINE_float('label_smoothing', None,
                   'Label smoothing.')
flags.DEFINE_string('metric_log_dir', None, 'Where to log results.')
flags.DEFINE_boolean('debug', False, 'To use the pudb.')
flags.DEFINE_boolean('deterministic', True,
                     'Whether to use deterministic functions.')

flags.mark_flag_as_required('algo')
flags.mark_flag_as_required('dataset')

FLAGS = flags.FLAGS


def read_file(path):
  if not _INTERNAL:
    with open(path, mode='r') as f:
      contents = f.read()
    return contents
  return contents


def get_params(algo, dataset, eval_task):
  """Returns the optimal hyper parameters for a dataset-algo-task combo."""
  path = os.path.join(_BASE_DIR, 'scripts', 'one_percent_tuning.json')
  file_contents = read_file(path)
  params = json.loads(file_contents)
  return params[algo][dataset][eval_task]


def train(writer, trial_num=0):
  """Trains model."""
  dataset_path = FLAGS.config.dataset_path

  algo_name = FLAGS.algo
  dataset_name = FLAGS.dataset
  eval_task = FLAGS.eval_task
  if isinstance(eval_task, str):
    eval_task = EvalTask(eval_task)

  # Read already tuned params, and update the default parameters
  if eval_task != EvalTask.BOTH and dataset_name in FLAGS.config.datasets:
    tuned_params = get_params(algo_name, dataset_name, eval_task.value)
    FLAGS.config.update(tuned_params)

  if 'large_batch_' in algo_name:
    algo_name = algo_name.replace('large_batch_', '')

  # config params
  trial_seed = FLAGS.config.random_seed+trial_num
  logdir = FLAGS.metric_log_dir
  model_name = FLAGS.config.model
  patience = FLAGS.config.patience
  use_momentum_encoder = FLAGS.config.use_momentum_encoder
  pretext_weight_decay = FLAGS.config.pretext_weight_decay

  # Override any parameters with ones specificed from user at runtime.
  supervised_epochs = FLAGS.supervised_epochs or FLAGS.config.supervised_epochs
  pretext_epochs = FLAGS.pretext_epochs or FLAGS.config.pretext_epochs
  batch_size = FLAGS.batch_size or FLAGS.config.batch_size
  learning_rate = FLAGS.learning_rate or FLAGS.config.learning_rate
  pretext_learning_rate = (
      FLAGS.pretext_learning_rate or FLAGS.config.pretext_learning_rate
  )
  supervised_weight_decay = (
      FLAGS.supervised_weight_decay or FLAGS.config.supervised_weight_decay
  )
  corruption_p = FLAGS.corruption_p or FLAGS.config.corruption_p
  query_corruption_p = (
      FLAGS.query_corruption_p or FLAGS.config.query_corruption_p
  )
  student_temperature = (
      FLAGS.student_temperature or FLAGS.config.student_temperature
  )
  temperature = FLAGS.temperature or FLAGS.config.temperature
  support_set_size = FLAGS.support_set_size or FLAGS.config.support_set_size
  label_smoothing = FLAGS.label_smoothing or FLAGS.config.label_smoothing
  tau = FLAGS.tau or FLAGS.config.tau
  hparams = {'learning_rate': learning_rate,
             'pretext_learning_rate': pretext_learning_rate,
             'supervised_weight_decay': supervised_weight_decay,
             'pretext_weight_decay': pretext_weight_decay,
             'corruption_p': corruption_p,
             'query_corruption_p': query_corruption_p,
             'support_set_size': support_set_size,
             'student_temperature': student_temperature,
             'temperature': temperature,
             'label_smoothing': label_smoothing,
             'tau': tau,
             'batch_size': batch_size,
             }
  writer.write_hparams(hparams)

  if FLAGS.debug:
    pudb.set_trace()

  dataset_class = get_dataset_class(dataset_name)
  dataset = dataset_class(dataset_path=dataset_path, batch_size=batch_size)

  pretext_algorithm_class, supervised_algorithm_class = get_algorithms(
      algo_name)

  # set up the model and eval model
  model, eval_model = get_models(
      model_name=model_name,
      num_classes=dataset.get_num_classes(),
      algorithm=algo_name)

  # initialize params and state
  example_data = jax.numpy.array(dataset.get_example_features())
  model_key = jax.random.PRNGKey(trial_seed)
  dropout_key = jax.random.PRNGKey(FLAGS.config.random_seed+1)
  rngs = {'params': model_key, 'dropout': dropout_key}
  variables = model.init(rngs, example_data)
  state, params = variables.pop('params')

  # Pretext task
  pretext_params = params
  pretext_state = state
  if pretext_algorithm_class is not None:
    pretext_algo = pretext_algorithm_class(
        logdir=logdir,
        dataset=dataset,
        batch_size=batch_size,
        learning_rate=pretext_learning_rate,
        model=model,
        eval_model=eval_model,
        epochs=pretext_epochs,
        params=params,
        state=state,
        writer=writer,
        weight_decay=pretext_weight_decay,
        corruption_p=corruption_p,
        query_corruption_p=query_corruption_p,
        support_set_size=support_set_size,
        student_temperature=student_temperature,
        temperature=temperature,
        label_smoothing=label_smoothing,
        use_momentum_encoder=use_momentum_encoder,
        tau=tau,
    )
    params, state = pretext_algo.run()

    pretext_params = copy.deepcopy(params)
    pretext_state = copy.deepcopy(state)
    linear_over_features = False
  else:
    linear_over_features = True

  # Supervised task
  if  supervised_algorithm_class is not None:
    if eval_task == EvalTask.FTC or eval_task == EvalTask.BOTH:
      supervised = supervised_algorithm_class(
          logdir=logdir,
          dataset=dataset,
          batch_size=batch_size,
          learning_rate=learning_rate,
          model=model,
          eval_model=eval_model,
          epochs=supervised_epochs,
          params=pretext_params,
          state=pretext_state,
          writer=writer,
          patience=patience,
          weight_decay=supervised_weight_decay,
      )
      finetune_params, finetune_state = supervised.run()
    else:
      finetune_params = None
      finetune_state = None

    if eval_task == EvalTask.LC or eval_task == EvalTask.BOTH:
      supervised_linear = supervised_algorithm_class(
          logdir=logdir,
          dataset=dataset,
          batch_size=batch_size,
          learning_rate=learning_rate,
          model=model,
          eval_model=eval_model,
          epochs=supervised_epochs,
          params=pretext_params,
          state=pretext_state,
          writer=writer,
          patience=patience,
          weight_decay=supervised_weight_decay,
          linear_head=True,
          linear_over_features=linear_over_features)
      linear_params, linear_state = supervised_linear.run()
    else:
      linear_params = None
      linear_state = None

  test_results = dict()

  # eval tasks
  downstream_classification_results = eval_test_dataset(
      eval_model,
      finetune_params,
      finetune_state,
      linear_params,
      linear_state,
      dataset.get_test_epoch_iterator(),
      linear_over_features=linear_over_features,
      eval_task=eval_task,
  )
  test_results = {**test_results, **downstream_classification_results}

  print('Test Results: ', test_results)

  writer.write_scalars(trial_num, test_results)
  writer.flush()
  return test_results


def aggregate_results(results):
  """Aggregates the results using the mean and std.

  Args:
    results: List of dictionaries, all with the same keys.

  Returns:
    Means and standard deviations.
  """
  means = dict()
  stds = dict()
  keys = list(results[0].keys())
  for key in keys:
    values = []
    for result in results:
      values.append(result[key])
    values = np.array(values)
    means[key+'_mean'] = float(values.mean())
    stds[key+'_std'] = float(values.std())
  return means, stds


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  num_trials = FLAGS.num_trials or FLAGS.config.num_trials

  tf.config.experimental.set_visible_devices([], 'GPU')
  random_seed = FLAGS.config.random_seed
  if random_seed is not None and FLAGS.deterministic:
    tf.data.Options.deterministic = True
    tf.random.set_seed(random_seed)
    tf.random.set_global_generator(tf.random.Generator.from_seed(random_seed))
    tf.compat.v1.random.set_random_seed(random_seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(random_seed)
    np.random.seed(random_seed)

  writer = metric_writers.create_default_writer(
      FLAGS.metric_log_dir,
      asynchronous=True)

  results = []
  for trial in range(num_trials):
    results.append(train(writer, trial))

  means, stds = aggregate_results(results)
  writer.write_scalars(0, means)
  writer.write_scalars(0, stds)
  writer.flush()

  print('Means: \n', means)
  print('Stds: \n', stds)

if __name__ == '__main__':
  app.run(main)
