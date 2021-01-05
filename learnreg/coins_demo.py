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

"""Demo of TuneReg algorithm on coins problem defined in the paper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

from absl import app
from absl import flags
import numpy as np
import six
from six.moves import xrange
from typing import Dict, List

from learnreg import tuning_algorithms

flags.DEFINE_integer('num_coins', 100000, 'Number of coins')
flags.DEFINE_integer('num_training_flips_per_coin', 1,
                     'Number of observations for each coin in training data.')
flags.DEFINE_integer('num_test_flips_per_coin', 10,
                     'Number of observations for each coin in test data.')
flags.DEFINE_float('true_lambda', 1.0,
                   'Coin biases are drawn from a Beta(lambda,lambda) '
                   'distribution with given lambda.')
flags.DEFINE_integer('num_initial_random_hparams', 2,
                     'Number of initial random hyperparameter values to '
                     'sample in TuneReg algorithm.')
flags.DEFINE_integer('max_training_runs', 16, 'Max. number of training runs.')
flags.DEFINE_integer('random_seed', 314159, 'Seed for random number generator.')
flags.DEFINE_float('min_lambda', .1, 'Minimum value for regularization lambda.')
flags.DEFINE_float('max_lambda', 10, 'Maximum value for regularization lambda.')

FLAGS = flags.FLAGS


Dataset = Dict[int, List[bool]]  # from coin index to list of heads/tails
Model = Dict[int, float]  # from coin index to estimated Prob[heads]


def sample_coin_bias(true_lambda):
  # type: (float) -> float
  """Sample coin bias from Beta(lambda, lambda) distribution."""
  return np.random.beta(true_lambda, true_lambda)


def sample_examples(biases, num_flips, rand):
  # type: (Iterable[float], int, random.Random) -> Dataset
  """Samples a Dataset where each coin is flipped a given number of times."""
  return {
      i: [rand.random() < p for _ in xrange(num_flips)]
      for i, p in enumerate(biases)
  }


def sample_problem(rand):
  # type: (random.Random) -> Tuple[Dataset, Dataset]
  """Samples a (training Dataset, test Dataset) pair."""
  biases = [sample_coin_bias(FLAGS.true_lambda)
            for _ in xrange(FLAGS.num_coins)]
  training_examples = sample_examples(biases, FLAGS.num_training_flips_per_coin,
                                      rand)
  test_examples = sample_examples(biases, FLAGS.num_test_flips_per_coin, rand)
  return training_examples, test_examples


def sample_hparams(rand):
  """Samples dict from hyperparameter name to value."""
  ranges = {'lambda': (FLAGS.min_lambda, FLAGS.max_lambda)}
  return tuning_algorithms.sample_hparams(ranges, rand=rand)


def train(reg_lambda, examples):
  # type: (float, Dataset) -> Model
  """Returns a Model obtained by training with given regularization strength."""
  model = {}
  for i, outcomes in six.iteritems(examples):
    num_heads = sum(outcomes)
    num_flips = len(outcomes)
    # Training with log loss using the logit-beta regularizer with strength
    # reg_lambda produces the following probability:
    p = (num_heads + reg_lambda) / (num_flips + 2 * reg_lambda)
    model[i] = p
  return model


def log_loss_one_example(prob, outcome):
  # type: (float, int) -> float
  """Returns log loss of given prediction for given 0/1 outcome."""
  return - outcome*math.log(prob) - (1-outcome)*math.log(1-prob)


def log_loss(examples, model):
  """Returns total log loss of a Model on a Dataset."""
  sum_loss = 0.
  for i, outcomes in six.iteritems(examples):
    p = model[i]
    for outcome in outcomes:
      sum_loss += log_loss_one_example(p, int(outcome))
  return sum_loss


def logit_beta_regularizer_value(model):
  # type: (Model) -> float
  """Returns value of logit-Beta regularizer for a given Model."""
  return sum(
      -math.log(p) - math.log(1-p)
      for p in six.itervalues(model))


def get_data_point(reg_lambda, training_examples, test_examples):
  # type: (float, Dataset, Dataset) -> tuning_algorithms.DataPoint
  """Returns Datapoint for given training/test set and reg. strength."""
  model = train(reg_lambda, training_examples)

  average_test_loss = log_loss(test_examples, model) / len(test_examples)
  total_training_loss = log_loss(training_examples, model)
  feature_vector = [logit_beta_regularizer_value(model)]
  return tuning_algorithms.DataPoint(average_test_loss,
                                     total_training_loss,
                                     feature_vector)


def create_tuning_algorithm(rand):
  # type: (random.Random) -> tuning_algorithms.TuningAlgorithm
  """Returns TuningAlgorithm to use for experiment."""
  hparams = ['lambda']
  sample = lambda: sample_hparams(rand)
  initial_hparams = [sample()
                     for _ in xrange(FLAGS.num_initial_random_hparams)]
  algorithm = tuning_algorithms.TuneReg(
      hparams,
      initial_hparams,
      sample
  )
  return algorithm


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('True lambda =', FLAGS.true_lambda)

  rand = random.Random(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)

  training_examples, test_examples = sample_problem(rand)

  model = train(FLAGS.true_lambda, training_examples)
  avg_test_loss = log_loss(test_examples, model) / len(test_examples)
  print('Test log loss for model trained using true lambda as regularizer =',
        avg_test_loss)
  print('Note: this should be close to optimal when number of coins is large.')

  algorithm = create_tuning_algorithm(rand)

  def get_data_point_for_problem(hparam_dict):
    assert list(hparam_dict.keys()) == ['lambda'], hparam_dict
    reg_lambda = hparam_dict['lambda']
    return get_data_point(reg_lambda, training_examples, test_examples)

  # Run tuning algorithm
  iter_pairs = algorithm.run(get_data_point_for_problem)
  best_lambda = None
  best_test_loss = None
  for i, (hparam_dict, data_point) in enumerate(iter_pairs):
    if (best_test_loss is None or
        data_point.test_loss < best_test_loss):
      best_test_loss = data_point.test_loss
      best_lambda = hparam_dict['lambda']

    n = i + 1
    if n & (n - 1) == 0:
      print('#training runs =', n, 'best lambda =', best_lambda,
            'best test loss =', best_test_loss)
    if n >= FLAGS.max_training_runs:
      break


if __name__ == '__main__':
  app.run(main)
