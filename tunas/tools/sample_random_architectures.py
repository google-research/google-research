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

# Lint as: python2, python3
"""Tool for sampling a set of random network architectures."""


from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import sys

from absl import app
from absl import flags
from six.moves import map
import tensorflow.compat.v1 as tf

from tunas import controller
from tunas import cost_model_lib
from tunas import mobile_classifier_factory
from tunas import mobile_cost_model
from tunas import mobile_search_space_v3
from tunas import search_space_utils

flags.DEFINE_integer(
    'min_cost', 83,
    'Lower bound on the target inference time.')
flags.DEFINE_integer(
    'max_cost', 85,
    'Upper bound on the target inference time.')
flags.DEFINE_integer(
    'num_samples', 1,
    'Number of network architectures to sample.',
    lower_bound=1)
flags.DEFINE_string(
    'ssd', mobile_search_space_v3.PROXYLESSNAS_SEARCH,
    'Search space definition to use.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  ssd = FLAGS.ssd
  min_cost = FLAGS.min_cost
  max_cost = FLAGS.max_cost
  num_samples = FLAGS.num_samples

  model_spec = mobile_classifier_factory.get_model_spec(ssd)
  model_spec, _ = controller.independent_sample(model_spec)

  tf_indices = search_space_utils.tf_indices(model_spec)

  cost_model_features = mobile_cost_model.coupled_tf_features(model_spec)
  cost = cost_model_lib.estimate_cost(cost_model_features, ssd)

  outputs = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    num_attempts = 0
    while len(outputs) < num_samples:
      num_attempts += 1
      indices_value, cost_value = sess.run([tf_indices, cost])

      if min_cost <= cost_value <= max_cost:
        outputs.append({
            'indices': ':'.join(map(str, indices_value)),
            'cost': float(cost_value),
        })

      if num_attempts % 100 == 0 or len(outputs) == num_samples:
        print('generated {:d} samples, found {:d} / {:d} valid architectures'
              .format(num_attempts, len(outputs), num_samples))

  # Generate output in a formatted JSON style that's (hopefully) easy for
  # both computers and humans to read.
  json.dump(outputs, sys.stdout, indent=2)
  print()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.ERROR)
  tf.disable_v2_behavior()
  tf.app.run(main)
