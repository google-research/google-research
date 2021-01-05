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

# Lint as: python2, python3
"""Binary that runs inference on a pre-trained cost model."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

from tunas import mobile_cost_model
from tunas import mobile_search_space_v3
from tunas import search_space_utils

flags.DEFINE_string(
    'indices', '',
    'Colon-separated list of integers specifying the network architecture '
    'to evaluate.')
flags.DEFINE_string(
    'ssd', mobile_search_space_v3.PROXYLESSNAS_SEARCH,
    'Search space definition to use.')


FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  indices = search_space_utils.parse_list(FLAGS.indices, int)
  ssd = FLAGS.ssd
  cost = mobile_cost_model.estimate_cost(indices, ssd)
  print('estimated cost: {:f}'.format(cost))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.ERROR)
  tf.disable_v2_behavior()
  tf.app.run(main)
