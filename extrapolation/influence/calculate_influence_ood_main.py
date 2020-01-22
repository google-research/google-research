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

"""Run code from calculate_influence_ood.

Calculates influence function-type metrics on a pre-trained classifier with
some classes out-of-distribution.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import app
from absl import flags
from extrapolation.influence import calculate_influence_ood

FLAGS = flags.FLAGS

FLAGS = flags.FLAGS
flags.DEFINE_string('training_results_dir',
                    '/tmp',
                    'Main folder for experimental results.')
flags.DEFINE_string('clf_name',
                    'test_infl_cla',
                    'Name of pre-trained classifier.')
flags.DEFINE_integer('seed', 0, 'random seed for optimization')
flags.DEFINE_integer('n_test_infl', 50,
                     'Number of testing examples to get influences of.')
flags.DEFINE_integer('start_ix_test_infl', 0,
                     'where to start taking examples from the saved tensor.')
flags.DEFINE_float('lam', 0.01, 'L2-weight regularization')
flags.DEFINE_integer('cg_maxiter', 100, 'maximum number of iterations for CG')
flags.DEFINE_string('output_dir', '', 'where to write results - defaults '
                    'to training_results_dir/clf_name/influence_results')
flags.DEFINE_bool('squared', False, 'calculate the squared IHVP instead')
flags.DEFINE_float('tol', 1e-5, 'tolerance for CG optimization')
flags.DEFINE_string('tname', '', 'string to add to front of saved tensors')
flags.DEFINE_integer('hvp_samples', 1,
                     'how many batches to sample in HVP calculation')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  params = FLAGS.flag_values_dict()
  params['preloaded_model'] = None
  params['preloaded_itr'] = None
  calculate_influence_ood.calculate_influence_ood(params)


if __name__ == '__main__':
  app.run(main)
