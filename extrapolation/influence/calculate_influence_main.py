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

"""Main function for calculating influence of examples in a trained classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from extrapolation.influence import calculate_influence as ci

FLAGS = flags.FLAGS
flags.DEFINE_string('training_results_dir',
                    '/tmp',
                    'Main folder for experimental results.')
flags.DEFINE_string('clf_name',
                    'test_infl_cla',
                    'Name of pre-trained classifier.')
flags.DEFINE_integer('seed', 0, 'random seed for optimization')
flags.DEFINE_integer('n_train_infl', 50,
                     'Number of training examples to get influences of.')
flags.DEFINE_integer('n_test_infl', 50,
                     'Number of testing examples to get influences of.')
flags.DEFINE_integer('lissa_recursion_depth', 10,
                     'How many iterations to run LiSSA - i.e. the'
                     ' number of terms to compute in the Taylor approximation'
                     ' of the inverse Hessian.')
flags.DEFINE_float('lissa_scale', 100, 'scale for lissa optimization')
flags.DEFINE_float('lissa_damping', 0.01, 'damping for LiSSA')
flags.DEFINE_float('lam', 0.01, 'L2-weight regularization for LiSSA')
flags.DEFINE_string('conv_dims', '50,20',
                    'comma-separated list of integers for conv layer sizes')
flags.DEFINE_string('conv_sizes', '5,5',
                    'comma-separated list of integers for conv filter sizes')
flags.DEFINE_integer('n_classes', 10,
                     'number of classes in the dataset')
flags.DEFINE_integer('n_img_pairs', 10,
                     'number of image pairs to calculate influence for')
flags.DEFINE_string('output_dir', '', 'where to write results - defaults '
                    'to training_results_dir/clf_name/influence_results')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  params = FLAGS.flag_values_dict()
  ci.run(params)

if __name__ == '__main__':
  app.run(main)
