# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Run a Variational Autoencoder on MNIST. Some code due to jamieas@."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from extrapolation.utils import dataset_utils
from extrapolation.utils import utils
from extrapolation.vae import run_vae_mnist
from extrapolation.vae.vae import VAE as VAE


tfd = tfp.distributions

flags.DEFINE_string('ood_classes', '5', 'a comma-separated list of labels'
                    'which will be considered Out-of-Distribution')
FLAGS = flags.FLAGS



def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('print this')

  params = FLAGS.flag_values_dict()
  plt.rcParams['savefig.format'] = params['mpl_format']

  params['results_dir'] = utils.make_subdir(
      params['results_dir'], params['expname'])
  params['figdir'] = utils.make_subdir(params['results_dir'], 'figs')
  params['sampledir'] = utils.make_subdir(params['figdir'], 'samples')
  params['ckptdir'] = utils.make_subdir(params['results_dir'], 'ckpts')
  params['logdir'] = utils.make_subdir(params['results_dir'], 'logs')
  params['tensordir'] = utils.make_subdir(
      params['results_dir'], 'tensors')

  ood_classes = [int(x) for x in params['ood_classes'].split(',')]
  # assume we train on all non-OOD classes
  n_classes = 10
  all_classes = range(n_classes)
  ind_classes = [x for x in all_classes if x not in ood_classes]
  (itr_train,
   itr_valid,
   itr_test,
   itr_test_ood) = dataset_utils.load_dset_ood_unsupervised(ind_classes,
                                                            ood_classes)

  conv_dims = [int(x) for x in params['conv_dims'].split(',')]
  conv_sizes = [int(x) for x in params['conv_sizes'].split(',')]
  vae = VAE(conv_dims, conv_sizes)

  run_vae_mnist.train_vae(vae, itr_train, itr_valid, params)
  run_vae_mnist.test_vae(vae, itr_test, params)

  params['tensordir'] = utils.make_subdir(
      params['results_dir'], 'ood_tensors')
  run_vae_mnist.test_vae(vae, itr_test_ood, params)


if __name__ == '__main__':
  app.run(main)
