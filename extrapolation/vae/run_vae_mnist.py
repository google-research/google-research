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

"""Run a Variational Autoencoder on MNIST. Some code due to jamieas@."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf  # tf
import tensorflow_probability as tfp
from extrapolation.utils import dataset_utils
from extrapolation.utils import utils
from extrapolation.utils.running_average_loss import RunningAverageLoss as RALoss
from extrapolation.vae.vae import VAE as VAE

tfd = tfp.distributions


flags.DEFINE_string('expname', 'temp', 'name of this experiment directory')
flags.DEFINE_integer('max_steps', 1000, 'number of steps of optimization')
flags.DEFINE_integer('max_steps_test', 10, 'number of steps of testing')
flags.DEFINE_integer('run_avg_len', 50,
                     'number of steps of average losses over')
flags.DEFINE_integer('print_freq', 50, 'number of steps between printing')
flags.DEFINE_float('lr', 0.001, 'Adam learning rate')
flags.DEFINE_string('conv_dims', '80,40,20',
                    'comma-separated list of integers for conv layer sizes')
flags.DEFINE_string('conv_sizes', '5,5,5',
                    'comma-separated list of integers for conv filter sizes')
flags.DEFINE_string('mpl_format', 'pdf',
                    'format to save matplotlib  figures in, also '
                    'becomes filename extension')
flags.DEFINE_string('results_dir', '/tmp',
                    'main folder for experimental results')
flags.DEFINE_integer('patience', 50, 'steps of patience for early stopping')
flags.DEFINE_integer('seed', 0, 'random seed for Tensorflow')

FLAGS = flags.FLAGS



def train_vae(vae, itr_train, itr_valid, params):
  """Train a VAE.

  Args:
    vae (VAE): a VAE.
    itr_train (Iterator): an iterator over training data.
    itr_valid (Iterator): an iterator over validation data.
    params (dict): flags for training.

  """
  run_avg_len = params['run_avg_len']
  max_steps = params['max_steps']
  print_freq = params['print_freq']

  # RALoss is an object which tracks the running average of a loss.
  ra_loss = RALoss('elbo', run_avg_len)
  ra_kl = RALoss('kl', run_avg_len)
  ra_recon = RALoss('recon', run_avg_len)
  ra_trainloss = RALoss('train-elbo', run_avg_len)

  min_val_loss = sys.maxsize
  min_val_step = 0
  opt = tf.train.AdamOptimizer(learning_rate=params['lr'])
  finished_training = False
  start_printing = 0
  for i in range(max_steps):
    batch = itr_train.next()
    with tf.GradientTape() as tape:
      train_loss, _, _ = vae.get_loss(batch)
      mean_train_loss = tf.reduce_mean(train_loss)

    val_batch = itr_valid.next()
    valid_loss, kl_loss, recon_loss = vae.get_loss(val_batch)
    loss_list = [ra_loss, ra_kl, ra_recon, ra_trainloss]
    losses = zip(loss_list,
                 [tf.reduce_mean(l) for l in
                  (valid_loss, kl_loss, recon_loss, train_loss)])
    utils.update_losses(losses)

    grads = tape.gradient(mean_train_loss, vae.weights)
    opt.apply_gradients(zip(grads, vae.weights))

    curr_ra_loss = ra_loss.get_value()
    # Early stopping: stop training when validation loss stops decreasing.
    # The second condition ensures we don't checkpoint every step early on.
    if curr_ra_loss < min_val_loss and \
        i - min_val_step > params['patience'] / 10:
      min_val_loss = curr_ra_loss
      min_val_step = i
      save_path, ckpt = utils.checkpoint_model(vae, params['ckptdir'])
      logging.info('Step {:d}: Checkpointed to {}'.format(i, save_path))
    elif i - min_val_step > params['patience'] or i == max_steps - 1:
      ckpt.restore(save_path)
      logging.info('Best validation loss was {:.3f} at step {:d}'
                   ' - stopping training'.format(min_val_loss, min_val_step))
      finished_training = True

    if i % print_freq == 0 or finished_training:
      utils.print_losses(loss_list, i)
      utils.write_losses_to_log(loss_list, range(start_printing, i + 1),
                                params['logdir'])
      start_printing = i + 1
      utils.plot_losses(params['figdir'], loss_list, params['mpl_format'])
      utils.plot_samples(
          params['sampledir'], vae, itr_valid, params['mpl_format'])

    if finished_training:
      break


def test_vae(vae, itr_test, params):
  """Test a trained VAE."""

  max_steps_test = params['max_steps_test']

  ra_loss = RALoss('elbo', max_steps_test)
  ra_kl = RALoss('kl', max_steps_test)
  ra_recon = RALoss('recon', max_steps_test)

  loss_tensor = []
  kl_tensor = []
  recon_tensor = []
  for i in range(max_steps_test):
    batch = itr_test.next()
    loss, kl, recon = vae.get_loss(batch)

    losses = zip([ra_loss, ra_kl, ra_recon],
                 [tf.reduce_mean(l) for l in
                  (loss, kl, recon)])
    utils.update_losses(losses)
    utils.print_losses([l[0] for l in losses], i)
    loss_tensor.append(loss)
    kl_tensor.append(kl)
    recon_tensor.append(recon)

  loss_tensor = tf.concat(loss_tensor, 0)
  kl_tensor = tf.concat(kl_tensor, 0)
  recon_tensor = tf.concat(recon_tensor, 0)
  utils.save_tensors(
      zip([ra_loss, ra_kl, ra_recon],
          [loss_tensor, kl_tensor, recon_tensor]), params['tensordir'])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('print this')

  params = FLAGS.flag_values_dict()
  tf.set_random_seed(params['seed'])
  plt.rcParams['savefig.format'] = params['mpl_format']

  params['results_dir'] = utils.make_subdir(params['results_dir'],
                                            params['expname'])
  params['figdir'] = utils.make_subdir(params['results_dir'], 'figs')
  params['sampledir'] = utils.make_subdir(params['figdir'], 'samples')
  params['ckptdir'] = utils.make_subdir(params['results_dir'], 'ckpts')
  params['logdir'] = utils.make_subdir(params['results_dir'], 'logs')
  params['tensordir'] = utils.make_subdir(params['results_dir'], 'tensors')

  itr_train, itr_valid, itr_test = dataset_utils.load_dset_unsupervised()

  conv_dims = [int(x) for x in params['conv_dims'].split(',')]
  conv_sizes = [int(x) for x in params['conv_sizes'].split(',')]
  vae = VAE(conv_dims, conv_sizes)

  train_vae(vae, itr_train, itr_valid, params)
  test_vae(vae, itr_test, params)


if __name__ == '__main__':
  app.run(main)
