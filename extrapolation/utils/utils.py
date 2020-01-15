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

"""Utilities for monitoring and reporting results from models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf  # tf


def update_losses(losses):
  """Update losses with new values from this training step.

  Args:
    losses (list): list of tuples where the first item is the loss object, and
    the second item is the new value of that loss from this training step.
  """
  for loss, value in losses:
    loss.update(value)


def print_losses(loss_list, iter_num):
  """Print the current running average losses at this step.

  Args:
    loss_list (list): a list of RALosses.
    iter_num (int): the current training step.
  """
  s = 'Step={:d}: '.format(iter_num) + ', '.join(
      ['{}={:3f}'.format(l.name, l.get_value()) for l in loss_list])
  logging.info(s)


def save_tensors(tensor_list, tensor_dir):
  """Save some model output tensor.

  Args:
    tensor_list (list): tuples of RALoss (or string), tensor to save.
    tensor_dir (str): name of directory to save tensors in.
  """
  for identifier, tensor in tensor_list:
    tensor_fname = os.path.join(tensor_dir, '{}.npy'.format(
        identifier.name if hasattr(identifier, 'name') else identifier))
    with tf.io.gfile.GFile(tensor_fname, 'wb') as f:
      np.save(f, tensor.numpy(), allow_pickle=False)
      logging.info('Saving to %s', tensor_fname)


def write_losses_to_log(loss_list, iter_nums, logdir):
  """Write losses at steps in iter_nums for loss_list to log in logdir.

  Args:
    loss_list (list): a list of losses to write out.
    iter_nums (list): which steps to write the losses for.
    logdir (str): dir for log file.
  """
  for loss in loss_list:
    log_fname = os.path.join(logdir, '{}.csv'.format(loss.name))
    history = loss.get_history()
    with tf.io.gfile.GFile(log_fname, 'w' if 0 in iter_nums else 'a') as f:
      write_list = ['{:d},{:.7f}'.format(iter_num, history[iter_num])
                    for iter_num in iter_nums]
      write_str = '\n'.join(write_list) + '\n'
      if iter_nums[0] == 0:
        write_str = 'Iter_num,{}\n'.format(loss.name) + write_str
      f.write(write_str)


def write_metrics(metric_list, logdir, test_name):
  """Write the metrics in metric_list to a log file in logdir.

  Args:
    metric_list (list): what metrics to log.
    logdir (str): where the log file will be.
    test_name (str): string identified for part of the log file name.
  """
  log_fname = os.path.join(logdir, '{}_metrics.csv'.format(test_name))
  with tf.io.gfile.GFile(log_fname, 'w') as f:
    for mnm, mval in metric_list:
      f.write('{},{:.7f}\n'.format(mnm, mval))


def plot_losses(figdir, loss_list, mpl_format):
  """Plot the losses in loss_list in figures.

  Args:
    figdir (str): directory to save figures in.
    loss_list (list): list of RALosses to plot.
    mpl_format (str): figure format(e.g. pdf).
  """
  plt.rcParams['savefig.format'] = mpl_format
  start_window = 20
  skip_length = 50
  for l in loss_list:
    figname = os.path.join(figdir, '{}.{}'.format(l.name, mpl_format))
    plt.clf()
    l_data = l.get_history()
    l_data = l_data[start_window:] if len(l_data) > start_window else l_data
    plt.plot(l_data[::skip_length], ls='-')
    l_data_smooth = [l.get_value(i=i) for i in
                     range(start_window, len(l_data) - 1)]
    plt.plot(l_data_smooth[::skip_length], ls=':', lw=10)
    plt.yscale('log')
    with tf.io.gfile.GFile(figname, 'w') as f:
      plt.savefig(f)


def checkpoint_model(model, ckpt_dir, ckpt_name='bestmodel'):
  ckpt_name = os.path.join(ckpt_dir, ckpt_name)
  ckpt = tf.Checkpoint(model=model)
  save_path = ckpt.save(file_prefix=ckpt_name)
  return save_path, ckpt


def load_model(ckpt_path, model_class, model_args):
  """Load a checkpointed model.

  Args:
    ckpt_path (str): where the model is checkpointed.
    model_class (type): class of the checkpointed model.
    model_args (dict): named arguments to pass to the model initializer.
  Returns:
    model (model_class): the loaded model.
  """
  model = model_class(**model_args)
  ckpt = tf.Checkpoint(model=model)
  ckpt.restore(ckpt_path)
  return model


def load_tensor(fname):
  with tf.io.gfile.GFile(fname, 'rb') as f:
    df = np.load(f)
    if hasattr(f, 'keys'):
      return df['tensor']
    else:
      return df


def make_subdir(results_dir, dirname):
  """Make a new subdirectory for storing experimental results.

  Args:
    results_dir (str): head results directory.
    dirname (str): name of subdirectory.
  Returns:
    newpath (str): path to new subdirectory.
  """
  newpath = os.path.join(results_dir, dirname)
  if not os.path.exists(newpath):
    os.makedirs(newpath)
  return newpath


def plot_samples(sampledir, vae, data_iterator, mpl_format, num_samples=5):
  """Plot samples, reconstructions, and generated images.

  Args:
    sampledir (str): directory to save samples in.
    vae (VAE): a VAE.
    data_iterator (Iterator): a data iterator.
    mpl_format (str): what format to save mpl images in, e.g. 'pdf'.
    num_samples (int): the number of images to sample/reconstruct.

  """
  for i in range(num_samples):
    img = data_iterator.next()
    g = vae.generate()
    y, _ = vae(img)

    for tensor, fname in zip([img, tf.sigmoid(y), tf.sigmoid(g)],
                             ['{:d}_original.{}'.format(i, mpl_format),
                              '{:d}_recon.{}'.format(i, mpl_format),
                              '{:d}_sample.{}'.format(i, mpl_format)
                             ]):
      plt.clf()
      sns.heatmap(tensor[0, :, :, 0])
      figname = os.path.join(sampledir, fname)
      with tf.io.gfile.GFile(figname, 'w') as f:
        plt.savefig(f)


def save_images(fname, images, rows=1, titles=None):
  """Display a list of images in a single figure with matplotlib.

  Args:
    fname (str): filename to save the images at.
    images (list): list of 2-D tensors to save as images.
    rows (int): number of  rows to display images in.
    titles (list): subtitles for each image in images, should be same length.
  """
  plt.rcParams['savefig.format'] = fname.split('.')[-1]
  with tf.io.gfile.GFile(fname, 'w') as f:
    num_images = len(images)
    if titles is None:
      titles = ['Image {:d}'.format(i) for i in range(num_images)]

    fig = plt.figure()
    # We find the number of columns by rounding up since we may have
    # incomplete columns.
    cols = int(np.ceil(num_images / float(rows)))
    for i in range(len(images)):
      ax = fig.add_subplot(rows, cols, i + 1)
      if images[i].ndim == 2:
        plt.gray()
      plt.imshow(images[i])
      ax.set_title(titles[i])

    fig.set_size_inches(fig.get_size_inches() * np.array([cols, rows]))
    plt.savefig(f)


def aggregate_batches(itr, n, tensor_names):
  """Aggregate n examples from itr.

  Args:
    itr (Iterator): the iterator we want examples from.
    n (int): the number of examples we want.
    tensor_names (list): names of the tensors we're aggregating.
  Returns:
    tensor_dict (dict): a dictionary of tensor_name: tensor with n elements.
  """

  aggregated_tensor = {nm: [] for nm in tensor_names}
  ttl_examples = 0
  while ttl_examples < n:
    itr_tensors = itr.next()
    for t, tnm in zip(itr_tensors, tensor_names):
      aggregated_tensor[tnm].append(t)
    ttl_examples += itr_tensors[0].shape[0]
  for nm in aggregated_tensor:
    aggregated_tensor[nm] = tf.concat(aggregated_tensor[nm], 0)[:n]
  return aggregated_tensor


def write_json(fname, params):
  params_str = json.dumps(params, indent=4)
  with tf.io.gfile.GFile(fname, 'w') as f:
    f.write(params_str)


def load_json(fname):
  with tf.io.gfile.GFile(fname, 'r') as f:
    params = json.load(f)
  return params


