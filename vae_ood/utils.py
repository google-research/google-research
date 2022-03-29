# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Utilities for Tensorboard logging, checkpointing and Log Likelihood computation."""

import collections
import io

from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import seaborn as sns
import sklearn.metrics
import tensorflow as tf
import tqdm

from vae_ood import dataset_utils
from tensorflow.python.ops import summary_ops_v2  # pylint: disable=g-direct-tensorflow-import

# pylint: disable=g-bad-import-order


class TensorBoardWithLLStats(tf.keras.callbacks.TensorBoard):
  """Logs Log Likelihood statistics in the form of histograms and mean AUROC."""

  def __init__(self, eval_every, id_data, datasets, mode, normalize,
               visible_dist, **kwargs):
    super(TensorBoardWithLLStats, self).__init__(**kwargs)
    self.eval_every = eval_every
    self.id_data = id_data
    self.datasets = datasets
    self.mode = mode
    self.normalize = normalize
    self.visible_dist = visible_dist

  def write_ll_hist(self, epoch, probs_res):
    plt.subplot(2, 1, 1)
    for dataset in self.datasets:
      sns.distplot(probs_res['orig_probs'][dataset], label=dataset)
    plt.title('Log Likelihood')
    plt.legend()
    plt.subplot(2, 1, 2)
    for dataset in self.datasets:
      sns.distplot(probs_res['corr_probs'][dataset], label=dataset)
    plt.title('Corrected Log Likelihood')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    hist_img = tf.image.decode_png(buf.getvalue(), channels=4)
    hist_img = tf.expand_dims(hist_img, 0)

    with summary_ops_v2.always_record_summaries():
      with self._val_writer.as_default():
        summary_ops_v2.image('ll_hist', hist_img, step=epoch)

  def write_auc(self, epoch, probs_res):
    for probs in probs_res:
      score_sum = 0
      for dataset in self.datasets:
        if dataset == self.id_data:
          continue
        targets = np.concatenate([np.zeros_like(probs_res[probs][dataset]),
                                  np.ones_like(probs_res[probs][self.id_data])])
        lls = np.concatenate([probs_res[probs][dataset],
                              probs_res[probs][self.id_data]])
        score_sum += sklearn.metrics.roc_auc_score(targets, lls)
      with summary_ops_v2.always_record_summaries():
        with self._val_writer.as_default():
          summary_ops_v2.scalar(f'{probs}_auroc_mean',
                                score_sum/(len(self.datasets)-1),
                                step=epoch)

  def on_epoch_end(self, epoch, logs=None):
    logging.info('Epoch %d completed', epoch)
    if epoch != 0 and epoch % self.eval_every == 0:
      probs_res = get_probs(self.datasets,
                            self.model,
                            self.mode,
                            self.normalize,
                            n_samples=10,
                            split='val',
                            visible_dist=self.visible_dist)
      self.write_ll_hist(epoch, probs_res)
      self.write_auc(epoch, probs_res)
    super(TensorBoardWithLLStats, self).on_epoch_end(epoch, logs)


def cb_neglogprob(x, target):
  if x != 0.5:
    c = 2*np.arctanh(1-2*x)/(1-2*x)
  else:
    c = 2
  return -np.log(c * (x**target) * ((1-x)**(1-target)))


def get_correction_func():
  """Returns the correction function for a given visible distribution."""
  if not hasattr(get_correction_func, 'perf_recon_lp'):
    perf_recon_lp = {}
    targets = np.round(np.linspace(1e-3, 1-1e-3, 999), decimals=3)
    for target in targets:
      perf_recon_lp[target] = -cb_neglogprob(
          scipy.optimize.fmin(cb_neglogprob, 0.5, args=(target,))[0], target)
    get_correction_func.perf_recon_lp = perf_recon_lp

  def correction_func(test_batch):
    mapper = np.vectorize(lambda pix: get_correction_func.perf_recon_lp[pix])
    x = np.clip(test_batch, 1e-3, 1-1e-3).astype(float)
    r = np.round(x, decimals=3)
    return mapper(r).astype(np.float32)
  return correction_func


def get_probs(datasets,
              model,
              mode,
              normalize,
              n_samples,
              split='val',
              training=False,
              visible_dist='cont_bernoulli'):
  """Returns the log likelihoods of examples from given datasets using a given VAE.

  Args:
    datasets: A list of OOD dataset names
    model: A Keras VAE model
    mode: Load in "grayscale" or "color" modes
    normalize: Type of normalization to apply. Supported values are:
      None
      pctile-x (x is an integer)
      histeq
    n_samples: No. of samples to use to compute the importance weighted log
      likelihood estimate
    split: Data split to use for OOD log likelihood estimates,
    training: Get LL estimates using the model in training or eval mode
    visible_dist: VAE's Visible distribution

  Returns:
    A dictionay in the format:
    {
      'orig_probs': {
                      <dataset_name1>: <list of log likelihoods>,
                      <dataset_name2>: <list of log likelihoods>,
                      ...
                    }
      'corr_probs': {
                      <dataset_name1>: <list of log likelihoods>,
                      <dataset_name2>: <list of log likelihoods>,
                      ...
                    }
    }
  """
  logging.info('Computing Log Probs')
  orig_probs = collections.defaultdict(list)
  corr_probs = collections.defaultdict(list)

  for dataset in datasets:
    logging.info('Dataset: %s', dataset)
    if split == 'test':
      _, _, test = dataset_utils.get_dataset(
          dataset,
          100 // n_samples,
          mode=mode,
          normalize=normalize,
          dequantize=False,
          visible_dist=visible_dist)
    elif split == 'val':
      _, test, _ = dataset_utils.get_dataset(
          dataset,
          100 // n_samples,
          mode=mode,
          normalize=normalize,
          dequantize=False,
          visible_dist=visible_dist)
    else:
      test, _, _ = dataset_utils.get_dataset(
          dataset,
          100 // n_samples,
          mode=mode,
          normalize=normalize,
          dequantize=False,
          visible_dist=visible_dist)

    for test_batch in tqdm.tqdm(test):
      inp = test_batch[0]
      target = test_batch[1]
      probs = model.log_prob(
          inp,
          target,
          n_samples=n_samples,
          training=training)
      probs = probs.numpy()

      orig_probs[dataset].append(probs)

      target = target.numpy()
      if visible_dist == 'cont_bernoulli':
        target = (np.clip(target, 1e-3, 1-1e-3) * 1000).round().astype(np.int32)
      elif visible_dist == 'categorical':
        target = target.astype(np.int32)
        if model.inp_shape[-1] == 3:
          target[:, :, :, 1:] += 256
          target[:, :, :, 2:] += 256
      if visible_dist == 'gaussian':
        target[:, :, :, 1:] += 1
        target[:, :, :, 2:] += 1

      corr_probs[dataset].append(probs -
                                 model.correct(target).sum(axis=(1, 2, 3)))

    orig_probs[dataset] = np.concatenate(orig_probs[dataset], axis=0)
    corr_probs[dataset] = np.concatenate(corr_probs[dataset], axis=0)
  return {'orig_probs': orig_probs, 'corr_probs': corr_probs}


@tf.function
def get_pix_ll(batch, model):
  posterior = model.encoder(batch[0], training=False)
  code = posterior.mean()
  visible_dist = model.decoder(code, training=False)
  pix_ll = visible_dist.log_prob(batch[1])
  return pix_ll
