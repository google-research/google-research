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

"""Evaluate a model along paths between input images."""

import os
import sys
from absl import app
from absl import flags
from absl import logging
import numpy as np
import scipy
import tensorflow as tf
sys.path.insert(1, '../label_smoothing')
import data_utils as data_utils_cifar  # pylint: disable=g-import-not-at-top, g-bad-import-order
import helper_utils  # pylint: disable=g-bad-import-order
from shake_drop import build_shake_drop_model  # pylint: disable=g-bad-import-order
from shake_shake import build_shake_shake_model  # pylint: disable=g-bad-import-order
from wrn import build_wrn_model  # pylint: disable=g-bad-import-order

flags.DEFINE_integer('num_pairs', 200, 'Number of image pairs per pair of '
                     'classes.')
flags.DEFINE_string('modelname', 'wrn_32', 'Name of the model to use.')
flags.DEFINE_integer('class_1', 1,
                     'Class that the first example should come from.')
flags.DEFINE_integer('class_2', 1,
                     'Class that the second example should come from.')
flags.DEFINE_string('interp_type', 'mixup',
                    'Type of interpolation. Only supports `mixup`, which'
                    'denotes linear interpolation.')
flags.DEFINE_integer('batch_size', 200, 'Batch size for model evaluation.')
flags.DEFINE_float('sampling_distance', 1,
                   'Distance between adjacent samples in interpolation.')
flags.DEFINE_integer('num_samples', None,
                     'Number of samples to take along each interpolation path.'
                     'If None, sampling_distance is used instead.')
flags.DEFINE_string('dsname', 'cifar10', 'Dataset to use. Must be cifar10.')
flags.DEFINE_string('split', 'test', 'Dataset split; train or test.')
flags.DEFINE_string(
    'dirname', '/tmp/training', 'Directory name where the model is saved.')
flags.DEFINE_integer(
    'ckpt_num', None, 'If provided, use the checkpoint at this epoch number.'
    'Otherwise (default), use the most recent checkpoint.')
FLAGS = flags.FLAGS

prefix = './results'


def npsave(filename, arr):
  """Save NumPy array to a specified path."""
  path = os.path.join(prefix, filename)
  with open(path, 'wb') as f:
    return np.save(f, arr)


def peak_fd(values, sampling_distance):
  shifted = np.roll(values, -1)
  diffs = np.abs((values - shifted)[0:-1])
  return np.max(diffs) / sampling_distance


def batch_interpolation(model, dataset, projection=None):
  """Dataset is a list of batches where each batch contains two images to be interpolated."""
  # Precompute all the images we want to run through the model
  all_images = []
  all_ts = []
  for batch in dataset:
    x0 = batch['image'][0]
    x1 = batch['image'][1]
    distance = np.linalg.norm((x0 - x1).flatten())
    sampling_dist = FLAGS.sampling_distance
    if FLAGS.num_samples is not None:
      sampling_dist = distance / FLAGS.num_samples
    ts = np.arange(0, distance, sampling_dist, dtype=np.float32)
    sampling_locations = (1 - ts[:, None, None, None] / distance) * x0[
        None, Ellipsis] + (ts[:, None, None, None] / distance) * x1[None, Ellipsis]
    if projection is not None:
      sampling_locations, new_ts = projection(sampling_locations)
      if new_ts is not None:
        ts = new_ts
    all_images.append(sampling_locations)
    all_ts.append(ts)
  all_images = np.concatenate(all_images, axis=0)
  all_ts = np.concatenate(all_ts)
  idx = 0
  all_logits = []
  while idx*FLAGS.batch_size < all_images.shape[0]:
    stop = (idx+1)*FLAGS.batch_size
    if stop > all_images.shape[0]:
      stop = all_images.shape[0]
    batch = all_images[idx*FLAGS.batch_size:stop, Ellipsis]
    logits = model(batch)
    all_logits.append(logits)
    idx += 1
  logits = np.concatenate(all_logits, axis=0)
  prob_spectra = []
  max_grads = []
  max_distances = []
  logit_distances = []
  logits_list = []
  idx = list(np.where(all_ts == 0)[0])
  idx.append(-1)
  for i in range(len(idx) - 1):
    logitsi = logits[idx[i]:idx[i+1], Ellipsis]
    probs = scipy.special.softmax(logitsi)
    prob_fds = np.linalg.norm(probs - np.roll(probs, -1, axis=0), axis=1)[0:-1]
    max_grad = np.max(prob_fds) / FLAGS.sampling_distance
    prob_distances = np.linalg.norm(probs - probs[0, :], axis=1)
    logit_distance = np.linalg.norm(logitsi[0] - logitsi[-1])
    prob_spectrum = np.abs(np.fft.fftshift(np.fft.fft(prob_distances)))**2
    prob_spectra.append(prob_spectrum)
    max_grads.append(max_grad)
    max_distances.append(np.max(prob_distances))
    logit_distances.append(logit_distance)
    logits_list.append(logitsi)
  return prob_spectra, max_grads, max_distances, logit_distances, logits_list


def aggregate_interp(dataset, model, projection, dataset2=None, numpy=False):
  """Group images into a desired number of pairs for interpolation."""
  if numpy:
    if dataset2 is not None:
      ds = []
      images1, labels1 = dataset
      images2, labels2 = dataset2
      idx = 0
      while True:
        imgs = np.stack([images1[idx, Ellipsis], images2[idx, Ellipsis]], axis=0)
        labels = np.stack([labels1[idx, Ellipsis], labels2[idx, Ellipsis]], axis=0)
        ds.append({'image': imgs, 'label': labels})
        idx += 1
        if idx >= FLAGS.num_pairs:
          break
    else:
      ds = []
      images, labels = dataset
      idx = 0
      counter = 0
      while True:
        imgs = np.stack([images[idx, Ellipsis], images[idx + 1, Ellipsis]], axis=0)
        lbls = np.stack([labels[idx, Ellipsis], labels[idx + 1, Ellipsis]], axis=0)
        ds.append({'image': imgs, 'label': lbls})
        idx += 2
        counter += 1
        if counter >= FLAGS.num_pairs:
          break
  else:
    if dataset2 is not None:
      # zip together the two datasets so we get one image of each dataset
      # in each batch of 2 images
      ds1 = dataset.batch(1)
      ds2 = dataset2.batch(1)
      ds = []
      counter = 0
      for (batch1, batch2) in zip(ds1, ds2):
        imgs1 = batch1['image'].numpy()
        labels1 = batch1['label'].numpy()
        imgs2 = batch2['image'].numpy()
        labels2 = batch2['label'].numpy()
        imgs = np.stack([imgs1, imgs2], axis=0)
        labels = np.stack([labels1, labels2], axis=0)
        ds.append({'image': imgs, 'label': labels})
        counter += 1
        if counter >= FLAGS.num_pairs:
          break
    else:
      ds = []
      counter = 0
      for batch in dataset.batch(2):
        ds.append(batch)
        counter += 1
        if counter >= FLAGS.num_pairs:
          break
  spectra, max_grads, max_prob_dists, logit_dists, logits_list = batch_interpolation(
      model,
      ds,
      projection)
  return spectra, max_grads, max_prob_dists, logit_dists, logits_list


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not os.path.isdir(prefix):
    os.makedirs(prefix)
  if FLAGS.dsname == 'cifar10':
    hparams = tf.contrib.training.HParams(
        train_size=50000,
        validation_size=0,
        eval_test=True,
        dataset='cifar10',
        data_path='./cifar10_data/',
        extra_dataset='cifar10_1',
        use_batchnorm=1,
        use_fixup=0,
        use_gamma_swish=0)
    if FLAGS.modelname == 'wrn_32':
      setattr(hparams, 'model_name', 'wrn')
      hparams.add_hparam('wrn_size', 32)
    elif FLAGS.modelname == 'wrn_160':
      setattr(hparams, 'model_name', 'wrn')
      hparams.add_hparam('wrn_size', 160)
    elif FLAGS.modelname == 'shake_shake_32':
      setattr(hparams, 'model_name', 'shake_shake')
      hparams.add_hparam('shake_shake_widen_factor', 2)
    elif FLAGS.modelname == 'shake_shake_96':
      setattr(hparams, 'model_name', 'shake_shake')
      hparams.add_hparam('shake_shake_widen_factor', 6)
    elif FLAGS.modelname == 'shake_shake_112':
      setattr(hparams, 'model_name', 'shake_shake')
      hparams.add_hparam('shake_shake_widen_factor', 7)
    elif FLAGS.modelname == 'pyramid_net':
      setattr(hparams, 'model_name', 'pyramid_net')
      hparams.batch_size = 64
    (all_images, all_labels, test_images, test_labels, extra_test_images,
     extra_test_labels) = data_utils_cifar.load_cifar(hparams)
    images = test_images
    labels = test_labels
    if FLAGS.split == 'train':
      images = all_images
      labels = all_labels
    elif FLAGS.split == 'extra':
      images = extra_test_images
      labels = extra_test_labels
    images1 = images[np.argmax(labels, axis=-1) == FLAGS.class_1, Ellipsis]
    labels1 = labels[np.argmax(labels, axis=-1) == FLAGS.class_1, Ellipsis]

    g = tf.Graph()
    with g.as_default():
      inputs = tf.placeholder('float', [None, 32, 32, 3])
      scopes = helper_utils.setup_arg_scopes(is_training=False, hparams=hparams)
      with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        with helper_utils.nested(*scopes):
          if hparams.model_name == 'pyramid_net':
            logits, hiddens = build_shake_drop_model(
                inputs, num_classes=10, is_training=False)
          elif hparams.model_name == 'wrn':
            logits, hiddens = build_wrn_model(
                inputs, num_classes=10, hparams=hparams)
          elif hparams.model_name == 'shake_shake':
            logits, hiddens = build_shake_shake_model(
                inputs, num_classes=10, hparams=hparams, is_training=False)
          else:
            print(f'unrecognized hparams.model_name: {hparams.model_name}')
            assert 0

    sess = tf.InteractiveSession(graph=g)
    if FLAGS.ckpt_num is None:
      ckpt = tf.train.latest_checkpoint(os.path.join(FLAGS.dirname, 'model'))
    else:
      ckpt = os.path.join(FLAGS.dirname, 'model',
                          'modelckpt.ckpt-' + str(FLAGS.ckpt_num))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt)
    model = lambda imgs: logits.eval(feed_dict={inputs: imgs})

    if FLAGS.class_2 != FLAGS.class_1:
      images2 = images[np.argmax(labels, axis=-1) == FLAGS.class_2, Ellipsis]
      labels2 = labels[np.argmax(labels, axis=-1) == FLAGS.class_2, Ellipsis]
      spectra, max_grads, max_prob_dists, logit_dists, logits_list = aggregate_interp(
          dataset=(images1, labels1),
          model=model,
          projection=None,
          dataset2=(images2, labels2),
          numpy=True)
    else:
      spectra, max_grads, max_prob_dists, logit_dists, logits_list = aggregate_interp(
          dataset=(images1, labels1),
          model=model,
          projection=None,
          dataset2=None,
          numpy=True)
  else:
    logging.warn('unsupported dataset')
    assert False

  # Save the outputs to files
  filename = FLAGS.modelname + '_' + str(
      FLAGS.class_1) + FLAGS.interp_type + str(FLAGS.sampling_distance) + str(
          FLAGS.class_2) + '_' + str(FLAGS.num_pairs)
  if FLAGS.dsname == 'cifar10':
    filename = filename + '_dir_' + FLAGS.dirname.replace('/', '.')
  if FLAGS.ckpt_num is not None:
    filename = filename + '_ckpt' + str(FLAGS.ckpt_num)
  npsave(filename + '_logitdists.npz', logit_dists)
  npsave(filename + '_logitslist.npz', logits_list)


if __name__ == '__main__':
  app.run(main)
