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

"""Data utils for cifar."""

import copy
import os
import pickle
import augmentation_transforms
import freq_helpers
import numpy as np
import policies as found_policies
import scipy.io as sio
from six.moves import cPickle
import tensorflow as tf
import tensorflow_datasets as tfds


def load_arr(prefix, filename):
  pathname = os.path.join(prefix, filename)
  assert os.path.exists(pathname)
  with open(pathname, 'r') as f:
    arr = np.loadtxt(f)
  return arr


def load_obj(prefix, filename):
  pathname = os.path.join(prefix, filename)
  assert os.path.exists(pathname)
  with open(pathname, 'rb') as f:
    return pickle.load(f)


def load_cifar(hparams):
  """Load the cifar10/cifar100 dataset according to hparams.

  Args:
    hparams: Dictionary of hyperparameters

  Returns:
    all_data: A tensor of type uint8 of shape [50000, 32, 32, 3] that contains
      all of the training and validation images.
    all_labels: A tensor containing the one hot labels of type int32 of
      shape [50000, num_classes].
    test_images: A tensor of type uint8 of shape [10000, 32, 32, 3] that
      contains all of the test images if `eval_test` is set to 1 else it will
      be an empty matrix.
    test_labels: A tensor containing the one hot labels of type int32 of
      shape [10000, num_classes] if `eval_test` is set to 1 else it will
      be an empty matrix.
    extra_test_images: CIFAR10.1 images, normalized using CIFAR10 stats (or
      None, if `extra_dataset` is not 'cifar10_1').
    extra_test_labels: CIFAR10.1 one-hot labels (or None, if `extra_dataset` is
      not 'cifar10_1').
  """
  all_labels = []

  total_batches_to_load = 5
  assert hparams.train_size + hparams.validation_size <= 50000
  if hparams.eval_test:
    total_batches_to_load += 1
  # Determine how many images we have loaded
  total_dataset_size = 50000
  train_dataset_size = total_dataset_size
  if hparams.eval_test:
    total_dataset_size += 10000

  if hparams.dataset == 'cifar10':
    all_images = []
  elif hparams.dataset == 'cifar100':
    all_images = np.empty((1, 50000, 3072), dtype=np.uint8)
    if hparams.eval_test:
      test_data = np.empty((1, 10000, 3072), dtype=np.uint8)
  if hparams.dataset == 'cifar10':
    datafiles = [
        'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
        'data_batch_5']

    if hparams.eval_test:
      datafiles.append('test_batch')
    num_classes = 10
  elif hparams.dataset == 'cifar100':
    datafiles = ['train']
    if hparams.eval_test:
      datafiles.append('test')
    num_classes = 100
  else:
    raise NotImplementedError('Unimplemented dataset: ', hparams.dataset)
  if hparams.dataset != 'test':
    for file_num, f in enumerate(datafiles):
      d = unpickle(os.path.join(hparams.data_path, f))
      if hparams.dataset == 'cifar10':
        labels = np.array(d['labels'])
      else:
        labels = np.array(d['fine_labels'])
      if f == 'test':
        test_data[0] = copy.deepcopy(d['data'])
        if hparams.dataset == 'cifar10':
          all_images.append(test_data)
        else:
          all_images = np.concatenate([all_images, test_data], axis=1)
      else:
        if hparams.dataset == 'cifar10':
          all_images.append(copy.deepcopy(d['data']))
        else:
          all_images[file_num] = copy.deepcopy(d['data'])
      nsamples = len(labels)
      for idx in range(nsamples):
        all_labels.append(labels[idx])
  if hparams.dataset == 'cifar10':
    all_images = np.concatenate(all_images, axis=0)
  all_images = all_images.reshape(-1, 3072)
  all_images = all_images.reshape(-1, 3, 32, 32)  # pylint: disable=too-many-function-args
  all_images = all_images.transpose(0, 2, 3, 1).copy()
  all_images = all_images / 255.0
  mean = augmentation_transforms.MEANS
  std = augmentation_transforms.STDS
  tf.logging.info('mean:{}    std: {}'.format(mean, std))
  all_images = (all_images - mean) / std
  all_labels = np.eye(num_classes)[np.array(all_labels, dtype=np.int32)]

  assert len(all_images) == len(all_labels)
  tf.logging.info(
      'In CIFAR10 loader, number of images: {}'.format(len(all_images)))

  extra_test_images = None
  extra_test_labels = None
  if hparams.extra_dataset == 'cifar10_1':
    extra_test_ds = tfds.as_numpy(
        tfds.load('cifar10_1', split='test', batch_size=-1))
    extra_test_images = ((extra_test_ds['image'] / 255.0) - mean) / std
    extra_test_labels = np.eye(num_classes)[np.array(
        extra_test_ds['label'], dtype=np.int32)]

  # Break off test data
  if hparams.eval_test:
    test_images = all_images[train_dataset_size:]
    test_labels = all_labels[train_dataset_size:]
  else:
    test_images = []
    test_labels = []
  all_images = all_images[:train_dataset_size]
  all_labels = all_labels[:train_dataset_size]
  return all_images, all_labels, test_images, test_labels, extra_test_images, extra_test_labels


def load_svhn(hparams):
  """Load the svhn dataset according to hparams.

  Args:
    hparams: Dictionary of hyperparameters

  Returns:
    all_data: A tensor of type uint8 of shape [50000, 32, 32, 3] that contains
      all of the training and validation images.
    all_labels: A tensor containing the one hot labels of type int32 of
      shape [50000, num_classes].
    test_images: A tensor of type uint8 of shape [10000, 32, 32, 3] that
      contains all of the test images if `eval_test` is set to 1 else it will
      be an empty matrix.
    test_labels: A tensor containing the one hot labels of type int32 of
      shape [10000, num_classes] if `eval_test` is set to 1 else it will
      be an empty matrix.
  """
  all_labels = []
  all_labels = []
  data_path = hparams.data_path
  train_data = unpickle(os.path.join(data_path, 'train_32x32.mat'),
                        use_sio=True)
  all_images = train_data['X']
  all_labels = train_data['y']

  all_labels = all_labels[:, 0]-1
  all_images = all_images.transpose(3, 0, 1, 2).copy()
  num_classes = 10

  all_labels = np.eye(num_classes)[np.array(all_labels, dtype=np.int32)]
  assert len(all_images) == len(all_labels)
  tf.logging.info(
      'In SVHN loader, number of images: {}'.format(len(all_images)))

  if hparams.eval_test:
    test_data = unpickle(os.path.join(data_path, 'test_32x32.mat'),
                         use_sio=True)
    test_images = test_data['X'].transpose(3, 0, 1, 2).copy()
    test_images = test_images[:26025]
    test_labels = test_data['y'][:, 0]-1
    test_labels = test_labels[:26025]
    test_labels = np.eye(num_classes)[np.array(test_labels, dtype=np.int32)]
  else:
    test_images = []
    test_labels = []
  # Normalize data
  mean = np.reshape([0.4376821, 0.4437697, 0.47280442], [1, 1, 1, 3])
  std = np.reshape([0.19803012, 0.20101562, 0.19703614], [1, 1, 1, 3])
  all_images = all_images / 255.0
  all_images = (all_images - mean) / std
  test_images = test_images / 255.0
  test_images = (test_images - mean) / std
  tf.compat.v1.logging.info('svhn: {} {} {} {}'.format(
      all_images, all_labels, test_images, test_labels))

  return all_images, all_labels, test_images, test_labels


class DataSet(object):
  """Dataset object that produces augmented training and eval data."""

  def __init__(self, hparams):
    self.hparams = hparams
    self.epochs = 0
    self.curr_train_index = 0

    if self.hparams.noise_type == 'fourier':
      i, j = freq_helpers.get_spatial_freqij(self.hparams.spatial_frequency)
      self.direction = freq_helpers.get_fourier_basis_image(i, j)
    elif self.hparams.noise_type == 'random':
      np.random.seed(hparams.noise_seed)
      self.direction = np.random.randn(32*32*3).reshape(32, 32, 3)
    elif self.hparams.noise_type == 'f' or self.hparams.noise_type == '1/f':
      self.direction = freq_helpers.get_fourier_composite_image(
          kind=self.hparams.noise_type)

    self.good_policies = found_policies.good_policies()

    (all_images, all_labels, test_images, test_labels, extra_test_images,
     extra_test_labels) = load_cifar(hparams)
    self.test_images, self.test_labels = test_images, test_labels
    self.extra_test_images, self.extra_test_labels = extra_test_images, extra_test_labels

    # Shuffle the data
    all_images = all_images[:]
    all_labels = all_labels[:]
    tf.logging.info('all_images size: {}'.format(all_images.shape))
    np.random.seed(0)
    perm = np.arange(len(all_images))
    np.random.shuffle(perm)
    all_images = all_images[perm]
    all_labels = all_labels[perm]

    # Break into train and val
    train_size, val_size = hparams.train_size, hparams.validation_size
    assert 50000 >= train_size + val_size
    self.train_images = all_images[:train_size]
    self.train_labels = all_labels[:train_size]
    self.val_images = all_images[train_size:train_size + val_size]
    self.val_labels = all_labels[train_size:train_size + val_size]
    self.num_train = self.train_images.shape[0]

  def next_batch(self):
    """Return the next minibatch of augmented data."""
    next_train_index = self.curr_train_index + self.hparams.batch_size
    if next_train_index > self.num_train:
      # Increase epoch number
      epoch = self.epochs + 1
      self.reset()
      self.epochs = epoch
    batched_data = (
        self.train_images[self.curr_train_index:self.curr_train_index +
                          self.hparams.batch_size],
        self.train_labels[self.curr_train_index:self.curr_train_index +
                          self.hparams.batch_size])
    final_imgs = []
    images, labels = batched_data
    if self.hparams.augment_type == 'mixup':
      images, labels = augmentation_transforms.mixup_batch(
          images, labels, self.hparams.mixup_alpha)
    elif self.hparams.augment_type == 'image_freq':
      images, labels = augmentation_transforms.freq_augment(
          images,
          labels,
          amplitude=self.hparams.freq_augment_amplitude,
          magnitude=self.hparams.augmentation_magnitude,
          proportion_f=self.hparams.freq_augment_ffrac,
          probability=self.hparams.augmentation_probability)
    for data in images:
      if self.hparams.augment_type == 'autoaugment':
        epoch_policy = self.good_policies[np.random.choice(
            len(self.good_policies))]
        final_img = augmentation_transforms.apply_policy(epoch_policy, data)
      elif self.hparams.augment_type == 'random':
        epoch_policy = found_policies.random_policy(
            self.hparams.num_augmentation_layers,
            self.hparams.augmentation_magnitude,
            self.hparams.augmentation_probability)
        final_img = augmentation_transforms.apply_policy(epoch_policy, data)
      else:
        final_img = np.copy(data)
      if self.hparams.apply_flip_crop:
        final_img = augmentation_transforms.random_flip(
            augmentation_transforms.zero_pad_and_crop(data, 4))
      # Apply cutout
      if self.hparams.apply_cutout:
        final_img = augmentation_transforms.cutout_numpy(final_img)

      final_imgs.append(final_img)
    final_imgs = np.array(final_imgs, np.float32)
    if self.hparams.noise_type == 'radial':
      labels = augmentation_transforms.add_radial_noise(
          final_imgs, labels, self.hparams.frequency, self.hparams.amplitude,
          self.hparams.noise_class, self.hparams.normalize_amplitude)
    elif self.hparams.noise_type == 'random' or self.hparams.noise_type == 'fourier' or self.hparams.noise_type == 'f' or self.hparams.noise_type == '1/f':
      labels = augmentation_transforms.add_sinusoidal_noise(
          final_imgs, labels, self.hparams.frequency, self.hparams.amplitude,
          self.direction, self.hparams.noise_class,
          self.hparams.normalize_amplitude)
    elif self.hparams.noise_type == 'uniform':
      labels = augmentation_transforms.add_uniform_noise(
          labels, self.hparams.amplitude, self.hparams.noise_class)

    batched_data = (final_imgs, labels)
    self.curr_train_index += self.hparams.batch_size
    return batched_data

  def reset(self):
    """Reset training data and index into the training data."""
    self.epochs = 0
    # Shuffle the training data
    perm = np.arange(self.num_train)
    np.random.shuffle(perm)
    assert self.num_train == self.train_images.shape[
        0], 'Error incorrect shuffling mask'
    self.train_images = self.train_images[perm]
    self.train_labels = self.train_labels[perm]
    self.curr_train_index = 0


def unpickle(f, use_sio=False):
  tf.logging.info('loading file: {}'.format(f))
  fo = tf.gfile.Open(f, 'rb')
  if use_sio:
    d = sio.loadmat(fo)
  else:
    d = cPickle.load(fo, encoding='latin1')
  fo.close()
  return d
