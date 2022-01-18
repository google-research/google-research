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

"""CIFAR-10/CIFAR-100/SVHN input pipeline.
"""

from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3


def _augment_image(image, xlat=4, flip_lr=True):
  """Augment small image with random crop and h-flip.

  Args:
    image: image to augment
    xlat: random offset range
    flip_lr: if True perform random horizontal flip

  Returns:
    augmented image
  """
  if xlat > 0:
    # Pad with reflection padding
    # (See https://arxiv.org/abs/1605.07146)
    # Section 3
    image = tf.pad(image, [[xlat, xlat],
                           [xlat, xlat], [0, 0]], 'REFLECT')

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

  if flip_lr:
    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  return image


def _preprocess_train_image(image, mean_rgb, stddev_rgb):
  image = tf.cast(image, tf.float32)
  image = _augment_image(image)
  image = (image - mean_rgb) / stddev_rgb
  return image


def _preprocess_eval_image(image, mean_rgb, stddev_rgb):
  image = tf.cast(image, tf.float32)
  image = (image - mean_rgb) / stddev_rgb
  return image


class AbstractSmallImageDataSource(object):
  """Abstract small image data source."""

  MEAN_RGB = [0.5 * 255, 0.5 * 255, 0.5 * 255]
  STDDEV_RGB = [1.0, 1.0, 1.0]

  AUG_CROP_PADDING = 0
  AUG_FLIP_LR = False

  N_CLASSES = None
  TRAIN_IMAGES = None
  TEST_IMAGES = None

  def __init__(self, n_val, n_sup, train_batch_size, eval_batch_size,
               augment_twice, subset_seed=12345, val_seed=131):
    """Constructor.

    Args:
      n_val: number of validation samples to hold out from training set
      n_sup: number of samples for supervised learning
      train_batch_size: batch size for training
      eval_batch_size: batch_size for evaluation
      augment_twice: should unsupervised sample pairs be augmented differently
      subset_seed: the random seed used to choose the supervised samples
      val_seed: the random seed used to choose the hold out validation samples

    Attributes:
      n_train: number of training samples
      n_sup: number of supervised samples
      n_val: number of validation samples
      n_test: number of test samples
      train_semisup_ds: Semi-supervised training dataset
      train_unsup_ds: Unsupervised training dataset
      train_sup_ds: Supervised training dataset
      val_ds: Validation dataset
      test_ds: Test dataset
      n_classes: number of classes
    """
    mean_rgb = tf.constant(self.MEAN_RGB, shape=[1, 1, 3], dtype=tf.float32)
    stddev_rgb = tf.constant(self.STDDEV_RGB, shape=[1, 1, 3],
                             dtype=tf.float32)

    #
    # Get data
    #

    base_train_ds = self._load_train_set()

    @tf.function
    def get_train():
      return next(iter(base_train_ds.batch(self.TRAIN_IMAGES)))

    trainval = get_train()

    #
    # Split dataset into train and validation, if requested
    #

    if n_val > 0:
      train_val_splitter = StratifiedShuffleSplit(
          1, test_size=n_val, random_state=val_seed)
      train_ndx, val_ndx = next(train_val_splitter.split(
          trainval['label'], trainval['label']))
      X_train = trainval['image'].numpy()[train_ndx]  # pylint: disable=invalid-name
      y_train = trainval['label'].numpy()[train_ndx]
      X_val = trainval['image'].numpy()[val_ndx]  # pylint: disable=invalid-name
      y_val = trainval['label'].numpy()[val_ndx]
    else:
      X_train = trainval['image'].numpy()  # pylint: disable=invalid-name
      y_train = trainval['label'].numpy()
      X_val = None  # pylint: disable=invalid-name
      y_val = None

    train_ds = tf.data.Dataset.from_tensor_slices(
        {'image': X_train, 'label': y_train}
    ).cache()

    #
    # Select supervised subset
    #

    if n_sup == -1:
      n_sup = self.TRAIN_IMAGES

    if n_sup < self.TRAIN_IMAGES:
      splitter = StratifiedShuffleSplit(1, test_size=n_sup,
                                        random_state=subset_seed)
      _, sup_ndx = next(splitter.split(y_train, y_train))
      X_sup = X_train[sup_ndx]  # pylint: disable=invalid-name
      y_sup = y_train[sup_ndx]

      train_sup_ds = tf.data.Dataset.from_tensor_slices(
          {'image': X_sup, 'label': y_sup}
      ).cache()
    else:
      train_sup_ds = train_ds
      X_sup = X_train  # pylint: disable=invalid-name
      y_sup = y_train

    train_unsup_ds = train_ds

    train_sup_ds = train_sup_ds.repeat()
    train_sup_ds = train_sup_ds.shuffle(16 * train_batch_size)

    train_unsup_ds = train_unsup_ds.repeat()
    train_unsup_ds = train_unsup_ds.shuffle(16 * train_batch_size)

    train_semisup_ds = tf.data.Dataset.zip((train_sup_ds, train_unsup_ds))

    # Sample augmentation functions

    def _augment_sup(sup_sample):
      """Augment supervised sample."""
      sample = {
          'sup_image': _preprocess_train_image(
              sup_sample['image'], mean_rgb, stddev_rgb),
          'sup_label': sup_sample['label'],
      }
      return sample

    def _augment_unsup_once(unsup_sample):
      """Augment unsupervised sample, single augmentation."""
      unsup_x0 = _preprocess_train_image(
          unsup_sample['image'], mean_rgb, stddev_rgb)
      sample = {
          'unsup_image0': unsup_x0,
          'unsup_image1': unsup_x0,
      }
      return sample

    def _augment_unsup_twice(unsup_sample):
      """Augment unsupervised sample, two augmentations."""
      sample = {
          'unsup_image0': _preprocess_train_image(
              unsup_sample['image'], mean_rgb, stddev_rgb),
          'unsup_image1': _preprocess_train_image(
              unsup_sample['image'], mean_rgb, stddev_rgb),
      }
      return sample

    def _augment_semisup_once(sup_sample, unsup_sample):
      """Augment semi-supervised sample, single augmentation."""
      unsup_x0 = _preprocess_train_image(
          unsup_sample['image'], mean_rgb, stddev_rgb)
      semisup_sample = {
          'sup_image': _preprocess_train_image(
              sup_sample['image'], mean_rgb, stddev_rgb),
          'sup_label': sup_sample['label'],
          'unsup_image0': unsup_x0,
          'unsup_image1': unsup_x0,
      }
      return semisup_sample

    def _augment_semisup_twice(sup_sample, unsup_sample):
      """Augment semi-supervised sample, two augmentations."""
      semisup_sample = {
          'sup_image': _preprocess_train_image(
              sup_sample['image'], mean_rgb, stddev_rgb),
          'sup_label': sup_sample['label'],
          'unsup_image0': _preprocess_train_image(
              unsup_sample['image'], mean_rgb, stddev_rgb),
          'unsup_image1': _preprocess_train_image(
              unsup_sample['image'], mean_rgb, stddev_rgb),
      }
      return semisup_sample

    def _eval_map_fn(x):
      """Pre-process evaluation sample."""
      image = _preprocess_eval_image(x['image'], mean_rgb, stddev_rgb)
      batch = {'image': image, 'label': x['label']}
      return batch

    if augment_twice:
      train_semisup_ds = train_semisup_ds.map(_augment_semisup_twice,
                                              num_parallel_calls=128)
      train_unsup_only_ds = train_unsup_ds.map(_augment_unsup_twice,
                                               num_parallel_calls=128)
    else:
      train_semisup_ds = train_semisup_ds.map(_augment_semisup_once,
                                              num_parallel_calls=128)
      train_unsup_only_ds = train_unsup_ds.map(_augment_unsup_once,
                                               num_parallel_calls=128)
    train_sup_only_ds = train_sup_ds.map(_augment_sup,
                                         num_parallel_calls=128)

    train_semisup_ds = train_semisup_ds.batch(train_batch_size,
                                              drop_remainder=True)
    train_unsup_only_ds = train_unsup_only_ds.batch(train_batch_size,
                                                    drop_remainder=True)
    train_sup_only_ds = train_sup_only_ds.batch(train_batch_size,
                                                drop_remainder=True)
    train_semisup_ds = train_semisup_ds.prefetch(10)
    train_unsup_only_ds = train_unsup_only_ds.prefetch(10)
    train_sup_only_ds = train_sup_only_ds.prefetch(10)
    self.train_semisup_ds = train_semisup_ds
    self.train_unsup_ds = train_unsup_only_ds
    self.train_sup_ds = train_sup_only_ds

    #
    # Validation set
    #

    if n_val > 0:
      val_ds = tf.data.Dataset.from_tensor_slices(
          {'image': X_val, 'label': y_val}
      ).cache()

      val_ds = val_ds.map(_eval_map_fn, num_parallel_calls=128)
      val_ds = val_ds.batch(eval_batch_size)
      val_ds = val_ds.repeat()
      val_ds = val_ds.prefetch(10)
      self.val_ds = val_ds
    else:
      self.val_ds = None

    #
    # Test set
    #

    test_ds = self._load_test_set().cache()

    test_ds = test_ds.map(_eval_map_fn, num_parallel_calls=128)
    test_ds = test_ds.batch(eval_batch_size)
    test_ds = test_ds.repeat()
    test_ds = test_ds.prefetch(10)
    self.test_ds = test_ds

    self.n_train = len(y_train)
    self.n_val = n_val
    self.n_sup = len(y_sup)
    self.n_test = self.TEST_IMAGES
    self.n_classes = self.N_CLASSES

  def _load_train_set(self):
    raise NotImplementedError('Abstract')

  def _load_test_set(self):
    raise NotImplementedError('Abstract')


class CIFAR10DataSource(AbstractSmallImageDataSource):
  """CIFAR-10 data source."""
  TRAIN_IMAGES = 50000
  TEST_IMAGES = 10000
  N_CLASSES = 10

  MEAN_RGB = [0.4914 * 255, 0.4822 * 255, 0.4465 * 255]
  STDDEV_RGB = [0.2470 * 255, 0.2435 * 255, 0.2616 * 255]

  AUG_CROP_PADDING = 4
  AUG_FLIP_LR = True

  def _load_train_set(self):
    return tfds.load('cifar10', split='train')

  def _load_test_set(self):
    return tfds.load('cifar10', split='test')


class CIFAR100DataSource(AbstractSmallImageDataSource):
  """CIFAR-100 data source."""
  TRAIN_IMAGES = 50000
  TEST_IMAGES = 10000
  N_CLASSES = 100

  MEAN_RGB = [0.5071 * 255, 0.4866 * 255, 0.4409 * 255]
  STDDEV_RGB = [0.2673 * 255, 0.2564 * 255, 0.2761 * 255]

  AUG_CROP_PADDING = 4
  AUG_FLIP_LR = True

  def _load_train_set(self):
    return tfds.load('cifar100', split='train')

  def _load_test_set(self):
    return tfds.load('cifar100', split='test')


class SVHNDataSource(AbstractSmallImageDataSource):
  """SVHN data source."""
  TRAIN_IMAGES = 73257
  TEST_IMAGES = 26032
  N_CLASSES = 10

  MEAN_RGB = [0.4377 * 255, 0.4438 * 255, 0.4728 * 255]
  STDDEV_RGB = [0.1980 * 255, 0.2010 * 255, 0.1970 * 255]

  AUG_CROP_PADDING = 4
  AUG_FLIP_LR = False  # SVHN digits should *not* be flipped

  def _load_train_set(self):
    return tfds.load('svhn_cropped', split='train')

  def _load_test_set(self):
    return tfds.load('svhn_cropped', split='test')
