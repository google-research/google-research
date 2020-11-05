# coding=utf-8
"""Loader for datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

from absl import flags
from ieg.dataset_utils.utils import cifar_process
from ieg.dataset_utils.utils import imagenet_preprocess_image
import numpy as np
import sklearn.metrics as sklearn_metrics
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS


def verbose_data(which_set, data, label):
  """Prints the number of data per class for a dataset.

  Args:
    which_set: a str
    data: A numpy 4D array
    label: A numpy array
  """
  text = ['{} size: {}'.format(which_set, data.shape[0])]
  for i in range(label.max() + 1):
    text.append('class{}-{}'.format(i, len(np.where(label == i)[0])))
  text.append('\n')
  text = ' '.join(text)
  tf.logging.info(text)


def shuffle_dataset(data, label, others=None, class_balanced=False):
  """Shuffles the dataset with class balancing option.

  Args:
    data: A numpy 4D array.
    label: A numpy array.
    others: Optional array corresponded with data and label.
    class_balanced: If True, after shuffle, data of different classes are
      interleaved and balanced [1,2,3...,1,2,3.].

  Returns:
    Shuffled inputs.
  """
  if class_balanced:
    sorted_ids = []

    for i in range(label.max() + 1):
      tmp_ids = np.where(label == i)[0]
      np.random.shuffle(tmp_ids)
      sorted_ids.append(tmp_ids)

    sorted_ids = np.stack(sorted_ids, 0)
    sorted_ids = np.transpose(sorted_ids, axes=[1, 0])
    ids = np.reshape(sorted_ids, (-1,))

  else:
    ids = np.arange(data.shape[0])
    np.random.shuffle(ids)

  if others is None:
    return data[ids], label[ids]
  else:
    return data[ids], label[ids], others[ids]


def load_asymmetric(x, y, noise_ratio, n_val, random_seed=12345):
  """Create asymmetric noisy data."""

  def _generate_asymmetric_noise(y_train, n):
    """Generate cifar10 asymmetric label noise.

    Asymmetric noise confuses
      automobile <- truck
      bird -> airplane
      cat <-> dog
      deer -> horse

    Args:
      y_train: label numpy tensor
      n: noise ratio

    Returns:
      corrupted y_train.
    """
    assert y_train.max() == 10 - 1
    classes = 10
    p = np.eye(classes)

    # automobile <- truck
    p[9, 9], p[9, 1] = 1. - n, n
    # bird -> airplane
    p[2, 2], p[2, 0] = 1. - n, n
    # cat <-> dog
    p[3, 3], p[3, 5] = 1. - n, n
    p[5, 5], p[5, 3] = 1. - n, n
    # automobile -> truck
    p[4, 4], p[4, 7] = 1. - n, n
    tf.logging.info('Asymmetric corruption p:\n {}'.format(p))

    noise_y = y_train.copy()
    r = np.random.RandomState(random_seed)

    for i in range(noise_y.shape[0]):
      c = y_train[i]
      s = r.multinomial(1, p[c, :], 1)[0]
      noise_y[i] = np.where(s == 1)[0]

    actual_noise = (noise_y != y_train).mean()
    assert actual_noise > 0.0

    return noise_y

  n_img = x.shape[0]
  n_classes = 10

  # holdout balanced clean
  val_idx = []
  if n_val > 0:
    for cc in range(n_classes):
      tmp_idx = np.where(y == cc)[0]
      val_idx.append(
          np.random.choice(tmp_idx, n_val // n_classes, replace=False))
    val_idx = np.concatenate(val_idx, axis=0)

  train_idx = list(set([a for a in range(n_img)]).difference(set(val_idx)))
  if n_val > 0:
    valdata, vallabel = x[val_idx], y[val_idx]
  traindata, trainlabel = x[train_idx], y[train_idx]
  trainlabel = trainlabel.squeeze()
  label_corr_train = trainlabel.copy()

  trainlabel = _generate_asymmetric_noise(trainlabel, noise_ratio)

  if len(trainlabel.shape) == 1:
    trainlabel = np.reshape(trainlabel, [trainlabel.shape[0], 1])

  traindata, trainlabel, label_corr_train = shuffle_dataset(
      traindata, trainlabel, label_corr_train)
  if n_val > 0:
    valdata, vallabel = shuffle_dataset(valdata, vallabel, class_balanced=True)
  else:
    valdata, vallabel = None, None
  return (traindata, trainlabel, label_corr_train), (valdata, vallabel)


def load_train_val_uniform_noise(x, y, n_classes, n_val, noise_ratio):
  """Make noisy data and holdout a clean val data.

  Constructs training and validation datasets, with controllable amount of
  noise ratio.

  Args:
    x: 4D numpy array of images
    y: 1D/2D numpy array of labels of images
    n_classes: The number of classes.
    n_val: The number of validation data to holdout from train.
    noise_ratio: A float number that decides the random noise ratio.

  Returns:
    traindata: Train data.
    trainlabel: Train noisy label.
    label_corr_train: True clean label.
    valdata: Validation data.
    vallabel: Validation label.
  """
  n_img = x.shape[0]
  val_idx = []
  if n_val > 0:
    # Splits a clean holdout set
    for cc in range(n_classes):
      tmp_idx = np.where(y == cc)[0]
      val_idx.append(
          np.random.choice(tmp_idx, n_val // n_classes, replace=False))
    val_idx = np.concatenate(val_idx, axis=0)

  train_idx = list(set([a for a in range(n_img)]).difference(set(val_idx)))
  # split validation set
  if n_val > 0:
    valdata, vallabel = x[val_idx], y[val_idx]
  traindata, trainlabel = x[train_idx], y[train_idx]
  # Copies the true label for verification
  label_corr_train = trainlabel.copy()
  # Adds uniform noises
  mask = np.random.rand(len(trainlabel)) <= noise_ratio
  random_labels = np.random.choice(n_classes, mask.sum())
  trainlabel[mask] = random_labels[Ellipsis, np.newaxis]
  # Shuffles dataset
  traindata, trainlabel, label_corr_train = shuffle_dataset(
      traindata, trainlabel, label_corr_train)
  if n_val > 0:
    valdata, vallabel = shuffle_dataset(valdata, vallabel, class_balanced=True)
  else:
    valdata, vallabel = None, None
  return (traindata, trainlabel, label_corr_train), (valdata, vallabel)


class CIFAR(object):
  """CIFAR dataset class."""

  def __init__(self):
    self.dataset_name = FLAGS.dataset
    self.is_cifar100 = 'cifar100' in self.dataset_name
    if self.is_cifar100:
      self.num_classes = 100
    else:
      self.num_classes = 10
    self.noise_ratio = float(self.dataset_name.split('_')[-1])
    assert self.noise_ratio >= 0 and self.noise_ratio <= 1,\
        'The schema {} of dataset is not right'.format(self.dataset_name)
    self.split_probe = FLAGS.probe_dataset_hold_ratio != 0

  def create_loader(self):
    """Creates loader as tf.data.Dataset."""
    # load data to memory.
    if self.is_cifar100:
      (x_train, y_train), (x_test,
                           y_test) = tf.keras.datasets.cifar100.load_data()
    else:
      (x_train, y_train), (x_test,
                           y_test) = tf.keras.datasets.cifar10.load_data()

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    x_train, y_train = shuffle_dataset(x_train, y_train)
    n_probe = int(math.floor(x_train.shape[0] * FLAGS.probe_dataset_hold_ratio))

    # TODO(zizhaoz): add other noise types.
    if 'asymmetric' in self.dataset_name:
      assert 'cifar100' not in self.dataset_name, 'Asymmetric only has CIFAR10'
      (x_train, y_train, y_gold), (x_probe, y_probe) = load_asymmetric(
          x_train,
          y_train,
          noise_ratio=self.noise_ratio,
          n_val=n_probe,
          random_seed=FLAGS.seed)
    elif 'uniform' in self.dataset_name:
      (x_train, y_train, y_gold), (x_probe,
                                   y_probe) = load_train_val_uniform_noise(
                                       x_train,
                                       y_train,
                                       n_classes=self.num_classes,
                                       noise_ratio=self.noise_ratio,
                                       n_val=n_probe)
    else:
      assert self.dataset_name in ['cifar10', 'cifar100']

    if not self.split_probe and x_probe is not None:
      # Usually used for supervised comparison.
      tf.logging.info('Merge train and probe')
      x_train = np.concatenate([x_train, x_probe], axis=0)
      y_train = np.concatenate([y_train, y_probe], axis=0)
      y_gold = np.concatenate([y_gold, y_probe], axis=0)

    conf_mat = sklearn_metrics.confusion_matrix(y_gold, y_train)
    conf_mat = conf_mat / np.sum(conf_mat, axis=1, keepdims=True)
    tf.logging.info('Corrupted confusion matirx\n {}'.format(conf_mat))
    x_test, y_test = shuffle_dataset(x_test, y_test)
    self.train_dataset_size = x_train.shape[0]
    self.val_dataset_size = x_test.shape[0]
    if self.split_probe:
      self.probe_size = x_probe.shape[0]

    input_tuple = (x_train, y_train.squeeze())
    self.train_dataflow = self.create_ds(input_tuple, is_train=True)
    self.val_dataflow = self.create_ds((x_test, y_test.squeeze()),
                                       is_train=False)
    if self.split_probe:
      self.probe_dataflow = self.create_ds((x_probe, y_probe.squeeze()),
                                           is_train=True)

    tf.logging.info('Init [{}] dataset loader'.format(self.dataset_name))
    verbose_data('train', x_train, y_train)
    verbose_data('test', x_test, y_test)
    if self.split_probe:
      verbose_data('probe', x_probe, y_probe)

    return self

  def create_ds(self, data, is_train=True):
    """Creates tf.data object given data.

    Args:
      data: data in format of tuple, e.g. (data, label)
      is_train: bool indicate train stage the original copy, so the resulting
        tensor is 5D

    Returns:
      An tf.data.Dataset object
    """
    ds = tf.data.Dataset.from_tensor_slices(data)
    map_fn = lambda x, y: (cifar_process(x, is_train), y)
    ds = ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


class WebVision(object):
  """Webvision dataset class."""

  def __init__(self, root, version='webvisionmini', use_imagenet_as_eval=False):
    self.version = version
    self.num_classes = 50 if 'mini' in version else 1000
    self.root = root
    self.image_size = 224
    self.use_imagenet_as_eval = use_imagenet_as_eval

    default_n_per_class = 10
    if '_' in FLAGS.dataset:
      self.probe_size = int(FLAGS.dataset.split('_')[1]) * self.num_classes
    else:
      # Uses default ones, assume there is a dataset saved
      self.probe_size = default_n_per_class * self.num_classes
    self.probe_folder = 'probe_' + str(self.probe_size)

  def wrapper_map_probe_v2(self, tfrecord):
    """tf.data.Dataset map function for probe data v2.

    Args:
      tfrecord: serilized by tf.data.Dataset.

    Returns:
      A map function
    """

    def _extract_fn(tfrecord):
      """Extracts the functions."""

      features = {
          'image/encoded': tf.FixedLenFeature([], tf.string),
          'image/label': tf.FixedLenFeature([], tf.int64)
      }
      example = tf.parse_single_example(tfrecord, features)
      image, label = example['image/encoded'], tf.cast(
          example['image/label'], dtype=tf.int32)

      return [image, label]

    image_bytes, label = _extract_fn(tfrecord)
    label = tf.cast(label, tf.int64)

    image = imagenet_preprocess_image(
        image_bytes, is_training=True, image_size=self.image_size)

    return image, label

  def wrapper_map_v2(self, train):
    """tf.data.Dataset map function for train data v2."""

    def _func(data):
      img, label = data['image'], data['label']
      image_bytes = tf.image.encode_jpeg(img)
      image_1 = imagenet_preprocess_image(
          image_bytes, is_training=train, image_size=self.image_size)
      if train:
        image_2 = imagenet_preprocess_image(
            image_bytes,
            is_training=train,
            image_size=self.image_size,
            autoaugment_name='v0',
            use_cutout=True)
        images = tf.concat(
            [tf.expand_dims(image_1, 0),
             tf.expand_dims(image_2, 0)], axis=0)
      else:
        images = image_1
      return images, label

    return _func

  def create_loader(self):
    """Creates loader."""

    if self.use_imagenet_as_eval:
      # To evaluate on webvision eval, set this to False.
      split = ['train']
      val_ds, imagenet_info = tfds.load(
          name='imagenet2012',
          download=True,
          split='validation',
          data_dir=self.root,
          with_info=True)
      val_info = imagenet_info.splits['validation']
      tf.logging.info('WebVision: use imagenet validation')
    else:
      split = ['train', 'val']
    assert tfds.__version__.startswith('2.'),\
        'tensorflow_dataset version must be 2.x.x to use image_label_folder.'
    ds, self.info = tfds.load(
        'image_label_folder',
        split=split,
        data_dir=self.root,
        builder_kwargs=dict(dataset_name=self.version),
        with_info=True)

    train_info = self.info.splits['train']

    if len(split) == 2:
      train_ds, val_ds = ds
      val_info = self.info.splits['val']
    else:
      train_ds = ds[0]

    self.train_dataset_size = train_info.num_examples
    self.val_dataset_size = val_info.num_examples
    self.test_dataset_size = self.val_dataset_size

    train_ds = train_ds.map(
        self.wrapper_map_v2(True),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(
        self.wrapper_map_v2(False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    self.train_dataflow = train_ds
    self.val_dataflow = val_ds

    def _get_probe():
      """Create probe data tf.data.Dataset."""
      probe_ds = tf.data.TFRecordDataset(
          os.path.join(self.root, self.version, self.probe_folder,
                       'imagenet2012-probe.tfrecord-1-of-1'))
      probe_ds = probe_ds.map(
          self.wrapper_map_probe_v2,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # For single file, we need to disable auto_shard_policy for multi-workers,
      # e.g. every worker takes the same file
      options = tf.data.Options()
      options.experimental_distribute.auto_shard_policy = (
          tf.data.experimental.AutoShardPolicy.OFF)
      probe_ds = probe_ds.with_options(options)

      return probe_ds

    self.probe_dataflow = _get_probe()

    tf.logging.info(self.info)
    tf.logging.info('[{}] Create {} \n train {} probe {} val {}'.format(
        self.version, FLAGS.dataset, self.train_dataset_size,
        self.probe_size, self.val_dataset_size))
    return self
