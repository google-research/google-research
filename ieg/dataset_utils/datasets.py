# coding=utf-8
"""Loader for datasets."""

from __future__ import absolute_import
from __future__ import division

import collections
import math
import os
import sys

from absl import flags
import numpy as np
from PIL import Image
from six.moves import cPickle
import sklearn.metrics as sklearn_metrics
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from ieg import utils
from ieg.dataset_utils import utils as dataset_utils
from ieg.dataset_utils.utils import cifar_process
from ieg.dataset_utils.utils import imagenet_preprocess_image

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

  traindata, trainlabel, label_corr_train = shuffle_dataset(  # pylint: disable=unbalanced-tuple-unpacking
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
    y: 1D numpy array of labels of images
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
  trainlabel[mask] = random_labels.reshape(trainlabel[mask].shape)
  # Shuffles dataset
  traindata, trainlabel, label_corr_train = shuffle_dataset(  # pylint: disable=unbalanced-tuple-unpacking
      traindata, trainlabel, label_corr_train)
  if n_val > 0:
    valdata, vallabel = shuffle_dataset(valdata, vallabel, class_balanced=True)
  else:
    valdata, vallabel = None, None
  return (traindata, trainlabel, label_corr_train), (valdata, vallabel)


def load_batch(fpath, label_key='labels'):
  """Internal utility for parsing CIFAR data.

  Args:
    fpath: path the file to parse.
    label_key: key for label data in the retrieve dictionary.

  Returns:
    A tuple `(data, labels)`.
  """

  with tf.io.gfile.GFile(fpath, 'rb') as f:
    if sys.version_info < (3,):
      d = cPickle.load(f)
    else:
      d = cPickle.load(f, encoding='bytes')
      # decode utf8
      d_decoded = {}
      for k, v in d.items():
        d_decoded[k.decode('utf8')] = v
      d = d_decoded
  data = d['data']
  labels = d[label_key]

  data = data.reshape(data.shape[0], 3, 32, 32)
  return data, labels


def cifar100_load_data(root, label_mode='fine'):
  """Loads CIFAR100 dataset.

  Args:
    root: path that saves data file.
    label_mode: one of "fine", "coarse".

  Returns:
    Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  Raises:
    ValueError: in case of invalid `label_mode`.
  """
  if not root:
    return tf.keras.datasets.cifar100.load_data()

  fpath = os.path.join(root, 'train')

  x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

  fpath = os.path.join(root, 'test')
  x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  x_train = x_train.transpose(0, 2, 3, 1)
  x_test = x_test.transpose(0, 2, 3, 1)

  return (x_train, y_train), (x_test, y_test)


def cifar10_load_data(root):
  """Loads CIFAR10 dataset.

  Args:
    root: path that saves data file.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  if not root:
    return tf.keras.datasets.cifar10.load_data()

  num_train_samples = 50000

  x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
  y_train = np.empty((num_train_samples,), dtype='uint8')

  for i in range(1, 6):
    fpath = os.path.join(root, 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000:i * 10000, :, :, :],
     y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

  fpath = os.path.join(root, 'test_batch')
  x_test, y_test = load_batch(fpath)

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  x_train = x_train.transpose(0, 2, 3, 1)
  x_test = x_test.transpose(0, 2, 3, 1)

  x_test = x_test.astype(x_train.dtype)
  y_test = y_test.astype(y_train.dtype)

  return (x_train, y_train), (x_test, y_test)


class CIFAR(object):
  """CIFAR dataset class with different label corruption options."""

  def __init__(self, include_metadata):
    self.dataset_name = FLAGS.dataset
    self.is_cifar100 = 'cifar100' in self.dataset_name
    self.include_metadata = include_metadata
    self.image_size = 32
    if self.is_cifar100:
      self.num_classes = 100
    else:
      self.num_classes = 10
    arguments = self.dataset_name.split('_')
    self.target_ratio = float(arguments[-1])
    self.uninoise_ratio = None
    # Used for label imbalace + noise complex, e.g. cifar10_imbal_0.005_0.2
    if len(arguments) > 3:
      self.target_ratio = float(arguments[-2])
      self.uninoise_ratio = float(arguments[-1])
    assert self.target_ratio >= 0 and self.target_ratio <= 1, (
        'The template {}'
        ' of dataset is '
        'not right').format(self.dataset_name)

    self.split_probe = FLAGS.probe_dataset_hold_ratio != 0

  def create_loader(self):
    """Creates loader as tf.data.Dataset."""
    np.random.seed(FLAGS.seed)
    # load data to memory.
    if 'imbal' not in self.dataset_name:
      if self.is_cifar100:
        (x_train, y_train), (x_test,
                             y_test) = tf.keras.datasets.cifar100.load_data()
      else:
        (x_train, y_train), (x_test,
                             y_test) = tf.keras.datasets.cifar10.load_data()


      y_train = y_train.astype(np.int32)
      y_test = y_test.astype(np.int32)
      total_data_size = y_train.shape[0]

      x_train, y_train = shuffle_dataset(x_train, y_train)
      n_probe = int(
          math.floor(x_train.shape[0] * FLAGS.probe_dataset_hold_ratio))
    if 'asymmetric' in self.dataset_name:
      assert 'cifar100' not in self.dataset_name, 'Asymmetric only has CIFAR10'
      (x_train, y_train, y_gold), (x_probe, y_probe) = load_asymmetric(
          x_train,
          y_train,
          noise_ratio=self.target_ratio,
          n_val=n_probe,
          random_seed=FLAGS.seed)
    elif 'uniform' in self.dataset_name:
      (x_train, y_train, y_gold), _ = load_train_val_uniform_noise(
          x_train,
          y_train,
          n_classes=self.num_classes,
          noise_ratio=self.target_ratio,
          n_val=n_probe)
    elif 'imbal' in self.dataset_name:
      version = '100' if 'cifar100' in FLAGS.dataset else '10'
      if self.target_ratio == 1:
        data_dir = os.path.join(FLAGS.dataset_dir, 'imbalance',
                                'cifar-{}-data'.format(version))
      else:
        data_dir = os.path.join(
            FLAGS.dataset_dir, 'imbalance',
            'cifar-{}-data-im-{}'.format(version, self.target_ratio))
      train_builder = CifarImbalance(
          data_dir,
          version,
          'train',
          imb_factor=self.target_ratio,
          include_metadata=self.include_metadata,
          noise_ratio=self.uninoise_ratio)
      val_builder = CifarImbalance(
          data_dir,
          version,
          'eval',
          imb_factor=self.target_ratio,
          include_metadata=self.include_metadata,
          noise_ratio=self.uninoise_ratio)

      self.train_dataflow = train_builder.make_ds()
      self.val_dataflow = val_builder.make_ds()
      self.train_dataset_size = train_builder.dataset_size
      self.val_dataset_size = val_builder.dataset_size
      return self
    else:
      assert self.dataset_name in ['cifar10', 'cifar100']

    x_probe = None

    if not self.split_probe and x_probe is not None:
      # Usually used for supervised comparison.
      tf.logging.info('Merge train and probe')
      x_train = np.concatenate([x_train, x_probe], axis=0)
      y_train = np.concatenate([y_train, y_probe], axis=0)
      y_gold = np.concatenate([y_gold, y_probe], axis=0)

    conf_mat = sklearn_metrics.confusion_matrix(y_gold, y_train)
    conf_mat = conf_mat / np.sum(conf_mat, axis=1, keepdims=True)
    print('Corrupted confusion matirx\n {}'.format(conf_mat))
    x_test, y_test = shuffle_dataset(x_test, y_test)
    self.train_dataset_size = x_train.shape[0]
    self.val_dataset_size = x_test.shape[0]
    if self.split_probe:
      self.probe_size = x_probe.shape[0]

    input_tuple = (x_train, y_train.squeeze(), y_gold.squeeze())
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
    if self.include_metadata and is_train:  # Do not do for eval data
      ids = np.arange(data[0].shape[0], dtype=np.int32)
      data = list(data[:2]) + [ids] + list(data[2:])
      map_fn = lambda x, y, *args: (cifar_process(x, is_train), y, *args)
    else:
      map_fn = lambda x, y: (cifar_process(x, is_train), y)
      data = data[:2]
    ds = tf.data.Dataset.from_tensor_slices(tuple(data))
    ds = ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


class WebVision(object):
  """Webvision dataset class."""

  def __init__(self,
               root,
               version='webvisionmini',
               use_imagenet_as_eval=False,
               add_strong_aug=False):
    self.version = version
    self.num_classes = 50 if 'mini' in version else 1000
    self.root = root
    self.image_size = 224
    self.use_imagenet_as_test = use_imagenet_as_eval
    self.use_strong_aug = add_strong_aug

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
      if train and self.use_strong_aug:
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

  def wrapper_map_w_dataid(self, train=True):
    """tf.data.Dataset map function for train data v2."""

    def _func(data):
      image, label = data['image'], data['label']
      image_bytes = tf.image.encode_jpeg(image, name='encode_jpeg')
      image = imagenet_preprocess_image(
          image_bytes, is_training=train, image_size=self.image_size)
      return image, label, data['id'], label  # Last field is useless

    return _func

  def create_loader(self):
    """Creates loader."""

    assert tfds.__version__.startswith(
        '2.'), 'tensorflow_dataset version must be 2.x.x.'
    ds, ds_info = tfds.load(
        'image_label_folder',
        split=['train', 'val'],
        data_dir=self.root,
        builder_kwargs=dict(dataset_name=self.version),
        with_info=True)
    val_info = ds_info.splits['val']
    train_ds, val_ds = ds

    train_info = ds_info.splits['train']
    self.train_dataset_size = train_info.num_examples
    self.val_dataset_size = val_info.num_examples

    if self.use_imagenet_as_test:
      test_ds, imagenet_info = tfds.load(
          name='imagenet2012',
          download=True,
          split='validation',
          data_dir=self.root,
          with_info=True)
      test_info = imagenet_info.splits['validation']

      if self.num_classes != 1000:
        # Build imagenet 50 class validation.
        test_path = os.path.join(self.root,
                                 '{}_imagenet50class'.format(self.version))
        utils.make_dir_if_not_exists(test_path)
        test_record_files = test_path + '/test.tfrecord'
        if not tf.gfile.IsDirectory(test_path) or not tf.gfile.Glob(
            test_record_files + '*'):
          dataset_utils.write_tfrecords(
              test_record_files,
              test_ds.map(lambda d: (d['image'], d['label'])),
              test_info.num_examples,
              include_ids=False,
              filter_fn=lambda label: int(label) < self.num_classes,
              nshard=10)
        tf.logging.info('Load re-generated tfrecords from {}'.format(test_path))
        test_ds, count = dataset_utils.read_tf_records(test_record_files + '*',
                                                       True)
        assert count == test_info.num_examples // 20
        self.test_dataset_size = count

    if FLAGS.ds_include_metadata:
      # Rewrite tfreord files to include data ids.
      train_path = os.path.join(self.root, '{}_w_dataid'.format(self.version))
      utils.make_dir_if_not_exists(train_path)
      train_record_files = train_path + '/train.tfrecord'
      if not tf.gfile.IsDirectory(train_path) or not tf.gfile.Glob(
          train_record_files + '*'):
        tf.logging.info('Write tfrecords to {}'.format(train_path))
        dataset_utils.write_tfrecords(
            train_record_files,
            train_ds.map(lambda d: (d['image'], d['label'])),
            train_info.num_examples,
            nshard=50)
      tf.logging.info('Load re-generated tfrecords from {}'.format(train_path))
      train_ds, count = dataset_utils.read_tf_records(
          train_record_files + '*', True, include_ids=True)
      assert count == self.train_dataset_size

      train_ds = train_ds.map(
          self.wrapper_map_w_dataid(),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
      train_ds = train_ds.map(
          self.wrapper_map_v2(True),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_ds = val_ds.map(
        self.wrapper_map_v2(False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    self.train_dataflow = train_ds
    self.val_dataflow = val_ds
    if self.use_imagenet_as_test:
      test_ds = test_ds.map(
          self.wrapper_map_v2(False),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
      self.test_dataflow = test_ds

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

    if FLAGS.probe_dataset_hold_ratio > 0:
      self.probe_dataflow = _get_probe()
    else:
      self.probe_dataflow = None

    tf.logging.info(ds_info)
    tf.logging.info('[{}] Create {} \n train {} probe {} val {}'.format(
        self.version, FLAGS.dataset, self.train_dataset_size, self.probe_size,
        self.val_dataset_size))
    return self


def check_version(cifar_version):
  if cifar_version not in ['10', '100', '20']:
    raise ValueError('cifar version must be one of 10, 20, 100.')


def img_num(cifar_version):
  check_version(cifar_version)
  dt = {'10': 5000, '100': 500, '20': 2500}
  return dt[cifar_version]


class CifarImbalance(object):
  """Cifar imbalance data set."""

  def __init__(self,
               data_dir,
               data_version='10',
               subset='train',
               imb_factor=None,
               include_metadata=False,
               use_distortion=True,
               noise_ratio=None,
               write_to_disk=False):
    self.data_dir = data_dir
    self.data_version = data_version
    self.subset = subset
    self.imb_factor = imb_factor
    self.use_distortion = use_distortion
    self.include_metadata = include_metadata
    self.noise_ratio = noise_ratio
    self.write_to_disk = write_to_disk

  def get_filenames(self):
    if self.subset == 'train_offline':  # so avoid shuffle during make_batch
      return [os.path.join(self.data_dir, 'train' + '.tfrecords')]
    if self.subset in ['train', 'eval']:
      return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def parser(self, serialized_example, skip_preprocess=False):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of
    # the input format.
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([3 * 32 * 32])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0]), tf.float32)
    label = tf.cast(features['label'], tf.int32)
    if not skip_preprocess:
      # Custom preprocessing.
      image = self.preprocess(image)
      image = image / 128 - 1

    return image, label


  def rebuild_tfdataset_w_metadata(self, ds):
    """Rebuilds tfds to include extra meta data."""
    ds = ds.map(lambda x: self.parser(x, skip_preprocess=True))
    iterator = ds.make_one_shot_iterator().get_next()
    images, labels = [], []
    with tf.Session() as sess:
      while True:
        try:
          image, label = sess.run(iterator)
          images.append(image)
          labels.append(label)
        except tf.errors.OutOfRangeError:
          break
    images = np.stack(images)
    labels = np.stack(labels)
    if self.noise_ratio:
      # Adding noise to imbalanced dastasets.
      num_classes = len(np.unique(labels))
      (images, labels, labels_gold), (_, _) = load_train_val_uniform_noise(
          images, labels, num_classes, n_val=0, noise_ratio=self.noise_ratio)
      conf_mat = sklearn_metrics.confusion_matrix(labels_gold, labels)
      conf_mat = conf_mat / np.sum(conf_mat, axis=1, keepdims=True)
      tf.logging.info('Corrupted confusion matirx\n %s', conf_mat)
    assert images.shape[0] == self.dataset_size
    assert labels.shape[0] == self.dataset_size
    ids = np.arange(images.shape[0], dtype=np.int32)
    data_tuple = (
        images, labels, ids, labels
    )  # The last term `labels` is useless for imbalanced experiments
    ds = tf.data.Dataset.from_tensor_slices(data_tuple)
    ds = ds.map(
        lambda image, *args: (self.preprocess(image) / 128 - 1, *args),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels, counts = np.unique(labels, return_counts=True)
    stats = {i: j for i, j in zip(labels, counts)}
    tf.logging.info(
        'Rebuilt tfds with metadata for {} samples. Label counts: {}'.format(
            self.dataset_size, stats))
    return ds

  def make_ds(self):
    """Reads the images and labels from 'filenames'."""
    filenames = self.get_filenames()

    if self.include_metadata and self.subset == 'train':
      dataset = self.rebuild_tfdataset_w_metadata(
          tf.data.TFRecordDataset(filenames))

    else:
      # Repeat infinitely.
      dataset = tf.data.TFRecordDataset(filenames)
      # Parse records.
      dataset = dataset.map(
          self.parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def preprocess(self, image):
    """Preprocesses a single image in [height, width, depth] layout."""
    if self.subset == 'train' and self.use_distortion:
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
      image = tf.random_crop(image, [32, 32, 3])
      image = tf.image.random_flip_left_right(image)
    return image

  @staticmethod
  def get_img_num_per_cls(cifar_version, imb_factor=None):
    """Get a list of image numbers for each class.

    Given cifar version Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min

    Args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max, None if geting
        default cifar data number

    Returns:
      img_num_per_cls: a list of number of images per class
    """
    cls_num = int(cifar_version)
    img_max = img_num(cifar_version)
    if imb_factor is None:
      return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
      num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
      img_num_per_cls.append(int(num))
    return img_num_per_cls

  @property
  def dataset_size(self):
    return int(
        CifarImbalance.num_examples_per_epoch(self.subset, self.imb_factor,
                                              self.data_version))

  @staticmethod
  def num_examples_per_epoch(subset='train',
                             imb_factor=None,
                             cifar_version='10'):
    """Returns number of examples."""
    if subset == 'train':
      if imb_factor is None:
        return 50000
      else:
        return sum(
            CifarImbalance.get_img_num_per_cls(cifar_version, imb_factor))
    elif subset == 'eval':
      return 10000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)


