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

# Lint as: python3
"""Data providers."""

import abc
import collections
import functools

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from supcon import enums
from supcon import preprocessing
from supcon import utils

InputData = collections.namedtuple('InputData', 'images labels')


class CommonInput(metaclass=abc.ABCMeta):
  """An abstract class of common code for input functions.

  Subclasses must override `dataset_parser` and `make_source_dataset`, which
  combine to specialize the subclass to produce a dataset of InputData tuples,
  each consisting of a single unaugmented image and its label,
  from whatever format the raw data is in, possibly sharding the dataset along
  the way. The shared code implemented in this abstract class is responsible for
  transforming this dataset of InputData per single image into a dataset of
  InputData tuples consisting of a batch of multi-view augmented images with a
  corresponding batch of labels.

  Example usage:
    # This could just be a dict containing a 'batch_size' key, or the actual
    # params provided by a TPUEstimator.
    params = tpu_estimator_params
    input_data_tensors = MyCommonInputSubclass(foo=bar, ...).input_fn(params)

  """

  def __init__(self,
               mode,
               preprocessor,
               num_classes,
               num_parallel_calls=64,
               shuffle_buffer=1024,
               shard_per_host=True,
               cache=False,
               max_samples=None,
               label_noise_prob=0.):
    """Abstract common input class.

    Args:
      mode: An enums.ModelMode indicating whether this is train, eval, or
        inference mode.
      preprocessor: Preprocessor object. Must implement
        preprocessng.Preprocessor interface.
      num_classes: The number of classes in the label set.
      num_parallel_calls: concurrency level to use when reading data from disk.
      shuffle_buffer: How large the shuffle buffer should be.
      shard_per_host: If True, during training the dataset is sharded per TPU
        host, rather than each host independently iterating over the full
        dataset. This guarantees that the same sample won't appear in the same
        global batch on different TPU cores, and also saves memory when caching
        the dataset. Does nothing if not running on TPU.
      cache: if True, fill the dataset by repeating from its cache.
      max_samples: If non-None, takes the first `max_samples` samples from the
        (possibly shuffled) dataset.
      label_noise_prob: The probability with which a class label should be
        replaced with a random class label.
    """
    self.mode = mode
    self.preprocessor = preprocessor
    self.num_classes = num_classes
    if num_classes < 1:
      raise ValueError(f'num_classes must be positive. Was {num_classes}.')
    self.num_parallel_calls = num_parallel_calls
    self.shard_per_host = shard_per_host
    self.cache = cache
    self.shuffle_buffer = shuffle_buffer
    self.max_samples = max_samples
    self.label_noise_prob = label_noise_prob
    if label_noise_prob < 0. or label_noise_prob > 1.:
      raise ValueError('Label noise probability must be between 0 and 1. '
                       f'Found {label_noise_prob}.')

  def _set_static_batch_dim(self, batch_size, input_data):
    """Statically set the batch_size dimension."""
    images = input_data.images
    labels = input_data.labels

    images.set_shape(images.get_shape().merge_with(
        tf.TensorShape([batch_size, None, None, None])))
    labels.set_shape(labels.get_shape().merge_with(
        tf.TensorShape([batch_size])))

    return InputData(images, labels)

  def _preprocess_image(self, input_data):
    """Preprocesses data to a multiviewed image.

    Args:
      input_data: An InputData instance. If `self.decode_images` is True,
        `input_data.image` should be an encoded string Tensor. Otherwise it
        should be a decoded numeric Tensor.

    Returns:
      An InputData instance with the image decoded and preprocessed and the
      label the same as the input.
    """
    image = self.preprocessor.preprocess(input_data.images)
    return InputData(images=image, labels=input_data.labels)

  def _label_noise_fn(self, input_data):
    """Adds label noise.

    With probability `self.label_noise_prob` assign a random label in
    `self.num_classes` rather than the true label.

    Args:
      input_data: An InputData instance.

    Returns:
      An InputData instance with `images` unchanged, but with a new label, which
      may evaluate to the same value as the old one, but the Tensors will not be
      the same.
    """
    original_label = input_data.labels
    selector = tf.random.uniform((), minval=0., maxval=1.)
    random_label = tf.cast(
        tf.math.floor(
            tf.random.uniform((), minval=0., maxval=1.) * self.num_classes),
        original_label.dtype)
    new_label = tf.cond(selector < self.label_noise_prob, lambda: random_label,
                        lambda: original_label)
    return InputData(images=input_data.images, labels=new_label)

  @abc.abstractmethod
  def dataset_parser(self, value):
    """Parses an image and its label from the output of `make_source_dataset`.

    Args:
      value: One value from the dataset produced by `make_source_dataset`.

    Returns:
      An InputData consisting of the image and label parsed from `value`. If
      self.decode_images is True then image should be an encoded string Tensor.
      If False, it should be a decoded numeric Tensor.
    """
    pass

  @abc.abstractmethod
  def make_source_dataset(self, current_host_index, num_hosts):
    """Creates a tf.data.Dataset for this dataset.

    Deterministically produces a tf.data.Dataset, each element of which
    corresponds to a single image. If applicable, it should be able to shard the
    dataset such that each independent training worker/host sees a different
    subset of the data, depending on `shard_per_host`.

    Args:
      current_host_index: The index of the host we are currently producing data
        for.
      num_hosts: The total number of hosts.

    Returns:
      A tf.data.Dataset with element types compatible with that `dataset_parser`
      expects, where each element corresponds to a single image.
    """
    pass

  def input_fn(self, params):
    """Input function which provides a single batch of data.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`. If that is
        not available, it can just be a `dict` containing `batch_size` as a key
        with the desired batch size to be produced by this Dataset as the value.

    Returns:
      A `tf.data.Dataset` object or an InputData tuple containing batched
      Tensors produced by the dataset, dependeing on the value of `as_dataset`.
    """
    with tf.variable_scope('data_provider'):
      if self.mode == enums.ModelMode.INFERENCE:
        images = tf.placeholder(tf.float32, [
            None, self.preprocessor.preprocessing_options.image_size,
            self.preprocessor.preprocessing_options.image_size, 3
        ])
        return tf.estimator.export.TensorServingInputReceiver(
            features=images, receiver_tensors=images)

      # Retrieves the batch size for the current shard. The # of shards is
      # computed according to the input pipeline deployment. See
      # tf.contrib.tpu.RunConfig for details.
      batch_size = params['batch_size']

      if 'context' in params:
        current_host = params['context'].current_input_fn_deployment()[1]
        num_hosts = params['context'].num_hosts
        num_cores = params['context'].num_replicas
      else:
        current_host = 0
        num_hosts = 1
        num_cores = 1

      dataset = self.make_source_dataset(current_host, num_hosts)

      if (self.mode == enums.ModelMode.TRAIN and self.max_samples and
          self.max_samples > 0):
        dataset = dataset.take(self.max_samples)

      dataset = dataset.map(self.dataset_parser, num_parallel_calls=num_cores)
      if self.label_noise_prob > 0. and self.mode == enums.ModelMode.TRAIN:
        dataset = dataset.map(
            self._label_noise_fn, num_parallel_calls=num_cores)

      if self.cache:
        dataset = dataset.cache()
      if self.mode == enums.ModelMode.TRAIN:
        dataset = dataset.shuffle(self.shuffle_buffer).repeat()

      # Use the fused map-and-batch operation.
      #
      # For XLA, we must used fixed shapes. Because we repeat the source
      # training dataset indefinitely, we can use `drop_remainder=True` to get
      # fixed-size batches without dropping any training examples.
      #
      # When evaluating, `drop_remainder=True` prevents accidentally evaluating
      # the same image twice by dropping the final batch if it is less than a
      # full batch size. As long as this validation is done with consistent
      # batch size, exactly the same images will be used.
      dataset = dataset.apply(
          tf.data.experimental.map_and_batch(
              self._preprocess_image,
              batch_size=batch_size,
              num_parallel_batches=num_cores,
              drop_remainder=True))

      # Assign static batch size dimension
      dataset = dataset.map(
          functools.partial(self._set_static_batch_dim, batch_size))

      # Prefetch overlaps in-feed with training
      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

      return dataset


class TfdsInput(CommonInput):
  """Generates an input_fn that works with TFDS datasets."""

  def __init__(self, dataset_name, split, *args, data_dir=None, **kwargs):
    """Creates an input_fn for a TFDS dataset.

    Args:
      dataset_name: The name of the TFDS dataset, passed as the `name` argument
        of tfds.load().
      split: The split name passed as the `split` argument to tfds.load().
      *args: Arguments passed on to CommonInput.
      data_dir: The directory passed as the `data_dir` argument to tfds.load().
      **kwargs: Keyword arguments passed on to CommonInput.
    """
    super(TfdsInput, self).__init__(*args, **kwargs)
    self.dataset_name = dataset_name
    self.split = split
    self.data_dir = data_dir

  def dataset_parser(self, value):
    """Parses a TFDS datum tuple into an InputData instance.

    Args:
      value: A dictionary with keys 'image' (or 'video') and 'label'.

    Returns:
      An InputData consisting of an image and label. Note: If `value` contains a
      'video' key, then the returned images entry will be set to it.
    """
    if 'image' in value:
      images = value['image']
    elif 'video' in value:
      images = value['video']
    else:
      raise ValueError('No "image" or "video" key found in TFDS datum')

    return InputData(images=images, labels=value['label'])

  def make_source_dataset(self, current_host_index, num_hosts):
    """Makes a dataset of dictionaries of images and labels.

    Args:
      current_host_index: current host index.
      num_hosts: total number of hosts.

    Returns:
      A `tf.data.Dataset` object where each dataset element is a dictionary. For
      image classification datasets, the dictionary will contain an 'image' key
      with a decoded uint8 image (of shape [height, width, channels]) and a
      'label' key with an int64 label. For video classification datasets, the
      dictionary will contain a 'video' key with a decoded uint8 video (of shape
      [frames, height, width, channels]) and a 'label' key with an int64 label.
    """
    split = self.split
    if self.mode == enums.ModelMode.TRAIN and self.shard_per_host:
      split = tfds.even_splits(split, n=num_hosts)[current_host_index]
    # Don't shuffle until after sharding, since otherwise you risk dropping
    # samples because the sharding is performed on different shufflings of the
    # data for each core.
    return tfds.load(
        name=self.dataset_name,
        split=split,
        data_dir=self.data_dir,
        shuffle_files=False)


def imagenet(mode, params):
  """An input_fn for ImageNet (ILSVRC 2012) data."""
  model_mode = utils.estimator_mode_to_model_mode(mode)
  hparams = params['hparams']
  is_training = model_mode == enums.ModelMode.TRAIN
  preprocessor = preprocessing.ImageToMultiViewedImagePreprocessor(
      is_training=is_training,
      preprocessing_options=hparams.input_data.preprocessing,
      dataset_options=preprocessing.DatasetOptions(decode_input=False),
      bfloat16_supported=params['use_tpu'])
  imagenet_input = TfdsInput(
      dataset_name='imagenet2012:5.*.*',
      split='train' if is_training else 'validation',
      mode=model_mode,
      preprocessor=preprocessor,
      shuffle_buffer=1024,
      shard_per_host=hparams.input_data.shard_per_host,
      cache=is_training,
      num_parallel_calls=64,
      max_samples=hparams.input_data.max_samples,
      label_noise_prob=hparams.input_data.label_noise_prob,
      num_classes=get_num_classes(hparams),
      data_dir=params['data_dir'],
  )

  return imagenet_input.input_fn(params)


def cifar10(mode, params):
  """CIFAR10 dataset creator."""
  # Images are naturally 32x32.
  model_mode = utils.estimator_mode_to_model_mode(mode)
  hparams = params['hparams']
  is_training = model_mode == enums.ModelMode.TRAIN
  preprocessor = preprocessing.ImageToMultiViewedImagePreprocessor(
      is_training=is_training,
      preprocessing_options=hparams.input_data.preprocessing,
      dataset_options=preprocessing.DatasetOptions(
          decode_input=False,
          image_mean_std=(np.array([[[-0.0172, -0.0356, -0.107]]]),
                          np.array([[[0.4046, 0.3988, 0.402]]]))),
      bfloat16_supported=params['use_tpu'])
  cifar_input = TfdsInput(
      dataset_name='cifar10:3.*.*',
      split='train' if is_training else 'test',
      mode=model_mode,
      preprocessor=preprocessor,
      shard_per_host=hparams.input_data.shard_per_host,
      cache=is_training,
      shuffle_buffer=50000,
      num_parallel_calls=64,
      max_samples=hparams.input_data.max_samples,
      label_noise_prob=hparams.input_data.label_noise_prob,
      num_classes=get_num_classes(hparams),
      data_dir=params['data_dir'],
  )

  return cifar_input.input_fn(params)


def get_num_train_images(hparams):
  """Returns the number of training images according to the dataset."""
  num_images_map = {
      'imagenet': 1281167,
      'cifar10': 50000,
  }
  if hparams.input_data.input_fn not in num_images_map:
    raise ValueError(
        f'Unknown dataset size for input_fn {hparams.input_data.input_fn}')

  num_images = num_images_map[hparams.input_data.input_fn]

  if hparams.input_data.max_samples > 0:
    return min(num_images, hparams.input_data.max_samples)
  return num_images


def get_num_eval_images(hparams):
  """Returns the number of eval images according to the dataset."""
  num_images_map = {
      'imagenet': 50000,
      'cifar10': 10000,
  }
  if hparams.input_data.input_fn not in num_images_map:
    raise ValueError(
        f'Unknown dataset size for input_fn {hparams.input_data.input_fn}')

  return num_images_map[hparams.input_data.input_fn]


def get_num_classes(hparams):
  """The cardinality of the label set."""
  num_classes_map = {
      'imagenet': 1000,
      'cifar10': 10,
  }
  if hparams.input_data.input_fn not in num_classes_map:
    raise ValueError(
        f'Unknown number of classes for input_fn {hparams.input_data.input_fn}')
  return num_classes_map[hparams.input_data.input_fn]
