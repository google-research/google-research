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

"""Base data loader classes."""

import functools

from absl import logging
from flax import jax_utils
import jax
import ml_collections
import numpy as onp
import tensorflow as tf
import tensorflow_datasets as tfds

from gift.data import dataset_utils
from gift.data import image_augment_utils


class ImageDataset(object):
  """Base dataset class for image datasets."""

  _THREAD_POOL_SIZE = 48

  def __init__(self,
               batch_size,
               eval_batch_size,
               num_shards,
               pseudo_label_generator=None,
               resolution=None,
               resize_mode=None,
               if_cache=True,
               data_augmentations=None,
               teacher_data_augmentations=None,
               dtype_str='float32',
               shuffle_seed=1):
    """Returns generators for train, validation and test sets of a tfds dataset.

    Args:
      batch_size: int; Determines the train batch size.
      eval_batch_size: int; Determines the evaluation batch size.
      num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
      pseudo_label_generator: function; A function that generate pseudo labels.
        If None, ground truth labels will be used.
      resolution: int; Resolution of images in the dataset.
      resize_mode: str; If needed how to resize the images (down/up sampling:
        `resize`, cropping: `center_crop` or `random_crop`)/
      if_cache: bool; Whether we should have caching in the data pipeline.
      data_augmentations: list(str); types of data augmentations to apply on the
        images (currently 'rand' and 'default').
      teacher_data_augmentations: list(str); This should be set if inputs to the
        teacher model (used in pseudo_label_generator), needs different type of
        augmentation (if None teacher would get the same input as the student).
        If you set this, you also need to set `teacher_input_key` to
        `teacher_inputs` in the config file of the experiment.
      dtype_str: str; Data type of the image (e.g. 'float32').
      shuffle_seed: int; Seed for shuffling the training data.
    """
    self.dtype = dataset_utils.DATA_TYPE[dtype_str]
    self.batch_size = batch_size
    self.eval_batch_size = eval_batch_size
    self.num_shards = num_shards
    self.shuffle_seed = shuffle_seed
    self.pseudo_label_generator = pseudo_label_generator
    self.data_augmentations = data_augmentations
    self.teacher_data_augmentations = teacher_data_augmentations

    self.if_cache = if_cache

    self.resolution = resolution
    self.resize_mode = resize_mode

    self.set_builder()
    self.set_static_dataset_configs()
    # Load data splits.
    self.set_splits()
    self.load_splits()
    self.set_metadata()

  @property
  def name(self):
    raise NotImplementedError

  def set_builder(self):
    self.builder = self.get_builder(self.name)

  def set_static_dataset_configs(self):
    """Sets dataset and data processing parameters."""
    self._channels = 3
    self._crop_padding = 32
    self._mean_rgb = [0.485, 0.456, 0.406]
    self._stddev_rgb = [0.229, 0.224, 0.225]
    self._splits_dict = {
        'train': 'train',
        'test': 'test',
        'validation': 'validation'
    }
    self.eval_augmentations = None

  def set_splits(self):
    """Define splits of the dataset.

    For single environment datasets we have tree splits and self.splits is
    a ConfigDict that maps each split to a config dict that contains information
    about that split:
    self.splits.train, self.splits.test, self.splits.validation
    """
    test_split_config = ml_collections.ConfigDict(
        dict(
            name=self._splits_dict['test'],
            batch_size=self.eval_batch_size,
            train=False))
    valid_split_config = ml_collections.ConfigDict(
        dict(
            name=self._splits_dict['validation'],
            batch_size=self.eval_batch_size,
            train=False))
    train_split_config = ml_collections.ConfigDict(
        dict(
            name=self._splits_dict['train'],
            batch_size=self.batch_size,
            train=True))

    self.splits = ml_collections.ConfigDict(
        dict(
            test=test_split_config,
            validation=valid_split_config,
            train=train_split_config))

  def load_splits(self):
    """Load dataset splits using tfds loader."""
    self.data_iters = ml_collections.ConfigDict()
    for key, split in self.splits.items():
      logging.info('Loading %s  split of the %s dataset.', split.name,
                   self.name)
      ds, num_examples = self.load_split_from_tfds(
          name=self.name,
          batch_size=split['batch_size'],
          train=split['train'],
          split=split.name,
          shuffle_seed=self.shuffle_seed)

      self.splits[key].num_examples = num_examples

      self.data_iters[key] = self.create_data_iter(ds, split['batch_size'])

  def create_data_iter(self, ds, batch_size):
    """Create an iterator from a tf dataset.

    Args:
      ds: tfds dataset; Dataset which we want to build an iterator for.
      batch_size: int; Batch size for the given dataset split.

    Returns:
      Data iter for the given dataset.
    """
    data_iter = iter(ds)

    def prepare_tf_data(xs):
      """Reshapes input batch."""

      def _prepare(x):
        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((self.num_shards, -1) + x.shape[1:])

      return jax.tree_map(_prepare, xs)

    def to_numpy(xs):
      """Convert a input batch from tf Tensors to numpy arrays."""

      return jax.tree_map(
          lambda x: x._numpy(),  # pylint: disable=protected-access
          xs)

    def maybe_pad_batch(batch):
      """Zero pad the batch on the right to the batch_size.

      Args:
        batch: dict; A dictionary mapping keys to arrays. We assume that inputs
          is one of the keys.

      Returns:
        A dictionary mapping the same keys to the padded batches. Additionally
        we
        add a key representing weights, to indicate how the batch was padded.
      """
      batch_pad = batch_size - batch['inputs'].shape[0]
      unpadded_mask_shape = batch['inputs'].shape[0]

      # Most batches will not need padding so we quickly return to avoid
      # slowdown.
      if batch_pad == 0:
        if 'weights' not in batch:
          batch['weights'] = onp.ones(unpadded_mask_shape, dtype=onp.float32)
        return batch

      def zero_pad(array):
        pad_with = [(0, batch_pad)] + [(0, 0)] * (array.ndim - 1)
        return onp.pad(array, pad_with, mode='constant')

      padded_batch = jax.tree_map(zero_pad, batch)
      padded_batch_mask = zero_pad(
          onp.ones(unpadded_mask_shape, dtype=onp.float32))
      if 'weights' in padded_batch:
        padded_batch['weights'] *= padded_batch_mask
      else:
        padded_batch['weights'] = padded_batch_mask

      return padded_batch

    it = map(to_numpy, data_iter)
    it = map(maybe_pad_batch, it)
    it = map(prepare_tf_data, it)
    it = jax_utils.prefetch_to_device(it, 2)
    return it

  def maybe_resize(self, image):
    """Resize the image if needed.

    In some cases images should be resized. If self.resolution is set or if
    images of the dataset don't have the same size.

    E.g.

    Args:
      image: tf tensor; Input image of type float.

    Returns:
      maybe resized image.
    """
    if self.resolution and self.resize_mode:
      # If resolution is set image size <-- resolution

      if self.resize_mode == 'resize':
        image = tf.image.resize_with_pad([image],
                                         self.resolution,
                                         self.resolution,
                                         method='bicubic',
                                         antialias=False)[0]

      elif self.resize_mode == 'center_crop':
        image = tf.image.resize_with_crop_or_pad(image, self.resolution,
                                                 self.resolution)
        image = tf.reshape(image, [self.resolution, self.resolution, 3])
      elif self.resize_mode == 'random_crop':
        image = image_augment_utils.random_crop(image, self.resolution,
                                                self._crop_padding)
        image = tf.reshape(image,
                           [self.resolution, self.resolution, self._channels])

    return image

  def process_batch(self, batches):
    """Post process the given batch.

    This is applied in the final stage of the input pipeline. In a
    self-supervised dataset, the labels are pseudo labels generated by
    self.pseudo_label_generator.

    Args:
      batches: dict; Batch of examples with  'inputs' and 'label' keys.

    Yields:
      A processed batch.
    """
    for batch in batches:
      if self.pseudo_label_generator:
        batch = self.pseudo_label_generator(batch=batch)

      yield batch

  def reset_pseudo_label_generator(self, pseudo_label_generator):
    """Restart training data iter to replace labels with pseudo labels.

    Args:
      pseudo_label_generator: fn: batch --> batch; The function updates the
        labels and weights of the examples in the batch.
    """
    self.pseudo_label_generator = pseudo_label_generator
    self.data_iters.train = self.process_batch(self.data_iters.train)

  def preprocess_example(self, example, env_name=''):
    """Preprocesses the given image and is called before cache/repeat.

    (If we want transformations to be true random, we shouldn't apply them
    here.)

    Args:
      example: dict; Example that has an 'image' and a 'label'.
      env_name: str; Unused variable (Used in multi env setup).

    Returns:
      A preprocessed image `Tensor`.
    """
    del env_name

    # We cast the image to have the type float32.
    image = example['image']

    if image.dtype in ['uint8', 'int32', 'int64']:
      image = tf.cast(image, 'float32') / 255.

    return {'inputs': image, 'label': example['label']}

  def augment_image(self, image, data_augmentations):
    """Process the given image and apply augmentations.

    Args:
      image: tf tensor; Image.
      data_augmentations: list(str); List if data augmentations to apply.

    Returns:
      A preprocessed image `Tensor`.
    """
    if data_augmentations:
      image = image_augment_utils.transform_image(
          image,
          data_augmentations,
          resolution=self.resolution,
          crop_padding=self._crop_padding,
          channels=self._channels)

    return image

  def process_train_example(self, example):
    """Preprocesssing of examples that includes data augmentations.

    Args:
      example: dict; Example input with keys such as inputs and label.

    Returns:
      processed example (dict).
    """
    image = example['inputs']

    example['inputs'] = self.augment_image(image, self.data_augmentations)
    example['inputs'] = self.maybe_resize(example['inputs'])
    example['inputs'] = self.convert_and_normalize(example['inputs'])

    if self.teacher_data_augmentations:
      example['teacher_inputs'] = self.augment_image(
          image, self.teacher_data_augmentations)
      example['teacher_inputs'] = self.maybe_resize(example['teacher_inputs'])
      example['teacher_inputs'] = self.convert_and_normalize(
          example['teacher_inputs'])
    return example

  def convert_and_normalize(self, image):
    """Convert and scale based on final type.

    If target type is int the output image range is [0,255]
    If target type is float the output image the range can depend on
    self._mean_rgb and self._stddev_rgb.

    Args:
      image: float32 tensor; Image of type float32 and in range of [0,1].

    Returns:
      image tensor of type self.dtype.tf_dtype.
    """
    image = tf.cast(image, dtype=self.dtype.tf_dtype)

    if image.dtype in ['float16', 'float32', 'float64']:
      image = image_augment_utils.normalize_image(
          image,
          mean_rgb=self._mean_rgb,
          stddev_rgb=self._stddev_rgb,
          channels=self._channels)

    if image.dtype in ['uint8', 'int32', 'int64']:
      image = image * 255

    return image

  def process_eval_example(self, example):
    """Preprocesses the given image for evaluation.

    Args:
      example: dict; Example from the dataset.

    Returns:
      A preprocessed example.
    """
    image = example['inputs']
    image = self.maybe_resize(
        self.augment_image(image, self.eval_augmentations))
    example['inputs'] = self.convert_and_normalize(image)

    return example

  def get_builder(self, unused_name):
    return tfds.builder(self.name)

  def load_split_from_tfds(self,
                           name,
                           batch_size,
                           train,
                           split=None,
                           shuffle_seed=1):
    """Loads a split from the dataset using TensorFlow Datasets.

    Args:
      name: str; Name of the environment passed to `tfds.load`.
      batch_size: int; The batch size returned by the data pipeline.
      train: bool; Whether to load the train or evaluation split.
      split: str; Name of the dataset split passed to tfds, if None, the value
        is set with respect to the `train` argument.
      shuffle_seed: int; Seed for shuffling the training data.

    Returns:
      A `tf.data.Dataset`.
    """
    if split is None:
      split = 'train' if train else 'test'

    builder = self.get_builder(name)
    # Each host is responsible for a fixed subset of data
    base_split_name, host_start, host_end = dataset_utils.get_data_range(
        builder, split, jax.host_id(), jax.host_count())
    data_range = tfds.core.ReadInstruction(
        base_split_name, unit='abs', from_=host_start, to=host_end)

    ds, ds_info = self.get_tfds_ds_and_info(name, data_range)

    # Applying preprocessing before `ds.cache()` to re-use it
    decode_example = functools.partial(self.preprocess_example, env_name=name)
    ds = ds.map(
        decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if self.if_cache:
      ds = ds.cache()
    if train:
      ds = ds.repeat()
      ds = ds.shuffle(16 * batch_size, seed=shuffle_seed)
      ds = ds.map(
          self.process_train_example,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.batch(batch_size, drop_remainder=False)
    else:
      ds = ds.map(
          self.process_eval_example,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.batch(batch_size, drop_remainder=False)
      ds = ds.repeat()

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds, ds_info.splits[split].num_examples

  def get_tfds_ds_and_info(self, name, data_range):
    """Load tfds data.

    Args:
      name: str; Name of the tfds dataset (e.g, `mnist`).
      data_range: Dataset split (e.g., test[10%:100%]).

    Returns:
      Dataset and its info.
    """
    ds, ds_info = tfds.load(name, split=data_range, with_info=True)
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 48
    ds = ds.with_options(options)

    return ds, ds_info

  def set_metadata(self):
    """Set meta information about the dataset."""

    num_classes = self.builder.info.features['label'].num_classes
    input_shape = (-1,) + self.builder.info.features['image'].shape
    if self.resolution:
      input_shape = (-1,) + (self.resolution,) * (
          len(self.builder.info.features['image'].shape) - 1) + (
              self.builder.info.features['image'].shape[-1],)

    self.meta_data = {
        'num_classes':
            num_classes,
        'input_shape':
            input_shape,
        'input_dtype':
            self.dtype.jax_dtype,
        'num_train_examples':
            self.splits.train.num_examples * jax.host_count(),
        'num_eval_examples':
            self.splits.validation.num_examples * jax.host_count(),
    }


class MutliEnvironmentImageDataset(ImageDataset):
  """Base class for multi environment image datasets."""

  def __init__(self,
               batch_size,
               eval_batch_size,
               num_shards,
               train_environments=None,
               eval_environments=None,
               pseudo_label_generator=None,
               resolution=None,
               resize_mode=None,
               if_cache=True,
               data_augmentations=None,
               teacher_data_augmentations=None,
               dtype_str='float32',
               shuffle_seed=0):
    """Returns generators for train, validation and test sets of a tfds dataset.

    Args:
      batch_size: int; Determines the train batch size.
      eval_batch_size: int; Determines the evaluation batch size.
      num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
      train_environments: list(str); List of training environment names.
      eval_environments: list(str); List of evaluation environment names.
      pseudo_label_generator: function; A function that given a batch dict
        generate pseudo labels. If None (default), actual labels will be used.
      resolution: int; Image size (should be set for datasets that have images
        with different sizes.
      resize_mode: bool; Whether to resize the image (in contrast to cropping
        it) so that it has the expected shape.
      if_cache: bool; Whether we should have caching in the data pipeline.
      data_augmentations: list(str); types of data augmentations to apply on the
        images (currently 'rand' and 'default').
      teacher_data_augmentations: list(str); This should be set if inputs to the
        teacher model (used in pseudo_label_generator), needs different type of
        augmentation (if None teacher would get the same input as the student).
        If you set this, you also need to set `teacher_input_key` to
        `teacher_inputs` in the config file of the experiment.
      dtype_str: str; Data type of the image (e.g. 'float32').
      shuffle_seed: int; Seed for shuffling the training data.
    """
    self.set_environments(train_environments, eval_environments)
    super().__init__(
        batch_size,
        eval_batch_size,
        num_shards,
        pseudo_label_generator=pseudo_label_generator,
        resolution=resolution,
        resize_mode=resize_mode,
        dtype_str=dtype_str,
        if_cache=if_cache,
        data_augmentations=data_augmentations,
        teacher_data_augmentations=teacher_data_augmentations,
        shuffle_seed=shuffle_seed)

  def set_environments(self, train_environments, eval_environments):
    """Sets the list of sub datasets' names."""
    self.train_environments = train_environments or self.get_all_environments()
    self.eval_environments = eval_environments or self.get_all_environments()

  def get_env_split_name(self, split, env):
    return self._splits_dict[split][env]

  @classmethod
  def get_all_environments(cls):
    return cls._ALL_ENVIRONMENTS

  @classmethod
  def env2id(cls, env_name):
    return cls.get_all_environments().index(env_name)

  @classmethod
  def id2env(cls, env_id):
    return cls.get_all_environments()[int(env_id)]

  def get_full_env_name(self, env_name):
    """Full environment name (including dataset name)."""
    return '/'.join([self.name, env_name])

  def get_tfds_env_name(self, env_name):
    """Environment name used to load tfds data."""
    return '/'.join([self.name, env_name])

  def reset_pseudo_label_generator(self, pseudo_label_generator):
    self.pseudo_label_generator = pseudo_label_generator

    for env in self.data_iters.train:
      self.data_iters.train[env] = self.process_batch(
          self.data_iters.train[env])

  def set_splits(self):
    """Define splits of the dataset.

    For multi environment datasets, we have tree main splits:
    test, train and valid.
    Each of these splits is a dict mapping environment id to the information
    about that particular split of that dataset:

    self.splits
    |__Test
    |  |__Eval env 1
    |  |__Eval env 2
    |
    |__ Train
    |  |__ Train env 1
    |  |__ Train env 2
    |
    |__ Valied
       |__Eval env 1
       |__Eval env 2
    """

    train_split_configs = ml_collections.ConfigDict()
    valid_split_configs = ml_collections.ConfigDict()
    test_split_configs = ml_collections.ConfigDict()

    for env_name in self.train_environments:
      env_id = self.env2id(env_name)
      train_split_configs[str(env_id)] = ml_collections.ConfigDict(
          dict(
              name=self.get_env_split_name('train', env_name),
              batch_size=self.batch_size,
              train=True))

    for env_name in self.eval_environments:
      env_id = self.env2id(env_name)
      valid_split_configs[str(env_id)] = ml_collections.ConfigDict(
          dict(
              name=self.get_env_split_name('validation', env_name),
              batch_size=self.eval_batch_size,
              train=False))
      test_split_configs[str(env_id)] = ml_collections.ConfigDict(
          dict(
              name=self.get_env_split_name('test', env_name),
              batch_size=self.eval_batch_size,
              train=False))

    self.splits = ml_collections.ConfigDict(
        dict(
            test=test_split_configs,
            validation=valid_split_configs,
            train=train_split_configs))

  def load_splits(self):
    """Load dataset splits using tfds loader."""
    self.data_iters = ml_collections.ConfigDict()
    for split_name, split in self.splits.items():
      logging.info('Loading %s  split of the %s dataset.', split_name,
                   self.name)
      self.data_iters[split_name] = ml_collections.ConfigDict()
      for env_key, env_cnfg in split.items():
        env_name = self.id2env(int(env_key))
        logging.info('Loading environment %s of the %s dataset.', env_name,
                     self.name)
        ds, num_examples = self.load_split_from_tfds(
            name=env_name,
            batch_size=env_cnfg.batch_size,
            train=env_cnfg.train,
            split=env_cnfg.name,
            shuffle_seed=self.shuffle_seed)

        self.splits[split_name][env_key].num_examples = num_examples
        self.data_iters[split_name][env_key] = self.create_data_iter(
            ds, env_cnfg.batch_size)

  def preprocess_example(self, example, env_name):
    """Preprocesses the given image.

    Args:
      example: dict: Example that has an 'image' and a 'label'.
      env_name: str; Environment name.

    Returns:
      A preprocessed image `Tensor`.
    """

    example = super().preprocess_example(example, env_name)

    return {
        'inputs': example['inputs'],
        'label': example['label'],
        'env_name': self.env2id(env_name)
    }

  def get_tfds_ds_and_info(self, name, data_range):
    return super().get_tfds_ds_and_info(
        self.get_tfds_env_name(name), data_range)

  def get_num_classes(self):
    return self.builder.info.features['label'].num_classes

  def get_input_shape(self):
    input_shape = (-1,) + self.builder.info.features['image'].shape
    if self.resolution:
      input_shape = (-1,) + (self.resolution,) * (
          len(self.builder.info.features['image'].shape) - 1) + (
              self.builder.info.features['image'].shape[-1],)
    return input_shape

  def set_metadata(self):
    """Set meta information about the dataset."""

    num_classes = self.get_num_classes()
    input_shape = self.get_input_shape()

    self.meta_data = {
        'num_classes':
            num_classes,
        'input_shape':
            input_shape,
        'input_dtype':
            self.dtype.jax_dtype,
        'num_train_examples_per_env': {
            env: self.splits.train[env].num_examples
            for env in self.splits.train
        },
        'num_eval_examples_per_env': {
            env: self.splits.validation[env].num_examples
            for env in self.splits.validation
        },
        # We don't sum them, because in this case we are processing them in
        # parallel batches, and we are taking the number of examples in the
        # first  one because we are assuming the number of training examples
        # in all environment is the same, otherwise the other attributes:
        # num_train_examples_per_env and num_eval_examples_per_env should be
        # used.
        'num_train_examples':
            self.splits.train[str(self.env2id(
                self.train_environments[0]))].num_examples * jax.host_count(),
        'num_eval_examples':
            self.splits.validation[str(self.env2id(
                self.eval_environments[0]))].num_examples * jax.host_count(),
    }
