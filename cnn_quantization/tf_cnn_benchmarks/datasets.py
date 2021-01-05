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

"""Benchmark dataset utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os

import numpy as np
import six
from six.moves import cPickle
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
from cnn_quantization.tf_cnn_benchmarks import preprocessing
from tensorflow.python.platform import gfile

IMAGENET_NUM_TRAIN_IMAGES = 1281167
IMAGENET_NUM_VAL_IMAGES = 50000

COCO_NUM_TRAIN_IMAGES = 118287
COCO_NUM_VAL_IMAGES = 4952


class Dataset(object):
  """Abstract class for cnn benchmarks dataset."""

  def __init__(self,
               name,
               data_dir=None,
               queue_runner_required=False,
               num_classes=None):
    self.name = name
    self.data_dir = data_dir
    self._queue_runner_required = queue_runner_required
    self._num_classes = num_classes

  def tf_record_pattern(self, subset):
    return os.path.join(self.data_dir, '%s-*-of-*' % subset)

  def reader(self):
    return tf.TFRecordReader()

  @property
  def num_classes(self):
    return self._num_classes

  @num_classes.setter
  def num_classes(self, val):
    self._num_classes = val

  @abc.abstractmethod
  def num_examples_per_epoch(self, subset):
    pass

  def __str__(self):
    return self.name

  def get_input_preprocessor(self, input_preprocessor='default'):
    assert not self.use_synthetic_gpu_inputs()
    return _SUPPORTED_INPUT_PREPROCESSORS[self.name][input_preprocessor]

  def queue_runner_required(self):
    return self._queue_runner_required

  def use_synthetic_gpu_inputs(self):
    return not self.data_dir


class LibrispeechDataset(Dataset):
  """Configuration for LibriSpeech dataset."""

  def __init__(self, data_dir=None):
    super(LibrispeechDataset, self).__init__(
        'librispeech', data_dir, num_classes=29)

  def tf_record_pattern(self, subset):
    if subset == 'train':
      return os.path.join(self.data_dir, 'train-clean-*.tfrecords')
    elif subset == 'validation':
      return os.path.join(self.data_dir, 'test-clean.tfrecords')
    else:
      return ''

  def num_examples_per_epoch(self, subset='train'):
    del subset
    return 2  # TODO(laigd): currently this is an arbitrary number.


class ImageDataset(Dataset):
  """Abstract class for image datasets."""

  def __init__(self,
               name,
               height,
               width,
               depth=None,
               data_dir=None,
               queue_runner_required=False,
               num_classes=1001):
    super(ImageDataset, self).__init__(name, data_dir, queue_runner_required,
                                       num_classes)
    self.height = height
    self.width = width
    self.depth = depth or 3


class ImagenetDataset(ImageDataset):
  """Configuration for Imagenet dataset."""

  def __init__(self, data_dir=None):
    super(ImagenetDataset, self).__init__(
        'imagenet', 300, 300, data_dir=data_dir)

  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return IMAGENET_NUM_TRAIN_IMAGES
    elif subset == 'validation':
      return IMAGENET_NUM_VAL_IMAGES
    else:
      raise ValueError('Invalid data subset "%s"' % subset)


class Cifar10Dataset(ImageDataset):
  """Configuration for cifar 10 dataset.

  It will mount all the input images to memory.
  """

  def __init__(self, data_dir=None):
    super(Cifar10Dataset, self).__init__(
        'cifar10',
        32,
        32,
        data_dir=data_dir,
        queue_runner_required=True,
        num_classes=11)

  def read_data_files(self, subset='train'):
    """Reads from data file and returns images and labels in a numpy array."""
    assert self.data_dir, ('Cannot call `read_data_files` when using synthetic '
                           'data')
    if subset == 'train':
      filenames = [
          os.path.join(self.data_dir, 'data_batch_%d' % i)
          for i in xrange(1, 6)
      ]
    elif subset == 'validation':
      filenames = [os.path.join(self.data_dir, 'test_batch')]
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

    inputs = []
    for filename in filenames:
      with gfile.Open(filename, 'rb') as f:
        # python2 does not have the encoding parameter
        encoding = {} if six.PY2 else {'encoding': 'bytes'}
        inputs.append(cPickle.load(f, **encoding))
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    all_images = np.concatenate(
        [each_input[b'data'] for each_input in inputs]).astype(np.float32)
    all_labels = np.concatenate(
        [each_input[b'labels'] for each_input in inputs])
    return all_images, all_labels

  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return 50000
    elif subset == 'validation':
      return 10000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)


class COCODataset(ImageDataset):
  """COnfiguration for COCO dataset."""

  def __init__(self, data_dir=None, image_size=300):
    super(COCODataset, self).__init__(
        'coco', image_size, image_size, data_dir=data_dir, num_classes=81)

  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return COCO_NUM_TRAIN_IMAGES
    elif subset == 'validation':
      return COCO_NUM_VAL_IMAGES
    else:
      raise ValueError('Invalid data subset "%s"' % subset)


_SUPPORTED_DATASETS = {
    'imagenet': ImagenetDataset,
    'cifar10': Cifar10Dataset,
    'librispeech': LibrispeechDataset,
    'coco': COCODataset,
}

_SUPPORTED_INPUT_PREPROCESSORS = {
    'imagenet': {
        'default': preprocessing.RecordInputImagePreprocessor,
        'official_models_imagenet': preprocessing.ImagenetPreprocessor,
    },
    'cifar10': {
        'default': preprocessing.Cifar10ImagePreprocessor
    },
    'librispeech': {
        'default': preprocessing.LibrispeechPreprocessor
    },
    'coco': {
        'default': preprocessing.COCOPreprocessor
    },
}


def create_dataset(data_dir, data_name):
  """Create a Dataset instance based on data_dir and data_name."""
  if not data_dir and not data_name:
    # When using synthetic data, use synthetic imagenet images by default.
    data_name = 'imagenet'

  # Infere dataset name from data_dir if data_name is not provided.
  if data_name is None:
    for supported_name in _SUPPORTED_DATASETS:
      if supported_name in data_dir:
        data_name = supported_name
        break
    else:  # Failed to identify dataset name from data dir.
      raise ValueError('Could not identify name of dataset. '
                       'Please specify with --data_name option.')
  if data_name not in _SUPPORTED_DATASETS:
    raise ValueError('Unknown dataset. Must be one of %s' % ', '.join(
        [key for key in sorted(_SUPPORTED_DATASETS.keys())]))

  return _SUPPORTED_DATASETS[data_name](data_dir)
