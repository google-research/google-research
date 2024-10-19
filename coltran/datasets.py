# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Wrapper for datasets."""

import functools
import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from coltran.utils import datasets_utils


def resize_to_square(image, resolution=32, train=True):
  """Preprocess the image in a way that is OK for generative modeling."""

  # Crop a square-shaped image by shortening the longer side.
  image_shape = tf.shape(image)
  height, width, channels = image_shape[0], image_shape[1], image_shape[2]
  side_size = tf.minimum(height, width)
  cropped_shape = tf.stack([side_size, side_size, channels])
  if train:
    image = tf.image.random_crop(image, cropped_shape)
  else:
    image = tf.image.resize_with_crop_or_pad(
        image, target_height=side_size, target_width=side_size)

  image = datasets_utils.change_resolution(image, res=resolution, method='area')
  return image


def preprocess(example, train=True, resolution=256):
  """Apply random crop (or) central crop to the image."""
  image = example

  is_label = False
  if isinstance(example, dict):
    image = example['image']
    is_label = 'label' in example.keys()

  image = resize_to_square(image, train=train, resolution=resolution)

  # keepng 'file_name' key creates some undebuggable TPU Error.
  example_copy = dict()
  example_copy['image'] = image
  example_copy['targets'] = image
  if is_label:
    example_copy['label'] = example['label']
  return example_copy


def get_gen_dataset(data_dir, batch_size):
  """Converts a list of generated TFRecords into a TF Dataset."""

  def parse_example(example_proto, res=64):
    features = {'image': tf.io.FixedLenFeature([res*res*3], tf.int64)}
    example = tf.io.parse_example(example_proto, features=features)
    image = tf.reshape(example['image'], (res, res, 3))
    return {'targets': image}

  # Provided generated dataset.
  def tf_record_name_to_num(x):
    x = x.split('.')[0]
    x = re.split(r'(\d+)', x)
    return int(x[1])

  assert data_dir is not None
  records = tf.io.gfile.listdir(data_dir)
  max_num = max(records, key=tf_record_name_to_num)
  max_num = tf_record_name_to_num(max_num)

  records = []
  for record in range(max_num + 1):
    path = os.path.join(data_dir, f'gen{record}.tfrecords')
    records.append(path)

  tf_dataset = tf.data.TFRecordDataset(records)
  tf_dataset = tf_dataset.map(parse_example, num_parallel_calls=100)
  tf_dataset = tf_dataset.batch(batch_size=batch_size)
  tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
  return tf_dataset


def create_gen_dataset_from_images(image_dir):
  """Creates a dataset from the provided directory."""
  def load_image(path):
    image_str = tf.io.read_file(path)
    return tf.image.decode_image(image_str, channels=3)

  child_files = tf.io.gfile.listdir(image_dir)
  files = [os.path.join(image_dir, file) for file in child_files]
  files = tf.convert_to_tensor(files, dtype=tf.string)
  dataset = tf.data.Dataset.from_tensor_slices((files))
  return dataset.map(load_image, num_parallel_calls=100)


def get_imagenet(subset, read_config):
  """Gets imagenet dataset."""
  train = subset == 'train'
  num_val_examples = 0 if subset == 'eval_train' else 10000
  if subset == 'test':
    ds = tfds.load('imagenet2012', split='validation', shuffle_files=False)
  else:
    # split 10000 samples from the imagenet dataset for validation.
    ds, info = tfds.load('imagenet2012', split='train', with_info=True,
                         shuffle_files=train, read_config=read_config)
    num_train = info.splits['train'].num_examples - num_val_examples
    if train:
      ds = ds.take(num_train)
    elif subset == 'valid':
      ds = ds.skip(num_train)
  return ds


def get_dataset(name,
                config,
                batch_size,
                subset,
                read_config=None,
                data_dir=None):
  """Wrapper around TF-Datasets.

  * Setting `config.random_channel to be True` adds
    ds['targets_slice'] - Channel picked at random. (of 3).
    ds['channel_index'] - Index of the randomly picked channel
  * Setting `config.downsample` to be True, adds:.
    ds['targets_64'] - Downsampled 64x64 input using tf.resize.
    ds['targets_64_up_back] - 'targets_64' upsampled using tf.resize

  Args:
    name: imagenet
    config: dict
    batch_size: batch size.
    subset: 'train', 'eval_train', 'valid' or 'test'.
    read_config: optional, tfds.ReadConfg instance. This is used for sharding
                 across multiple workers.
    data_dir: Data Directory, Used for Custom dataset.
  Returns:
   dataset: TF Dataset.
  """
  downsample = config.get('downsample', False)
  random_channel = config.get('random_channel', False)
  downsample_res = config.get('downsample_res', 64)
  downsample_method = config.get('downsample_method', 'area')
  num_epochs = config.get('num_epochs', -1)
  data_dir = config.get('data_dir') or data_dir
  auto = tf.data.AUTOTUNE
  train = subset == 'train'

  if name == 'imagenet':
    ds = get_imagenet(subset, read_config)
  elif name == 'custom':
    assert data_dir is not None
    ds = create_gen_dataset_from_images(data_dir)
  else:
    raise ValueError(f'Expected dataset in [imagenet, custom]. Got {name}')

  ds = ds.map(
      lambda x: preprocess(x, train=train), num_parallel_calls=100)
  if train and random_channel:
    ds = ds.map(datasets_utils.random_channel_slice)
  if downsample:
    downsample_part = functools.partial(
        datasets_utils.downsample_and_upsample,
        train=train,
        downsample_res=downsample_res,
        upsample_res=256,
        method=downsample_method)
    ds = ds.map(downsample_part, num_parallel_calls=100)

  if train:
    ds = ds.repeat(num_epochs)
    ds = ds.shuffle(buffer_size=128)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(auto)
  return ds
