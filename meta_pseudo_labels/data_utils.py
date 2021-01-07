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

# pylint: disable=logging-format-interpolation
# pylint: disable=unused-import
# pylint: disable=g-long-lambda
# pylint: disable=logging-not-lazy
# pylint: disable=g-direct-tensorflow-import
# pylint: disable=protected-access

r"""Data."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import sys

from absl import logging

import numpy as np

import tensorflow.compat.v1 as tf  # tf

from meta_pseudo_labels import augment
from meta_pseudo_labels import common_utils
from tensorflow.python.tpu import tpu_feed

CIFAR_PATH = ''
CIFAR_MEAN = np.array([0.491400, 0.482158, 0.4465309], np.float32) * 255.
CIFAR_STDDEV = np.array([0.247032, 0.243485, 0.26159], np.float32) * 255.

IMAGENET_PATH = ''
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.
IMAGENET_STDDEV = np.array([0.229, 0.224, 0.225], np.float32) * 255.



################################################################################
#                                                                              #
# OUTSIDE INTERFACE                                                            #
#                                                                              #
################################################################################


def _dataset_service(params, dataset, start_index=None, final_index=None):
  """Wrap `dataset` into `dataset_service`."""
  return dataset


def get_image_mean_and_std(params):
  """Builds `eval_data` depends on `params.dataset_name`."""
  if params.dataset_name.lower() in ['cifar10',
                                     'cifar10_4000',
                                     'cifar10_4000_uda',
                                     'cifar10_4000_mpl',
                                     'cifar100',
                                     'cifar100_10k']:
    return CIFAR_MEAN, CIFAR_STDDEV
  elif params.dataset_name.lower() in ['imagenet',
                                       'imagenet_heldout',
                                       'imagenet_noisy_student',
                                       'imagenet_uda',
                                       'imagenet_10_percent',
                                       'imagenet_10_percent_uda',
                                       'imagenet_10_percent_mpl']:
    return IMAGENET_MEAN, IMAGENET_STDDEV
  else:
    raise ValueError(f'Unknown dataset_name `{params.dataset_name}`')


def convert_and_normalize(params, images):
  """Subtract mean and divide stddev depending on the dataset."""
  dtype = tf.bfloat16 if params.use_bfloat16 else tf.float32
  if 'cifar' in params.dataset_name.lower():
    images = tf.cast(images, dtype)
  else:
    images = tf.image.convert_image_dtype(images, dtype)
  shape = [1, 1, 1, 3] if len(images.shape.as_list()) == 4 else [1, 1, 3]

  mean, stddev = get_image_mean_and_std(params)
  mean = tf.reshape(tf.cast(mean, images.dtype), shape)
  stddev = tf.reshape(tf.cast(stddev, images.dtype), shape)
  images = (images - mean) / stddev

  return images


def get_eval_size(params):
  """Builds `eval_data` depends on `params.dataset_name`."""
  eval_sizes = {
      'cifar10': 10000,
      'cifar10_4000': 10000,
      'cifar10_4000_uda': 10000,
      'cifar10_4000_mpl': 10000,
      'cifar100': 10000,
      'cifar100_10k': 10000,
      'imagenet_heldout': 25000,
      'imagenet': 50000,
      'imagenet_noisy_student': 50000,
      'imagenet_uda': 50000,
      'imagenet_10_percent': 50000,
      'imagenet_10_percent_uda': 50000,
      'imagenet_10_percent_mpl': 50000,
  }
  if params.dataset_name.lower() not in eval_sizes.keys():
    raise ValueError(f'Unknown dataset_name `{params.dataset_name}`')
  eval_size = eval_sizes[params.dataset_name.lower()]
  return compute_num_padded_data(params, eval_size)


def build_eval_dataset(params,
                       batch_size=None,
                       num_workers=None,
                       worker_index=None):
  """Builds `eval_data` depends on `params.dataset_name`."""
  if params.dataset_name.lower() in ['cifar10',
                                     'cifar10_4000',
                                     'cifar10_4000_uda',
                                     'cifar10_4000_mpl']:
    eval_data = cifar10_eval(
        params, batch_size=batch_size, eval_mode='test',
        num_workers=num_workers, worker_index=worker_index)
  elif params.dataset_name.lower() in ['imagenet',
                                       'imagenet_noisy_student',
                                       'imagenet_uda',
                                       'imagenet_10_percent',
                                       'imagenet_10_percent_uda',
                                       'imagenet_10_percent_mpl']:
    eval_data = imagenet_eval(
        params, batch_size=batch_size, eval_mode='test',
        num_workers=num_workers, worker_index=worker_index)
  elif params.dataset_name.lower() in ['imagenet_heldout']:
    eval_data = imagenet_eval(
        params, batch_size=batch_size, eval_mode='heldout',
        num_workers=num_workers, worker_index=worker_index)
  else:
    raise ValueError(f'Unknown dataset_name `{params.dataset_name}`')

  return eval_data




def build_train_infeeds(params):
  """Create the TPU infeed ops."""
  dev_assign = params.device_assignment
  host_to_tpus = {}
  for replica_id in range(params.num_replicas):
    host_device = dev_assign.host_device(replica=replica_id, logical_core=0)
    tpu_ordinal = dev_assign.tpu_ordinal(replica=replica_id, logical_core=0)
    logging.info(f'replica_id={replica_id} '
                 f'host_device={host_device} '
                 f'tpu_ordinal={tpu_ordinal}')

    if host_device not in host_to_tpus:
      host_to_tpus[host_device] = [tpu_ordinal]
    else:
      assert tpu_ordinal not in host_to_tpus[host_device]
      host_to_tpus[host_device].append(tpu_ordinal)

  infeed_ops = []
  infeed_graphs = []
  num_inputs = len(host_to_tpus)
  for i, (host, tpus) in enumerate(host_to_tpus.items()):
    infeed_graph = tf.Graph()
    infeed_graphs.append(infeed_graph)
    with infeed_graph.as_default():
      def enqueue_fn(host_device=host, input_index=i, device_ordinals=tpus):
        """Docs."""
        worker_infeed_ops = []
        with tf.device(host_device):
          dataset = build_train_dataset(
              params=params,
              batch_size=params.train_batch_size // num_inputs,
              num_inputs=num_inputs,
              input_index=input_index)
          inputs = tf.data.make_one_shot_iterator(dataset).get_next()

          if params.use_xla_sharding and params.num_cores_per_replica > 1:
            inputs, partition_dims = pad_inputs_for_xla_sharding(params, inputs)
            num_splits = len(device_ordinals)
            if len(device_ordinals) > 1:
              inputs = [tf.split(v, num_splits, 0) for v in inputs]
            else:
              inputs = [[v] for v in inputs]
            q = tpu_feed._PartitionedInfeedQueue(
                number_of_tuple_elements=len(inputs),
                host_id=int(host_device.split('/task:')[-1].split('/')[0]),
                input_partition_dims=partition_dims,
                device_assignment=dev_assign)
            inputs = [[v[i] for v in inputs] for i in range(num_splits)]
            worker_infeed_ops.extend(q.generate_enqueue_ops(inputs))
          else:
            num_splits = len(device_ordinals)
            if len(device_ordinals) > 1:
              inputs = [tf.split(v, num_splits, 0) for v in inputs]
            else:
              inputs = [[v] for v in inputs]
            input_shapes = [v[0].shape for v in inputs]
            for j, device_ordinal in enumerate(device_ordinals):
              worker_infeed_ops.append(tf.raw_ops.InfeedEnqueueTuple(
                  inputs=[v[j] for v in inputs],
                  shapes=input_shapes,
                  device_ordinal=device_ordinal))
        return worker_infeed_ops
      def _body(i):
        with tf.control_dependencies(enqueue_fn()):
          return i+1
      infeed_op = tf.while_loop(
          lambda step: tf.less(step, tf.cast(params.save_every, step.dtype)),
          _body, [0], parallel_iterations=1, name='train_infeed').op
      infeed_ops.append(infeed_op)

  return infeed_ops, infeed_graphs


def build_eval_infeeds(params):
  """Create the TPU infeed ops."""

  eval_size = get_eval_size(params)
  num_eval_steps = eval_size // params.eval_batch_size

  dev_assign = params.device_assignment
  host_to_tpus = {}
  for replica_id in range(params.num_replicas):
    host_device = dev_assign.host_device(replica=replica_id, logical_core=0)
    tpu_ordinal = dev_assign.tpu_ordinal(replica=replica_id, logical_core=0)

    if host_device not in host_to_tpus:
      host_to_tpus[host_device] = [tpu_ordinal]
    else:
      assert tpu_ordinal not in host_to_tpus[host_device]
      host_to_tpus[host_device].append(tpu_ordinal)

  infeed_ops = []
  infeed_graphs = []
  num_inputs = len(host_to_tpus)
  for i, (host, tpus) in enumerate(host_to_tpus.items()):
    infeed_graph = tf.Graph()
    infeed_graphs.append(infeed_graph)
    with infeed_graph.as_default():
      def enqueue_fn(host_device=host, input_index=i, device_ordinals=tpus):
        """Docs."""
        worker_infeed_ops = []
        with tf.device(host_device):
          dataset = build_eval_dataset(
              params,
              batch_size=params.eval_batch_size // num_inputs,
              num_workers=num_inputs,
              worker_index=input_index)
          inputs = tf.data.make_one_shot_iterator(dataset).get_next()

          if params.use_xla_sharding and params.num_cores_per_replica > 1:
            inputs, partition_dims = pad_inputs_for_xla_sharding(params, inputs)
            num_splits = len(device_ordinals)
            if len(device_ordinals) > 1:
              inputs = [tf.split(v, num_splits, 0) for v in inputs]
            else:
              inputs = [[v] for v in inputs]

            q = tpu_feed._PartitionedInfeedQueue(
                number_of_tuple_elements=len(inputs),
                host_id=int(host_device.split('/task:')[-1].split('/')[0]),
                input_partition_dims=partition_dims,
                device_assignment=dev_assign)
            inputs = [[v[i] for v in inputs] for i in range(num_splits)]
            worker_infeed_ops.extend(q.generate_enqueue_ops(inputs))
          else:
            num_splits = len(device_ordinals)
            if len(device_ordinals) > 1:
              inputs = [tf.split(v, num_splits, 0) for v in inputs]
            else:
              inputs = [[v] for v in inputs]
            input_shapes = [v[0].shape for v in inputs]
            for j, device_ordinal in enumerate(device_ordinals):
              worker_infeed_ops.append(tf.raw_ops.InfeedEnqueueTuple(
                  inputs=[v[j] for v in inputs],
                  shapes=input_shapes,
                  device_ordinal=device_ordinal))
        return worker_infeed_ops
      def _body(i):
        with tf.control_dependencies(enqueue_fn()):
          return i+1
      infeed_op = tf.while_loop(
          lambda step: tf.less(step, tf.cast(num_eval_steps, step.dtype)),
          _body, [0], parallel_iterations=1, name='eval_infeed').op
      infeed_ops.append(infeed_op)

  return infeed_ops, infeed_graphs, eval_size


def build_train_dataset(params, batch_size=None, num_inputs=1, input_index=0):
  """Builds `train_data` and `eval_data` depends on `params.dataset_name`."""
  if params.dataset_name.lower() == 'cifar10':
    dataset = cifar10_train(params, batch_size)
    dataset = _dataset_service(params, dataset)
  elif params.dataset_name.lower() == 'cifar10_4000':
    dataset = cifar10_4000(params, batch_size)
    dataset = _dataset_service(params, dataset)
  elif params.dataset_name.lower() in ['cifar10_4000_uda', 'cifar10_4000_mpl']:
    dataset = cifar10_4000_uda(params, batch_size)
    dataset = _dataset_service(params, dataset)
  elif params.dataset_name.lower() == 'imagenet':
    assert num_inputs is not None and input_index is not None
    num_shards_per_input = 1024 // num_inputs
    start_index = num_shards_per_input * input_index
    final_index = num_shards_per_input * (input_index+1)
    dataset = imagenet_train(params,
                             batch_size=batch_size,
                             start_index=start_index,
                             final_index=final_index)
    if params.dataset_service_replicas is not None:
      num_addresses_per_index = params.dataset_service_replicas // num_inputs
      start_address = num_addresses_per_index * input_index
      final_address = num_addresses_per_index * (input_index + 1)
      dataset = _dataset_service(params, dataset,
                                 start_index=start_address,
                                 final_index=final_address)
      logging.info(f'dataset_service: '
                   f'start_address={start_address:<7d}'
                   f'final_address={final_address:<7d}')
    return dataset
  elif params.dataset_name.lower() in ['imagenet_10_percent_uda',
                                       'imagenet_10_percent_mpl']:
    num_shards_per_input = 1024 // num_inputs
    start_index = num_shards_per_input * input_index
    final_index = num_shards_per_input * (input_index+1)
    dataset = imagenet_10_percent_uda(params,
                                      batch_size=batch_size,
                                      start_index=start_index,
                                      final_index=final_index)
    if params.dataset_service_replicas is not None:
      num_addresses_per_index = params.dataset_service_replicas // num_inputs
      start_address = num_addresses_per_index * input_index
      final_address = num_addresses_per_index * (input_index + 1)
      dataset = _dataset_service(params, dataset,
                                 start_index=start_address,
                                 final_index=final_address)
      logging.info('dataset_service: ' +
                   f'start_address={start_address:<7d} ' +
                   f'final_address={final_address:<7d}')
    return dataset
  elif params.dataset_name.lower() == 'imagenet_10_percent':
    dataset = imagenet_train(params, batch_size=batch_size,
                             start_index=0, final_index=128)
    if params.dataset_service_replicas is not None:
      num_addresses_per_index = params.dataset_service_replicas // num_inputs
      start_address = num_addresses_per_index * input_index
      final_address = num_addresses_per_index * (input_index + 1)
      dataset = _dataset_service(params, dataset,
                                 start_index=start_address,
                                 final_index=final_address)
      logging.info('dataset_service: ' +
                   f'start_address={start_address:<7d} ' +
                   f'final_address={final_address:<7d}')
    return dataset
  else:
    raise ValueError(f'Unknown dataset_name `{params.dataset_name}`')

  return dataset


def _smallest_mul(a, m):
  return (a + m-1) // m * m


def compute_num_padded_data(params, data_size):
  """Compute number of eval steps."""
  return (_smallest_mul(data_size, params.eval_batch_size)
          + params.eval_batch_size)


def _add_sample_weight(params, dataset, num_images):
  """Maps a `Dataset` of `d_0, d_1, ...` to compute its `sample_weights`."""

  if 'eval_image_size' in params:
    image_size = max(params.image_size, params.eval_image_size)
  else:
    image_size = params.image_size

  dtype = tf.bfloat16 if params.use_bfloat16 else tf.float32
  dummy_dataset = tf.data.Dataset.from_tensors((
      tf.zeros([image_size, image_size, 3], dtype=dtype),
      tf.zeros([params.num_classes], dtype=tf.float32),
      tf.constant(0., dtype=tf.float32),
  )).repeat()

  def _transform(images, labels):
    return images, labels, tf.constant(1., dtype=tf.float32)
  dataset = dataset.map(_transform, tf.data.experimental.AUTOTUNE)
  dataset = dataset.concatenate(dummy_dataset)

  if params.num_cores_per_replica < params.num_cores_per_worker:
    num_inputs = params.num_workers
  else:
    num_inputs = params.num_replicas
  dataset = dataset.take(num_images // num_inputs).cache().repeat()

  return dataset


def _optimize_dataset(dataset):
  """Routines to optimize `Dataset`'s speed."""
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  options = tf.data.Options()
  options.experimental_optimization.parallel_batch = True
  options.experimental_optimization.map_fusion = True
  options.experimental_optimization.map_vectorization.enabled = True
  options.experimental_optimization.map_parallelization = True
  dataset = dataset.with_options(options)
  dataset = dataset.prefetch(1)
  return dataset


def _flip_and_jitter(x, replace_value=0):
  """Flip left/right and jitter."""
  x = tf.image.random_flip_left_right(x)
  image_size = min([x.shape[0], x.shape[1]])
  pad_size = image_size // 8
  x = tf.pad(x, paddings=[[pad_size, pad_size], [pad_size, pad_size], [0, 0]],
             constant_values=replace_value)
  x = tf.image.random_crop(x, [image_size, image_size, 3])
  x.set_shape([image_size, image_size, 3])
  return x


def _jitter(x, replace_value=0):
  """Flip left/right and jitter."""
  image_size = min([x.shape[0], x.shape[1]])
  pad_size = image_size // 8
  x = tf.pad(x, paddings=[[pad_size, pad_size], [pad_size, pad_size], [0, 0]],
             constant_values=replace_value)
  x = tf.image.random_crop(x, [image_size, image_size, 3])
  x.set_shape([image_size, image_size, 3])
  return x


################################################################################
#                                                                              #
# CIFAR-10                                                                     #
#                                                                              #
################################################################################


def _cifar10_parser(params, value, training):
  """Cifar10 parser."""
  image_size = params.image_size
  value = tf.io.decode_raw(value, tf.uint8)
  label = tf.cast(value[0], tf.int32)
  label = tf.one_hot(label, depth=params.num_classes, dtype=tf.float32)
  image = tf.reshape(value[1:], [3, 32, 32])  # uint8
  image = tf.transpose(image, [1, 2, 0])
  if image_size != 32:
    image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
  image.set_shape([image_size, image_size, 3])

  if training:
    if params.use_augment:
      aug = augment.RandAugment(cutout_const=image_size//8,
                                translate_const=image_size//8,
                                magnitude=params.augment_magnitude)
      image = _flip_and_jitter(image, 128)
      image = aug.distort(image)
      image = augment.cutout(image, pad_size=image_size//4, replace=128)
    else:
      image = _flip_and_jitter(image, 128)
  image = convert_and_normalize(params, image)
  return image, label


def cifar10_train(params, batch_size=None):
  """Load CIFAR-10 data."""
  shuffle_size = batch_size * 16

  filenames = [os.path.join(CIFAR_PATH, 'train.bin')]
  record_bytes = 1 + (3 * 32 * 32)
  dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
  dataset = dataset.map(
      lambda x: _cifar10_parser(params, x, training=True),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.shuffle(shuffle_size).repeat()
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
  dataset = _optimize_dataset(dataset)

  return dataset


def cifar10_4000(params, batch_size=None):
  """Load CIFAR-10 data."""
  shuffle_size = batch_size * 16

  filenames = [os.path.join(CIFAR_PATH, 'train.bin')]
  record_bytes = 1 + (3 * 32 * 32)
  dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
  dataset = dataset.take(4000).repeat().shuffle(shuffle_size)

  dataset = dataset.map(
      lambda x: _cifar10_parser(params, x, training=True),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
  dataset = _optimize_dataset(dataset)

  return dataset


def cifar10_4000_uda(params, batch_size=None):
  """Load CIFAR-10 data."""
  shuffle_size = batch_size * 16

  def _lab_parser(value):
    """Cifar10 parser."""
    image_size = params.image_size
    value = tf.io.decode_raw(value, tf.uint8)
    label = tf.cast(value[0], tf.int32)
    label = tf.one_hot(label, depth=params.num_classes, dtype=tf.float32)
    image = tf.reshape(value[1:], [3, 32, 32])  # uint8
    image = tf.transpose(image, [1, 2, 0])
    if image_size != 32:
      image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
    image.set_shape([image_size, image_size, 3])
    image = _flip_and_jitter(image, 128)
    image = convert_and_normalize(params, image)
    return {'images': image, 'labels': label}

  filenames = [os.path.join(CIFAR_PATH, 'train.bin')]
  record_bytes = 1 + (3 * 32 * 32)
  lab = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
  lab = lab.take(4000).repeat().shuffle(shuffle_size)
  lab = lab.map(_lab_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  lab = lab.prefetch(tf.data.experimental.AUTOTUNE)
  lab = lab.batch(batch_size=batch_size, drop_remainder=True)

  def _unl_parser(value):
    """Cifar10 parser."""
    image_size = params.image_size
    value = tf.io.decode_raw(value, tf.uint8)
    image = tf.reshape(value[1:], [3, 32, 32])  # uint8
    image = tf.transpose(image, [1, 2, 0])
    if image_size != 32:
      image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
    image.set_shape([image_size, image_size, 3])
    image = _flip_and_jitter(image, replace_value=128)
    ori_image = image

    aug = augment.RandAugment(cutout_const=image_size//8,
                              translate_const=image_size//8,
                              magnitude=params.augment_magnitude)
    aug_image = aug.distort(image)
    aug_image = augment.cutout(aug_image, pad_size=image_size//4, replace=128)
    aug_image = _flip_and_jitter(aug_image, replace_value=128)

    ori_image = convert_and_normalize(params, ori_image)
    aug_image = convert_and_normalize(params, aug_image)
    return {'ori_images': ori_image, 'aug_images': aug_image}

  unl = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
  unl = unl.repeat().shuffle(shuffle_size)
  unl = unl.map(_unl_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  unl = unl.prefetch(tf.data.experimental.AUTOTUNE)
  unl = unl.batch(batch_size=batch_size*params.uda_data, drop_remainder=True)

  def _merge_datasets(a, b):
    """Merge two `Dict` of data."""
    return a['images'], a['labels'], b['ori_images'], b['aug_images']
  dataset = tf.data.Dataset.zip((lab, unl))
  dataset = dataset.map(_merge_datasets, tf.data.experimental.AUTOTUNE)
  dataset = _optimize_dataset(dataset)

  return dataset


def cifar10_eval(params, batch_size=None, eval_mode=None,
                 num_workers=None, worker_index=None):
  """Load CIFAR-10 data."""

  if batch_size is None:
    batch_size = params.eval_batch_size

  if eval_mode == 'valid':
    filenames = [os.path.join(CIFAR_PATH, 'val.bin')]
  elif eval_mode == 'test':
    filenames = [os.path.join(CIFAR_PATH, 'test_batch.bin')]
  else:
    raise ValueError(f'Unknown eval_mode {eval_mode}')

  record_bytes = 1 + (3 * 32 * 32)
  dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
  if num_workers is not None and worker_index is not None:
    dataset = dataset.shard(num_workers, worker_index)

  dataset = dataset.map(
      lambda x: _cifar10_parser(params, x, training=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = _add_sample_weight(params, dataset,
                               num_images=get_eval_size(params))

  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
  dataset = _optimize_dataset(dataset)

  return dataset


################################################################################
#                                                                              #
# IMAGENET                                                                     #
#                                                                              #
################################################################################


def _decode_and_center_crop(image_bytes, image_size):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((224 / (224 + 32)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = tf.image.resize_bicubic([image], [image_size, image_size])[0]

  return image


def _decode_and_random_crop(image_bytes, image_size):
  """Make a random crop of image_size."""

  def _distorted_bounding_box_crop(image_bytes,
                                   bbox,
                                   min_object_covered=0.1,
                                   aspect_ratio_range=(3./4., 4./3.),
                                   area_range=(0.08, 1.0),
                                   max_attempts=10):
    """See `tf.image.sample_distorted_bounding_box` for more documentation."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return image

  def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)

  bboxes = tf.constant([0.0, 0.0, 1.0, 1.0], tf.float32, shape=[1, 1, 4])
  image = _distorted_bounding_box_crop(image_bytes, bbox=bboxes)
  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad,
      lambda: _decode_and_center_crop(image_bytes, image_size),
      lambda: tf.image.resize_bicubic([image], [image_size, image_size])[0])
  image.set_shape([image_size, image_size, 3])

  return image


def _imagenet_parser(params, value, training):
  """ImageNet parser."""
  if training:
    image_size = params.image_size
  else:
    if 'eval_image_size' in params:
      image_size = max(params.image_size, params.eval_image_size)
    else:
      image_size = params.image_size
    logging.info(f'eval_image_size={image_size}')

  keys_to_features = {
      'image/encoded': tf.io.FixedLenFeature((), tf.string),
      'image/class/label': tf.io.FixedLenFeature([], tf.int64),
  }
  features = tf.io.parse_single_example(value, keys_to_features)
  label = tf.cast(features['image/class/label'], tf.int32) - 1
  label = tf.one_hot(label, depth=params.num_classes, dtype=tf.float32)

  if training:
    image = _decode_and_random_crop(features['image/encoded'], image_size)
    image = tf.image.random_flip_left_right(image)
    if params.use_augment:
      aug = augment.RandAugment(magnitude=params.augment_magnitude)
      image = aug.distort(image)
  else:
    image = _decode_and_center_crop(features['image/encoded'], image_size)
  image.set_shape([image_size, image_size, 3])
  image = convert_and_normalize(params, image)
  return image, label


def imagenet_train(params, batch_size=None, start_index=0, final_index=1024):
  """Load ImageNet data."""
  def _index_to_filename(index):
    """Turn `index` into `train-[]-of-01024`."""
    if params.dataset_name == 'imagenet_10_percent':
      index = tf.strings.as_string(tf.math.mod(index, 128), width=5, fill='0')
      filename = tf.strings.join(
          inputs=[IMAGENET_PATH, '/ten_percent-', index, '-of-00128'],
          separator='')
    else:
      index = tf.strings.as_string(index, width=5, fill='0')
      filename = tf.strings.join(
          inputs=[IMAGENET_PATH, '/train-', index, '-of-01024'],
          separator='')
    return filename

  logging.info(f'ImageNet shards: start={start_index} final={final_index}')
  dataset = tf.data.Dataset.range(start_index, final_index)
  dataset = dataset.repeat()
  dataset = dataset.shuffle(final_index - start_index)

  dataset = dataset.interleave(
      lambda f: tf.data.TFRecordDataset(_index_to_filename(f)).map(
          lambda x: _imagenet_parser(params, x, training=True),
          tf.data.experimental.AUTOTUNE),
      cycle_length=tf.data.experimental.AUTOTUNE,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.shuffle(
      1024*16 if params.use_tpu and not params.running_local_dev else 100)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
  dataset = _optimize_dataset(dataset)

  return dataset


def imagenet_eval(params, batch_size=None, eval_mode=None, num_workers=None,
                  worker_index=None):
  """Load ImageNet data."""
  assert eval_mode in ['heldout', 'test']

  def _index_to_filename(index):
    """Turn `index` into `validation-00abc-of-00128`."""
    index = tf.strings.as_string(index, width=5, fill='0')
    if eval_mode == 'heldout':
      filename = tf.strings.join(
          inputs=[IMAGENET_PATH, '/heldout-', index, '-of-00128'],
          separator='')
    elif eval_mode == 'test':
      filename = tf.strings.join(
          inputs=[IMAGENET_PATH, '/validation-', index, '-of-00128'],
          separator='')
    else:
      raise ValueError(f'Unknown eval_mode `{eval_mode}`')
    return filename

  if batch_size is None:
    batch_size = params.eval_batch_size

  if num_workers is None and worker_index is None:
    num_workers = 1
    worker_index = 0
  logging.info(f'eval_input {worker_index+1}/{num_workers}')
  dataset = tf.data.Dataset.range(128).shard(num_workers, worker_index)
  num_images = get_eval_size(params)

  dataset = dataset.flat_map(
      lambda f: tf.data.TFRecordDataset(_index_to_filename(f)).map(
          lambda x: _imagenet_parser(params, x, training=False),
          tf.data.experimental.AUTOTUNE))
  dataset = _add_sample_weight(params, dataset, num_images)

  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
  dataset = _optimize_dataset(dataset)

  return dataset


def imagenet_10_percent_uda(params, batch_size=None,
                            start_index=0, final_index=1024):
  """Load ImageNet data."""
  def _index_to_lab_filename(index):
    """Turn `index` into `train-[]-of-01024`."""
    index = tf.strings.as_string(index, width=5, fill='0')
    filename = tf.strings.join(
        inputs=[IMAGENET_PATH, '/ten_percent-', index, '-of-00128'],
        separator='')
    return filename

  def _lab_parser(value):
    """ImageNet parser."""
    image_size = params.image_size
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        'image/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    features = tf.io.parse_single_example(value, keys_to_features)
    label = tf.cast(features['image/class/label'], tf.int32) - 1
    label = tf.one_hot(label, depth=params.num_classes, dtype=tf.float32)

    image = _decode_and_random_crop(features['image/encoded'], image_size)
    image = tf.image.random_flip_left_right(image)
    image.set_shape([image_size, image_size, 3])

    image = convert_and_normalize(params, image)
    return {'images': image, 'labels': label}

  lab = tf.data.Dataset.range(128).repeat().shuffle(128)
  lab = lab.interleave(
      lambda f: tf.data.TFRecordDataset(_index_to_lab_filename(f)).map(
          _lab_parser, tf.data.experimental.AUTOTUNE),
      cycle_length=tf.data.experimental.AUTOTUNE,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  lab = lab.shuffle(1024*8 if params.use_tpu else 100)
  lab = lab.prefetch(tf.data.experimental.AUTOTUNE)
  lab = lab.batch(batch_size=batch_size, drop_remainder=True)

  def _index_to_unl_filename(index):
    """Turn `index` into `train-[]-of-01024`."""
    index = tf.strings.as_string(index, width=5, fill='0')
    filename = tf.strings.join(
        inputs=[IMAGENET_PATH, '/train-', index, '-of-01024'], separator='')
    return filename

  def _unl_parser(value):
    """ImageNet parser."""
    image_size = params.image_size
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
    }
    features = tf.io.parse_single_example(value, keys_to_features)
    image = _decode_and_random_crop(features['image/encoded'], image_size)

    ori_image = tf.image.random_flip_left_right(image)
    ori_image.set_shape([image_size, image_size, 3])

    aug_image = tf.image.random_flip_left_right(image)
    aug = augment.RandAugment(magnitude=params.augment_magnitude)
    aug_image = aug.distort(aug_image)
    aug_image.set_shape([image_size, image_size, 3])

    ori_image = convert_and_normalize(params, ori_image)
    aug_image = convert_and_normalize(params, aug_image)
    return {'ori_images': ori_image, 'aug_images': aug_image}

  unl = tf.data.Dataset.range(start_index, final_index)
  unl = unl.repeat().shuffle(final_index - start_index)
  unl = unl.interleave(
      lambda f: tf.data.TFRecordDataset(_index_to_unl_filename(f)).map(
          _unl_parser, tf.data.experimental.AUTOTUNE),
      cycle_length=tf.data.experimental.AUTOTUNE,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  unl = unl.shuffle(1024*16 if params.use_tpu else 100)
  unl = unl.prefetch(tf.data.experimental.AUTOTUNE)
  unl = unl.batch(batch_size=batch_size*params.uda_data, drop_remainder=True)

  def _merge_datasets(a, b):
    """Merge two `Dict` of data."""
    return a['images'], a['labels'], b['ori_images'], b['aug_images']
  dataset = tf.data.Dataset.zip((lab, unl))
  dataset = dataset.map(_merge_datasets, tf.data.experimental.AUTOTUNE)
  dataset = _optimize_dataset(dataset)

  return dataset
