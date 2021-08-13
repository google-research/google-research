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
# pylint: disable=g-long-lambda
# pylint: disable=logging-not-lazy
# pylint: disable=protected-access

r"""Data."""

import os

from absl import logging

import numpy as np

import tensorflow.compat.v1 as tf  # tf

from differentiable_data_selection import augment

CIFAR_PATH = ''
CIFAR_MEAN = np.array([0.491400, 0.482158, 0.4465309], np.float32) * 255.
CIFAR_STDDEV = np.array([0.247032, 0.243485, 0.26159], np.float32) * 255.



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
  if params.dataset_name.lower().startswith('cifar'):
    return CIFAR_MEAN, CIFAR_STDDEV
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
      'cifar10_dds': 10000,
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
  if params.dataset_name.lower() in ['cifar10', 'cifar10_dds']:
    eval_data = cifar10_eval(
        params, batch_size=batch_size, eval_mode='test',
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

          num_splits = len(device_ordinals)
          if len(device_ordinals) > 1:
            inputs = [tf.split(v, num_splits, 0) for v in inputs]
          else:
            inputs = [[v] for v in inputs]
          input_dtypes = [v[0].dtype for v in inputs]
          input_shapes = [v[0].shape for v in inputs]
          params.add_hparam('train_dtypes', input_dtypes)
          params.add_hparam('train_shapes', input_shapes)
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

          num_splits = len(device_ordinals)
          if len(device_ordinals) > 1:
            inputs = [tf.split(v, num_splits, 0) for v in inputs]
          else:
            inputs = [[v] for v in inputs]
          input_dtypes = [v[0].dtype for v in inputs]
          input_shapes = [v[0].shape for v in inputs]
          params.add_hparam('eval_dtypes', input_dtypes)
          params.add_hparam('eval_shapes', input_shapes)
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
  del num_inputs
  del input_index
  if params.dataset_name.lower() == 'cifar10':
    dataset = cifar10_train(params, batch_size)
    dataset = _dataset_service(params, dataset)
  elif params.dataset_name.lower() == 'cifar10_dds':
    dataset = cifar10_dds(params, batch_size)
    dataset = _dataset_service(params, dataset)
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
  dataset = dataset.take(num_images // params.num_workers).cache().repeat()

  return dataset


def _optimize_dataset(dataset):
  """Routines to optimize `Dataset`'s speed."""
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  options = tf.data.Options()
  options.experimental_optimization.parallel_batch = True
  options.experimental_optimization.map_fusion = True
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
  dataset = dataset.skip(5000).cache()
  dataset = dataset.map(
      lambda x: _cifar10_parser(params, x, training=True),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.shuffle(shuffle_size).repeat()
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
  dataset = _optimize_dataset(dataset)

  return dataset


def cifar10_dds(params, batch_size=None):
  """Load CIFAR-10 data."""
  shuffle_size = batch_size * 16

  filenames = [os.path.join(CIFAR_PATH, 'train.bin')]
  record_bytes = 1 + (3 * 32 * 32)
  dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)

  train_dataset = dataset.skip(5000).cache()
  train_dataset = train_dataset.map(
      lambda x: _cifar10_parser(params, x, training=True),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.shuffle(shuffle_size).repeat()
  train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.batch(batch_size=batch_size,
                                      drop_remainder=True)

  valid_dataset = dataset.take(5000).cache()
  valid_dataset = valid_dataset.map(
      lambda x: _cifar10_parser(params, x, training=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  valid_dataset = valid_dataset.shuffle(shuffle_size).repeat()
  valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  valid_dataset = valid_dataset.batch(batch_size=batch_size,
                                      drop_remainder=True)

  dataset = tf.data.Dataset.zip((train_dataset, valid_dataset))
  dataset = dataset.map(lambda a, b: tuple([a[0], a[1], b[0], b[1]]),
                        tf.data.experimental.AUTOTUNE)

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
