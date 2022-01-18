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

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import jax
import tensorflow as tf
import tensorflow_datasets as tfds


def get_data_scaler(config):
  """Assuming data are always in [0, 1]."""
  if config.data.centered:
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Assuming data are always in [0, 1]."""
  if config.data.centered:
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
      image,
      size=(resolution, resolution),
      antialias=True,
      method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def get_dataset(rng, config, evaluation=False):
  """return the datasets used for training and evaluation.

  Args:
    rng: jax rng state.
    config: a ConfigDict parsed from config files.
    evaluation: True when used in the `eval` mode. Set number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({jax.device_count()})')

  per_device_batch_size = batch_size // jax.device_count()
  data_rng = jax.random.fold_in(rng, jax.host_id())
  train_rng, eval_rng = jax.random.split(data_rng)
  # Reduce this when image resolution is too large and data pointer is stored.
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = config.training.n_epochs if not evaluation else 1
  batch_dims = [jax.local_device_count(), per_device_batch_size]

  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'
  else:
    raise NotImplementedError(
        f'Dataset {config.data.dataset} not yet supported.')

  dataset_builder.download_and_prepare()
  def preprocess_fn(d):
    """Basic preprocessing function scales data to [0, 1) and randomly flips."""
    img = tf.image.convert_image_dtype(d['image'], tf.float16)
    img = tf.image.flip_left_right(img)
    return dict(image=img, label=d['label'])

  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    ds = ds.repeat(num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for batch_size in reversed(batch_dims):
      ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  train_ds = create_dataset(dataset_builder, train_split_name)
  eval_ds = create_dataset(dataset_builder, eval_split_name)
  return train_ds, eval_ds, dataset_builder
