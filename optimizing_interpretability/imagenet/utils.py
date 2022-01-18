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

"""Utils for training."""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def add_noise(input_image, noise, multiple_image_std, size=224):
  """Transformation of a single image by adding noise.

  If a random gaussian distribution of noisy is specified (noise='r_normal'),
  the standard deviation of the noise added is based upon the dynamic range of
  the image weighed by multiple_image_std argument. This appears to work
  well empirically, and is the subject of additional research.

  Args:
    input_image: A single input image, float32 tensor
    noise: String that specifies the distribution of noise to add as either a
      gaussian distribution (r_normal) or a uniform distribution (r_uniform).
    multiple_image_std: Weight to place on the range of input values.
    size: size of noise matrix (should match image size)

  Returns:
    noisy_image: The input with the addition of a noise distribution.

  Raises:
      ValueError: Raised if the string specifying the noise distribution does
        not correspond to the noise implementations.
  """
  if noise == 'r_normal':
    image_min = tf.reduce_min(input_image)
    image_max = tf.reduce_max(input_image)
    diff = tf.reduce_mean(tf.subtract(image_max, image_min))
    range_ = tf.to_float(tf.multiply(tf.constant([multiple_image_std]), diff))
    noise = tf.random_normal(
        shape=[size, size, 3], stddev=range_, dtype=tf.float32)
  elif noise == 'r_uniform':
    percentile_ = tfp.stats.percentile(input_image, q=10.)
    noise = tf.random.uniform(
        minval=-percentile_,
        maxval=percentile_,
        shape=[size, size, 3],
        dtype=tf.float32)
  else:
    raise ValueError('Noise type not found:', noise)

  noisy_image = tf.add(input_image, noise)
  return noisy_image


def noise_layer(images,
                labels,
                multiple_image_std=0.15,
                size=224,
                jitter_multiplier=1,
                noise='r_normal'):
  """Add noise to a subset of images in a batch.

  Args:
    images: The batch of images.
    labels: Labels associated with images.
    multiple_image_std: Weight to place on the range of input values.
    size: The size of the image.
    jitter_multiplier: number of images to add noise to.
    noise: String that specifies the distribution of noise to add.

  Returns:
    noisy_images: A set of images (num_images*jitter_multiplier) with injected
      noise.
    tiled_labels: Associated labels for the noisy images.
  """
  images_noise = tf.tile(
      images, multiples=tf.constant([jitter_multiplier, 1, 1, 1], shape=[
          4,
      ]))

  noisy_images = tf.map_fn(
      lambda x: add_noise(x, noise, multiple_image_std, size), images_noise)

  noisy_images = tf.concat([images, noisy_images], axis=0)
  tiled_labels = tf.tile(labels, tf.constant([jitter_multiplier], shape=[1]))
  tiled_labels = tf.concat([labels, tiled_labels], axis=0)

  return noisy_images, tiled_labels


def format_tensors(*dicts):
  """Formats metrics to be callable as tf.summary scalars on tpu's.

  Args:
    *dicts: A set of metric dictionaries, containing metric name + value tensor.

  Returns:
    A single formatted dictionary that holds all tensors.

  Raises:
   ValueError: if any tensor is not a scalar.
  """
  merged_summaries = {}
  for d in dicts:
    for metric_name, value in d.items():
      shape = value.shape.as_list()
      if not shape:
        merged_summaries[metric_name] = tf.expand_dims(value, axis=0)
      elif shape == [1]:
        merged_summaries[metric_name] = value
      else:
        raise ValueError(
            'Metric {} has value {} that is not reconciliable'.format(
                metric_name, value))
  return merged_summaries


def host_call_fn(model_dir, **kwargs):
  """creates training summaries when using TPU.

  Args:
    model_dir: String indicating the output_dir to save summaries in.
    **kwargs: Set of metric names and tensor values for all desired summaries.

  Returns:
    Summary op to be passed to the host_call arg of the estimator function.
  """
  gs = kwargs.pop('global_step')[0]
  with tf.contrib.create_file_writer(model_dir).as_default():
    with tf.contrib.always_record_summaries():
      for name, tensor in kwargs.items():
        tf.summary.scalar(name, tensor[0], step=gs)
      return tf.contrib.summary.all_summary_ops()


def get_lr_schedule(train_steps, num_train_images, train_batch_size):
  """learning rate schedule."""
  steps_per_epoch = np.floor(num_train_images / train_batch_size)
  train_epochs = train_steps / steps_per_epoch
  return [  # (multiplier, epoch to start) tuples
      (1.0, np.floor(5 / 90 * train_epochs)),
      (0.1, np.floor(30 / 90 * train_epochs)),
      (0.01, np.floor(60 / 90 * train_epochs)),
      (0.001, np.floor(80 / 90 * train_epochs))
  ]


def learning_rate_schedule(params, current_epoch, train_batch_size,
                           num_train_images):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  Args:
    params: Python dict containing parameters for this run.
    current_epoch: `Tensor` for current epoch.
    train_batch_size: batch size adjusted for PIE
    num_train_images: total number of train images

  Returns:
    A scaled `Tensor` for current learning rate.
  """

  scaled_lr = params['base_learning_rate'] * (train_batch_size / 256.0)

  lr_schedule = get_lr_schedule(
      train_steps=params['train_steps'],
      num_train_images=num_train_images,
      train_batch_size=train_batch_size)

  decay_rate = (
      scaled_lr * lr_schedule[0][0] * current_epoch / lr_schedule[0][1])
  for mult, start_epoch in lr_schedule:
    decay_rate = tf.where(current_epoch < start_epoch, decay_rate,
                          scaled_lr * mult)
  return decay_rate
