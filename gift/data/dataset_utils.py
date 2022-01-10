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

"""Common utils for used by different dataset builders."""

import collections
import math

from absl import logging
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

Dtype = collections.namedtuple('dtype', ['tf_dtype', 'jax_dtype'])
DATA_TYPE = {
    'float16': Dtype(tf.float16, jnp.float16),
    'float32': Dtype(tf.float32, jnp.float32),
    'int32': Dtype(tf.int32, jnp.int32),
}

Dataset = collections.namedtuple(
    'Dataset', ['train_iter', 'valid_iter', 'test_iter', 'meta_data'])


def shard(pytree, n_devices=None):
  """Reshapes all arrays in the pytree to add a leading n_devices dimension.

  Note: We assume that all arrays in the pytree have leading dimension divisble
  by n_devices and reshape (host_batch_size, height, width, channel) to
  (local_devices, device_batch_size, height, width, channel).

  Args:
    pytree: A pytree of arrays to be sharded.
    n_devices: If None, this will be set to jax.local_device_count().

  Returns:
    Sharded data.
  """
  if n_devices is None:
    n_devices = jax.local_device_count()

  def _shard_array(array):
    return array.reshape((n_devices, -1) + array.shape[1:])

  return jax.tree_map(_shard_array, pytree)


def normalize(image, dtype=tf.float32):
  """Normalizes the value of pixels in the given image to the [0,1] interval.

  Args:
    image: `Tensor` representing an image binary of arbitrary size.
    dtype: Tensorflow data type, Data type of the image.

  Returns:
    A normalized image `Tensor`.
  """
  image = tf.cast(image, dtype=dtype)
  if dtype not in [tf.int32, tf.int64, tf.uint32, tf.uint64]:
    image /= tf.constant(255.0, shape=[1, 1, 1], dtype=dtype)
  return image


def degree2radian(angel_in_degree):
  """Converts degrees to radians.

  Args:
    angel_in_degree: float; Angle value in degrees.

  Returns:
    Converted angle value.
  """

  return (angel_in_degree / 180.0) * math.pi


def rotate_image(img, angle):
  """Rotates the input image.

  Args:
    img: float array; Input image with shape `[w, h, c]`.
    angle: float; The rotation angle in degrees.

  Returns:
    Rotated image.
  """
  new_img = tfa.image.rotate(
      images=[img], angles=[degree2radian(angle)], interpolation='nearest')[0]

  return new_img


def rotated_data_builder(image, env_name):
  """Tranform the image based on env_name."""
  start_angle, end_angle = map(float, env_name.split('_'))
  if start_angle == end_angle:
    angle = start_angle
  else:
    angle = tf.random.uniform(
        shape=(1,), minval=start_angle, maxval=end_angle)[0]

  image = rotate_image(image, angle=angle)

  return image


def split_convertor(split, from_percent, to_percent):
  """Converting tfds split str to include only examples in the specified range.

  Example:
  `train[10:60]`,  50, 100 --> `train[35:60]`

  Args:
    split: str; split name (with or without percentage).
    from_percent: int; Start of the range in percents.
    to_percent: int; End of the range in percents.

  Returns:
    Converted split name.
  """

  # Deconstruct the split name:
  split_parts = split.replace(']', '').replace('%', '').split('[')
  if len(split_parts) == 2:
    split_name = split_parts[0]
    split_from, split_to = split_parts[1].split(':')

    split_from = split_from or '0'
    split_to = split_to or '100'
  elif len(split_parts) == 1:
    split_name = split_parts[0]
    split_from, split_to = '0', '100'

  split_from, split_to = float(split_from), float(split_to)
  scale = float(split_to - split_from) / 100.0

  translated_from = int(onp.floor(from_percent * scale + split_from))
  translated_to = int(onp.ceil(to_percent * scale + split_from))

  return f'{split_name}[{translated_to}%:{translated_from}%]'


def get_data_range(builder, split, host_id, host_count):
  """Return a (sub)split adapted to a given host.

  Each host reads a (same-size) subset of the given `split` such that they
  all go through different parts of it. For doing so, we need to do some
  indexing computation on a given `builder` and `split`, because we want to
  support using contiguous subsets such as `train[10%:20%]`.


  Args:
    builder: TFDS Builder for the datset.
    split: str; Split for which we want to create the data ranage.
    host_id: int; Id of the host.
    host_count: int; Total Number of hosts.

  Returns:
    Data range for the current host and the given split.
  """
  # 1. canonicalize input to absolute indices
  abs_ri = tfds.core.ReadInstruction.from_spec(split).to_absolute(
      builder.info.splits)

  # 2. make sure it's only 1 continuous block
  assert len(abs_ri) == 1, 'Multiple non-contiguous TFDS splits not supported'
  full_range = abs_ri[0]
  # 3. get its start/end indices
  full_start = full_range.from_ or 0
  full_end = full_range.to or builder.info.splits[
      full_range.splitname].num_examples

  # 4. compute each host's subset
  # all hosts should perform exactly the same number iterations. To make sure
  # this is the case we ensure that all hosts get the same number of images.
  # some setups, do padding instead, at least for the validation set
  examples_per_host = (full_end - full_start) // host_count
  host_start = full_start + examples_per_host * host_id
  host_end = full_start + examples_per_host * (host_id + 1)
  logging.info('Host %d data range: from %s to %s (from split %s)',
               jax.host_id(), host_start, host_end, full_range.splitname)
  return full_range.splitname, host_start, host_end


def get_num_examples(dataset, split, data_dir=None):
  """Returns the total number of examples in a dataset split."""
  builder = tfds.builder(dataset, data_dir=data_dir)

  n = 0
  host_count = jax.host_count()
  for ihost in range(host_count):
    _, start, end = get_data_range(builder, split, ihost, host_count)
    n += end - start

  remainder = builder.info.splits[split].num_examples - n
  if remainder:
    warning = (f'Dropping {remainder} examples for the '
               f'{builder.info.name} dataset, {split} split. '
               f'The reason is that all hosts should have the same number '
               f'of examples in order to guarantee that they stay in sync.')
    logging.warning(warning)

  return n


@tf.function
def perturb_image(image, perturb_params):
  """Transform the input image based on perturb_params.

  Args:
    image: tensor; [height, width, dim].
    perturb_params: dict; Perturbation options.
      Should be set: base_size; (original input size) Scale; scale_factor
        (between 0 and 2) Translation; translate_factor (between -1 and 1)

  Returns:
    Perturbed image: tensor; [height, width, dim].
  """
  input_image_type = image.dtype
  if input_image_type in ['uint8', 'int32', 'int64']:
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0

  # Scale
  scale_factor = perturb_params.get('scale_factor', 1.0)
  scale = int(scale_factor * perturb_params['base_size'])

  image = tf.image.resize(image[None, Ellipsis], (scale, scale), 'lanczos5')[0]
  image = tf.image.resize_with_crop_or_pad(image, perturb_params['final_size'],
                                           perturb_params['final_size'])

  # Translate (shifting image in x|y direction)
  # Maximum amount of shit: -scale so that the image does not fall out of the
  # frame.
  max_shift = float(perturb_params['final_size'] - scale)
  # Horizental shift
  shift_x = perturb_params.get('translate_factor', 0.0) * max_shift / 2
  # Vertical shift
  shift_y = perturb_params.get('translate_factor', 0.0) * max_shift / 2

  image = tfa.image.translate(image, (shift_x, shift_y), 'BILINEAR')

  # Blur
  if perturb_params.get('blur_factor', 0.0):
    image = tfa.image.gaussian_filter2d(
        image=image,
        filter_shape=[5, 5],
        sigma=perturb_params.get('blur_factor', 0.0),
        padding='REFLECT',
        constant_values=0)

  if input_image_type in ['uint8', 'int32', 'int64']:
    image = tf.cast(image * 255.0, dtype=input_image_type)
  return image
