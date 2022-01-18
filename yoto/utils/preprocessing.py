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

"""Implementation of data preprocessing ops.

All preprocessing ops should return a data processing functors. A data
is represented as a dictionary of tensors, where field "image" is reserved
for 3D images (height x width x channels). The functors output dictionary with
field "image" being modified. Potentially, other fields can also be modified
or added.
"""

import collections
from absl import logging
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2


def maybe_repeat(arg, n_reps):
  if not isinstance(arg, collections.Sequence):
    arg = (arg,) * n_reps
  return arg


def tf_apply_to_image_or_images(fn, image_or_images, **map_kw):
  """Applies a function to a single image or each image in a batch of them.

  Args:
    fn: the function to apply, receives an image, returns an image.
    image_or_images: Either a single image, or a batch of images.
    **map_kw: Arguments passed through to tf.map_fn if called.

  Returns:
    The result of applying the function to the image or batch of images.

  Raises:
    ValueError: if the input is not of rank 3 or 4.
  """
  static_rank = image_or_images.shape.rank
  if static_rank == 3:  # A single image: HWC
    return fn(image_or_images)
  elif static_rank == 4:  # A batch of images: BHWC
    return tf.map_fn(fn, image_or_images, **map_kw)
  elif static_rank > 4:  # A batch of images: ...HWC
    input_shape = tf.shape(image_or_images)
    h, w, c = image_or_images.get_shape().as_list()[-3:]
    image_or_images = tf.reshape(image_or_images, [-1, h, w, c])
    image_or_images = tf.map_fn(fn, image_or_images, **map_kw)
    return tf.reshape(image_or_images, input_shape)
  else:
    raise ValueError("Unsupported image rank: %d" % static_rank)


class BatchedPreprocessing(object):
  """Decorator for preprocessing ops, which adds support for image batches."""

  def __init__(self, output_dtype=None, data_key="image"):
    self.output_dtype = output_dtype
    self.data_key = data_key

  def __call__(self, get_pp_fn):

    def get_batch_pp_fn(*args, **kwargs):
      """Preprocessing function that supports batched images."""

      def pp_fn(image):
        return get_pp_fn(*args, **kwargs)({self.data_key: image})[self.data_key]

      def _batch_pp_fn(data):
        image = data[self.data_key]
        data[self.data_key] = tf_apply_to_image_or_images(
            pp_fn, image, dtype=self.output_dtype)
        return data

      return _batch_pp_fn

    return get_batch_pp_fn


@BatchedPreprocessing()
def flip_lr():
  """Flips an image horizontally with probability 50%."""
  def _random_flip_lr_pp(data):
    image = data["image"]
    image = tf.image.random_flip_left_right(image)
    data["image"] = image
    return data
  return _random_flip_lr_pp


@BatchedPreprocessing(output_dtype=tf.float32)
def value_range(vmin=-1, vmax=1, in_min=0, in_max=255.0, clip_values=False):
  """Transforms a [in_min,in_max] image to [vmin,vmax] range.

  Input ranges in_min/in_max can be equal-size lists to rescale the invidudal
  channels independently.

  Args:
    vmin: A scalar. Output max value.
    vmax: A scalar. Output min value.
    in_min: A scalar or a list of input min values to scale. If a list, the
      length should match to the number of channels in the image.
    in_max: A scalar or a list of input max values to scale. If a list, the
      length should match to the number of channels in the image.
    clip_values: Whether to clip the output values to the provided ranges.
  Returns:
    A function to rescale the values.
  """

  def _value_range(data):
    """Scales values in given range."""
    in_min_t = tf.constant(in_min, tf.float32)
    in_max_t = tf.constant(in_max, tf.float32)
    image = tf.cast(data["image"], tf.float32)
    image = (image - in_min_t) / (in_max_t - in_min_t)
    image = vmin + image * (vmax - vmin)
    if clip_values:
      image = tf.clip_by_value(image, vmin, vmax)
    data["image"] = image
    return data
  return _value_range


@BatchedPreprocessing()
def random_crop(crop_size):
  """Makes a random crop of a given size.

  Args:
    crop_size: either an integer H, where H is both the height and width of
      the random crop, or a list or tuple [H, W] of integers, where H and W are
      height and width of the random crop respectively.

  Returns:
    A function, that applies random crop.
  """
  crop_size = maybe_repeat(crop_size, 2)

  def _crop(data):
    image = data["image"]
    h, w, c = crop_size[0], crop_size[1], image.shape[-1]
    image = tf.random_crop(image, [h, w, c])
    data["image"] = image
    return data
  return _crop


@BatchedPreprocessing()
def resize(resize_size):
  """Resizes image to a given size.

  Args:
    resize_size: either an integer H, where H is both the new height and width
      of the resized image, or a list or tuple [H, W] of integers, where H and W
      are new image"s height and width respectively.

  Returns:
    A function for resizing an image.

  """
  resize_size = maybe_repeat(resize_size, 2)

  def _resize(data):
    """Resizes image to a given size."""
    image = data["image"]
    # Note: use TF-2 version of tf.image.resize as the version in TF-1 is
    # buggy: https://github.com/tensorflow/tensorflow/issues/6720.
    dtype = image.dtype
    image = tf2.image.resize(image, resize_size)
    image = tf.cast(image, dtype)
    data["image"] = image
    return data

  return _resize


TPU_SUPPORTED_DTYPES = [
    tf.bool, tf.int32, tf.int64, tf.bfloat16, tf.float32, tf.complex64,
    tf.uint32
]


def get_preprocess_fn(pp_pipeline, remove_tpu_dtypes=True):
  """Transform an input string into the preprocessing function.

  The minilanguage is as follows:

    fn1|fn2(arg, arg2,...)|...

  And describes the successive application of the various `fn`s to the input,
  where each function can optionally have one or more arguments, which are
  either positional or key/value, as dictated by the `fn`.

  The output preprocessing function expects a dictinary as input. This
  dictionary should have a key "image" that corresponds to a 3D tensor
  (height x width x channel).

  Args:
    pp_pipeline: A string describing the pre-processing pipeline.
    remove_tpu_dtypes: Whether to remove TPU incompatible types of data.

  Returns:
    preprocessing function.

  Raises:
    ValueError: if preprocessing function name is unknown
  """

  def _preprocess_fn(data):
    """The preprocessing function that is returned."""

    # Validate input
    if not isinstance(data, dict):
      raise ValueError("Argument `data` must be a dictionary, "
                       "not %s" % str(type(data)))

    # Apply all the individual steps in sequence.
    logging.info("Data before pre-processing:\n%s", data)
    for fn_name in pp_pipeline.split("|"):
      data = eval(fn_name)(data)  # pylint: disable=eval-used

    if remove_tpu_dtypes:
      # Remove data that are TPU-incompatible (e.g. filename of type tf.string).
      for key in list(data.keys()):
        if data[key].dtype not in TPU_SUPPORTED_DTYPES:
          tf.logging.warning(
              "Removing key '{}' from data dict because its dtype {} is not in "
              " the supported dtypes: {}".format(key, data[key].dtype,
                                                 TPU_SUPPORTED_DTYPES))
          del data[key]
    logging.info("Data after pre-processing:\n%s", data)
    return data

  return _preprocess_fn
