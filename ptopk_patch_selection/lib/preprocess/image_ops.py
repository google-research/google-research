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

"""Operations for feature preprocessing.

Each function defines a preprocessing operation.

Images should be either uint8 in [0, 255] or float32 in [0, 1].

The first argument should be 'image' or 'features'.
In case of 'image' the op only gets the 'image' feature (a tensor of shape
[h, w, c]) and should return a transformation of it.
In case of 'feature' the op gets the full feature dictionary
(Dict[str, Tensor]) and should return a modified dictionary. It can
add, remove and modify entries.
"""

import inspect
import sys
from typing import Tuple, Union, Callable, List, Any

import tensorflow.compat.v2 as tf
import tensorflow_addons.image as tfa_image


def all_ops():
  ops = [
      fn for name, fn in inspect.getmembers(sys.modules[__name__])
      if not name.startswith("_") and hasattr(fn, "__name__") and callable(fn)
  ]
  return ops


def _are_all_ints(arr):
  """Check whether all elements in arr are ints."""
  for a in arr:
    if not isinstance(a, int):
      return False
  return True


def _get_image_size(img, dynamic=False):
  """Get width, height for input image."""
  if dynamic:
    size = tf.shape(img)[:2]
    return size[1], size[0]
  else:
    size = img.get_shape().as_list()[:2]
    return size[1], size[0]


def to_float_0_1(image):
  """Convert pixels to tf.float32 and rescale from [0, 255] to [0, 1]."""
  assert image.dtype == tf.uint8, image.dtype
  return tf.cast(image, tf.float32) / 255.0


def value_range(image, vmin, vmax, in_min=0, in_max=1.0, clip_values=False):
  """Transforms a [in_min,in_max] image to [vmin,vmax] range.

  Input ranges in_min/in_max can be equal-size lists to rescale the invidudal
  channels independently.

  Args:
    image: Input image. Will be cast to tf.float32 regardless of input type.
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
  assert vmin < vmax, "vmin {} not less than vmax {}".format(vmin, vmax)
  in_min_t = tf.constant(in_min, tf.float32)
  in_max_t = tf.constant(in_max, tf.float32)
  image = tf.cast(image, tf.float32)
  image = (image - in_min_t) / (in_max_t - in_min_t)
  image = vmin + image * (vmax - vmin)
  if clip_values:
    image = tf.clip_by_value(image, vmin, vmax)
  return image


def resize(features,
           resolution,
           keys = ("image",),
           methods = ("bilinear",)):
  """Resize features to resolution disregarding aspect ratio."""
  if len(methods) != len(keys):
    raise ValueError("Number of keys for resizing must equal methods.")

  for key, method in zip(keys, methods):
    old_dtype = features[key].dtype
    if features[key].shape.ndims == 2:
      squeeze_extra_dim = True
      blob = features[key][:, :, None]
    else:
      squeeze_extra_dim = False
      blob = features[key]

    blob = tf.cast(blob, dtype=tf.float32)
    blob_resized = tf.image.resize(blob, resolution, method=method)
    blob_resized = tf.cast(blob_resized, dtype=old_dtype)

    if squeeze_extra_dim:
      features[key] = blob_resized[:, :, 0]
    else:
      features[key] = blob_resized

  return features


def random_resize(features,
                  scale = (0.5, 2.0),
                  ensure_small = None,
                  keys = ("image",),
                  methods = ("bilinear",)):
  """Randomly resize the image and label by a uniformly sampled scale.

  Args:
      features: Input dictionary containing "image", "label", and other keys.
      scale: Output image and label will be scaled by a scale sampled uniformly
        at random in this range.
      ensure_small: Ignored if None. Else, if input image size * min(scale) is
        less than ensure_small, it will adjust the scale so that the output
        image is always at least as big as ensure_small. This is useful so that
        subsequent crop operations do not go out of range.
      keys: Keys to apply resize op to. Note that keys starting with prefix
        "label" will be resized using nearest neighbour.
      methods: Resize methods per key.

  Returns:
      features with randomly scaled "images" defined by keys.
  """
  if ensure_small is None:
    scale_min, scale_max = scale
  else:
    width, height = _get_image_size(features["image"], dynamic=True)
    scale_min = tf.maximum(ensure_small / tf.cast(width, dtype=tf.float32),
                           ensure_small / tf.cast(height, dtype=tf.float32))
    scale_max = tf.maximum(scale[1], scale_min)

  scale_chosen = tf.random.uniform(
      shape=(), minval=scale_min, maxval=scale_max, dtype=tf.float32)
  width, height = _get_image_size(features["image"], dynamic=True)
  new_width = tf.cast(tf.cast(width, tf.float32) * scale_chosen, tf.int32)
  new_height = tf.cast(tf.cast(height, tf.float32) * scale_chosen, tf.int32)

  return resize(features, (new_height, new_width), keys, methods)


def resize_small(features, size,
                 keys = ("image",),
                 methods = ("bilinear",)):
  """Resizes the image to `size` but preserves the aspect ratio."""
  for key, method in zip(keys, methods):
    image = features[key]
    ndims = image.shape.ndims
    if ndims == 2:
      image = image[:, :, None]
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    ratio = float(size) / tf.math.minimum(height, width)
    new_height = tf.cast(tf.math.ceil(height * ratio), tf.int32)
    new_width = tf.cast(tf.math.ceil(width * ratio), tf.int32)
    features[key] = tf.image.resize(image, [new_height, new_width],
                                    method=method)
    if ndims == 2:
      features[key] = features[key][Ellipsis, 0]
  return features


def _pad_multichannel(image, ensure_small, pad_value, mode):
  """Pad to ensure `ensure_small`."""
  pad_h = tf.maximum(ensure_small[0] - tf.shape(image)[0], 0)
  pad_h_l = pad_h // 2
  pad_h_r = pad_h - pad_h_l

  pad_w = tf.maximum(ensure_small[1] - tf.shape(image)[1], 0)
  pad_w_l = pad_w // 2
  pad_w_r = pad_w - pad_w_l

  def pad_2d(x, v):
    """Pad 2D input `x` with constant value `v`."""
    return tf.pad(
        x, [[pad_h_l, pad_h_r], [pad_w_l, pad_w_r]] + [[0, 0]] *
        (len(x.shape) - 2), mode=mode, constant_values=v)

  if isinstance(pad_value, (list, tuple)):
    image_new = tf.stack(
        [pad_2d(image[:, :, i], v) for i, v in enumerate(pad_value)], axis=2)
  else:
    image_new = pad_2d(image, pad_value)

  return image_new


def pad(features, ensure_small,
        pad_values=(0.,), keys=("image",), mode="CONSTANT"):
  """Pads features to minimum resolution."""
  padding_mode = mode
  if mode == "NOISE":
    assert all(v == 0. for v in pad_values), (
        "pad_values should be 0. when padding mode is NOISE")
    padding_mode = "CONSTANT"

  for k, pad_value in zip(keys, pad_values):
    original_features = features[k]
    features[k] = _pad_multichannel(features[k], ensure_small, pad_value,
                                    padding_mode)

    if mode == "NOISE":
      # Pad the image with Gaussian noise with the same statistics (mean and
      # standard deviation) as the image.
      mask = tf.ones_like(original_features)
      mask = (1. - _pad_multichannel(mask, ensure_small, 0., "CONSTANT"))

      raw_noise = tf.random.normal(tf.shape(mask))
      std = tf.math.reduce_std(original_features, axis=(0, 1), keepdims=True)
      mean = tf.math.reduce_mean(original_features, axis=(0, 1), keepdims=True)
      noise = (raw_noise * std + mean) * mask
      features[k] += noise

  return features


def central_crop(features,
                 crop_size,
                 keys = ("image",)):
  """Center crops given input.

  Args:
    features: Input features dictionary.
    crop_size: Output resolution.
    keys: Fields in features which need to be cropped.

  Returns:
    Cropped features.
  """
  h_c, w_c = crop_size
  for key in keys:
    h, w = tf.unstack(tf.shape(features[key]))[:2]
    h_offset = (h - h_c) // 2
    w_offset = (w - w_c) // 2
    features[key] = features[key][h_offset:h_offset + h_c,
                                  w_offset:w_offset + w_c]
  for key in keys:
    features[key].set_shape([h_c, w_c] + features[key].get_shape()[2:])

  return features


def crop(features,
         rect,
         keys = ("image",)):
  """Crop given input rectangle.

  Rectangle ends are not included in crop.

  Args:
    features: Input features dictionary.
    rect: Rectangle format is [h_offset, w_offset, height, width].
    keys: Fields in features which need to be cropped.

  Returns:
    Cropped features.
  """
  for key in keys:
    features[key] = features[key][rect[0]:rect[0] + rect[2],
                                  rect[1]:rect[1] + rect[3]]

  if _are_all_ints(rect):
    for key in keys:
      features[key].set_shape([rect[2], rect[3]] +
                              features[key].get_shape()[2:])
  return features


def _get_random_crop_rectangle(image_shape, resolution):
  """Given image shape and desired crop resolution sample a crop rectangle.

  Rectangle format is [h_offset, w_offset, height, width].

  Args:
    image_shape: tf.shape(image) output.
    resolution: height x width target crop resolution.  Fails when the crop is
      too big for the given image. No checks are performed.

  Returns:
    rect: Tuple[int, int, int, int].
  """
  h = image_shape[0]
  w = image_shape[1]
  h_offset = tf.random.uniform((),
                               minval=0,
                               maxval=h - resolution[0] + 1,
                               dtype=tf.int32)
  w_offset = tf.random.uniform((),
                               minval=0,
                               maxval=w - resolution[1] + 1,
                               dtype=tf.int32)
  rect = (h_offset, w_offset, resolution[0], resolution[1])
  return rect


def random_crop(features,
                resolution,
                keys = ("image",)):
  """Random crop."""
  rect = _get_random_crop_rectangle(tf.shape(features[keys[0]]), resolution)
  features = crop(features, rect, keys=keys)
  if _are_all_ints(resolution):
    for key in keys:
      features[key].set_shape([resolution[0], resolution[1]] +
                              features[key].get_shape()[2:])

  return features


def random_size_crop(features,
                     resolution_min,
                     resolution_max = None,
                     keys = ("image",)):
  """Crop of random size with minimum and maximum resolution.

  Args:
    features: Input features that must include specified keys.
    resolution_min: Minimum resolution of the crop.
    resolution_max: Maximum resolution of the crop. Defaults to inputs shape.
    keys: On which keys to apply this function.
  Returns:
    Transformed features.
  """
  image_shape = tf.unstack(tf.shape(features[keys[0]]))

  if not resolution_max:
    resolution_max = (image_shape[0], image_shape[1])

  h = tf.random.uniform([], tf.minimum(resolution_min[0], resolution_max[0]),
                        resolution_max[0] + 1, dtype=tf.int32)
  w = tf.random.uniform([], tf.minimum(resolution_min[1], resolution_max[1]),
                        resolution_max[1] + 1, dtype=tf.int32)
  resolution = (h, w)
  rect = _get_random_crop_rectangle(image_shape, resolution)
  features = crop(features, rect, keys=keys)

  return features


def random_left_right_flip(features, keys = ("image",)):
  """Randomly left-right flip feature fields with 50% probability."""
  stride = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32) * 2 - 1
  for key in keys:
    old_shape = features[key].get_shape().as_list()
    features[key] = features[key][:, ::stride]
    features[key].set_shape(old_shape)
  return features


def random_rotate(features, angle_range, keys=("image",)):
  """Randomly rotates all features with defined keys."""
  angle = tf.random.uniform([], *angle_range, dtype=tf.float32)
  for k in keys:
    features[k] = tfa_image.rotate(features[k], angle)
  return features


def _gauss_filter(kernel_size, sigma):
  """Creates Gaussian filter."""
  x = tf.range(-kernel_size // 2, kernel_size // 2 + 1, 1, dtype=tf.float32)
  if not sigma:
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
  x = tf.exp(x**2 / (-2 * sigma**2))
  x /= tf.reduce_sum(x, axis=-1, keepdims=True)
  return x


def gaussian_blur(image, kernel_size,
                  sigma = None):
  """Puts blur on the image."""
  fil_x = _gauss_filter(kernel_size[0], sigma=sigma[0] if sigma else None)
  fil_y = _gauss_filter(kernel_size[1], sigma=sigma[1] if sigma else None)
  fil = fil_x[:, None, None, None] * fil_y[None, :, None, None]
  fil = tf.tile(fil, [1, 1, image.shape[-1], 1])
  res = tf.nn.depthwise_conv2d(
      image[None], fil, strides=[1, 1, 1, 1], padding="SAME")
  res = tf.squeeze(res, 0)
  return res


def label_map(features,
              source_labels,
              target_labels = None,
              default_label = 0,
              keys = ("label",)):
  """Label mapping.

  Args:
    features: Dictionary of data features to preprocess.
    source_labels: Tuple of source labels.
    target_labels: Tuple of target labels, aligned with source_labels. If not
                   set, source_labels is used for target_labels.
    default_label: Labels that aren't in source_labels are mapped to this label.
    keys: On which keys to apply this function.

  Returns:
    Features with mapped labels.
  """
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          source_labels,
          source_labels if target_labels is None else target_labels),
      default_label)
  for key in keys:
    features[key] = table.lookup(features[key])
  return features


def binarize(features,
             threshold = 0,
             keys = ("image",)):
  """Binarizes a (grayscale, uint8) image.

  Args:
    features: Dictionary of data features to preprocess.
    threshold: Threshold to distinguish background from foreground.
    keys: On which keys to apply this function.

  Returns:
    Features with the binarized image.
  """

  for key in keys:
    image = features[key]
    assert image.dtype == tf.uint8
    # assert (image.shape.ndims == 3 and image.shape[-1] == 1 or
    #         image.shape.ndims == 2)
    binarized = tf.cast(image > tf.cast(threshold, tf.uint8), tf.uint8) * 255
    features[key] = binarized
  return features


# Some operations on point clouds.
def binary_image_to_points(features,
                           normalize_coords = True,
                           keys = ("image",)):
  """Converts a (binary) image into a 2D point cloud.

  Args:
    features: Dictionary of data features to preprocess.
    normalize_coords: Normalize coords to be in [0,1] by preserving the aspect
                      ratio.
    keys: On which keys to apply this function.

  Returns:
    Features with the image as a point cloud.
  """

  for key in keys:
    image = features[key]  # [HxW] or [HxWx1]
    image = tf.reshape(image, [image.shape[0], image.shape[1], 1])
    # We map background pixels to the origin, which may be suboptimal
    # but saves us some engineering work.
    coords = tf.cast(
        tf.stack(tf.meshgrid(tf.range(image.shape[0]), tf.range(image.shape[1]),
                             indexing="ij"), axis=-1),
        tf.float32)
    if normalize_coords:
      coords /= tf.cast(tf.reduce_max(image.shape[:2]), tf.float32)
    mask = tf.tile(image > 0, [1, 1, 2])
    features[key] = tf.reshape(tf.cast(mask, tf.float32) * coords, [-1, 2])
  return features


def points_shuffle(features,
                   keys = ("image",)):
  """Shuffle points.

  Args:
    features: Dictionary of data features to preprocess.
    keys: On which keys to apply this function.

  Returns:
    Features with shuffled points.
  """
  for key in keys:
    features[key] = tf.random.shuffle(features[key])
  return features


def points_padded_to_end(features,
                         keys = ("image",)):
  """Move all padded points to the end.

  Args:
    features: Dictionary of data features to preprocess.
    keys: On which keys to apply this function.

  Returns:
    Features with padded points at the end.
  """
  for key in keys:
    features[key] = _points_sort(
        features[key],
        key=lambda x: tf.cast(tf.reduce_sum(x, axis=-1) == 0.0, tf.float32),
        stable=True)
  return features


def _points_sort(points, key, stable=False):
  indices = tf.argsort(key(points), stable=stable)
  return tf.gather(points, indices)


def points_select_first_n(features,
                          num_points,
                          keys = ("image",)):
  """Resize the number of points.

  Args:
    features: Dictionary of data features to preprocess.
    num_points: The target number of points.
    keys: On which keys to apply this function.

  Returns:
    Features with resized number of points.
  """
  for key in keys:
    features[key] = tf.slice(features[key], begin=(0, 0), size=(num_points, -1))
  return features




def points_scale(features,
                 stddev,
                 keys = ("image",)):
  """Randomly scale points from the origin.

  Args:
    features: Dictionary of data features to preprocess.
    stddev: The stddev of the scale.
    keys: On which keys to apply this function.

  Returns:
    Features with scaled points.
  """
  for key in keys:
    features[key] = features[key] * (1.0 + tf.random.truncated_normal(
        shape=(1, 3), mean=0.0, stddev=stddev))
  return features


def points_translate(features,
                     stddev,
                     keys = ("image",)):
  """Randomly translate points.

  Args:
    features: Dictionary of data features to preprocess.
    stddev: The stddev of the translation.
    keys: On which keys to apply this function.

  Returns:
    Features with translated points.
  """
  for key in keys:
    features[key] = features[key] + tf.random.truncated_normal(
        shape=(1, 3), mean=0.0, stddev=stddev)
  return features


def points_rotate(features,
                  max_rotation,
                  min_rotation = 0.0,
                  axis = "z",
                  keys = ("image",)):
  """Randomly rotate points on a given axis.

  Args:
    features: Dictionary of data features to preprocess.
    max_rotation: The maximum possible rotation in radians.
    min_rotation: The minimum possible rotation in radians.
    axis: The rotation axis.
    keys: On which keys to apply this function.

  Returns:
    Features with rotated points.
  """
  assert axis in {"x", "y", "z"}, "invalid rotation axis"

  for key in keys:
    phi = tf.random.uniform(
        shape=(1,), minval=min_rotation, maxval=max_rotation)
    cos, sin, zero, one = (tf.cos(phi), tf.sin(phi), tf.zeros((1,)),
                           tf.ones((1,)))
    # Matrices from
    # https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations.
    if axis == "x":
      rotation_matrix = [one, zero, zero, zero, cos, -sin, zero, sin, cos]
    elif axis == "y":
      rotation_matrix = [cos, zero, sin, zero, one, zero, -sin, zero, cos]
    elif axis == "z":
      rotation_matrix = [cos, -sin, zero, sin, cos, zero, zero, zero, one]
    rotate = tf.reshape(tf.stack(rotation_matrix, axis=0), [3, 3])
    features[key] = tf.matmul(features[key], rotate)

  return features




def random_linear_transform(image,
                            a_bounds,
                            b_bounds,
                            p):
  """Random linear augmentation to compute a * image + b with probability p.

  Based on
  https://github.com/idiap/attention-sampling/blob/504b1733869e18005d099ec04f7cbd1793043d67/ats/utils/layers.py#L125

  Args:
    image: The image to transform.
    a_bounds: Lower and upper bounds of a.
    b_bounds: Lower and upper bounds of b.
    p: Probability to apply the transform, otherwise keep image identical.

  Returns:
    The transformed image.
  """
  assert image.dtype == tf.float32, image.dtype
  shape = (1, 1, 1)
  indicator = tf.random.uniform(shape=shape, minval=0, maxval=1) < p
  indicator = tf.cast(indicator, image.dtype)
  a = tf.random.uniform(shape=shape, minval=a_bounds[0], maxval=a_bounds[1])
  b = tf.random.uniform(shape=shape, minval=b_bounds[0], maxval=b_bounds[1])
  a = indicator * a + (1 - indicator)
  b = indicator * b
  return a * image + b


def normalize(image,
              mu,
              sigma):
  assert image.dtype == tf.float32, image.dtype
  mu = tf.constant(mu)[None, None, :]
  sigma = tf.constant(sigma)[None, None, :]
  return (image - mu) / sigma

