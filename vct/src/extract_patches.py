# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Various TF based implementations of patch extraction.

The most high level function here is simply `extract_patches`, which
automatically chooses the most efficient implementation, see docstring.
"""



import math
from typing import Optional
from typing import Tuple
import tensorflow as tf


def _reflect_pad(tensor, target_factor):
  _, height, width, _ = tensor.shape
  height_padded = math.ceil(height / target_factor) * target_factor
  width_padded = math.ceil(width / target_factor) * target_factor
  return tf.pad(
      tensor,
      [[0, 0], [0, height_padded - height], [0, width_padded - width], [0, 0]],
      mode="REFLECT")


def window_partition(features,
                     window_size,
                     pad = True,
                     ):
  """Partition the input feature maps into *non-overlapping* windows.

  Args:
    features: [B, H, W, C] feature maps.
    window_size: The window size.
    pad: If True, will REFLECT pad spatial dims of features such that they
      are divisible by `window_size` before partitioning.

  Returns:
    Partitioned features: [B, nH, nW, wSize*wSize, c] (note that this is a
      different shape from the rest of this file!)

  Raises:
    ValueError: If the feature map sizes are not divisible by window sizes.
  """
  b, h, w, c = features.shape
  if h % window_size != 0 or w % window_size != 0:
    if not pad:
      raise ValueError(f"Feature map sizes {(h, w)} "
                       f"not divisible by window size ({window_size}).")
    features = _reflect_pad(features, window_size)
    assert features is not None  # pytype
    _, h, w, _ = features.shape
  features = tf.reshape(
      features,
      (b, h // window_size, window_size, w // window_size, window_size, c))
  features = tf.einsum("bhiwjc->bhwijc", features)
  features = tf.reshape(
      features, (b, h // window_size, w // window_size, window_size**2, c))
  return features


def unwindow(features,
             window_size,
             unpad = None):
  """Inverse of `window_partition`.

  Args:
    features: Features to unwindow, shape (b, nh, nw, window_size**2, c)
    window_size: Window size.
    unpad: Shape of the latent before it was padded by `window_partition`.

  Returns:
    Features of shape (b, H, W, c).
  """
  b, nh, nw, _, c = features.shape
  features = tf.reshape(features, (b, nh, nw, window_size, window_size, c))
  features = tf.einsum("bhwijc->bhiwjc", features)
  b, nh, _, nw, _, c = features.shape
  features = tf.reshape(features, (b, nh * window_size, nw * window_size, c))
  if unpad:
    orig_h, orig_w = unpad
    return features[:, :orig_h, :orig_w, :]
  else:
    return features


def extract_patches_conv2d(image,
                           size,
                           stride = 1,
                           padding = "SAME"):
  """Extracts patches from an image batch tensor (conv2d implementation).

  NOTE: This implementation uses a kernel with `(size * size * channels)**2`
  elements, and thus may blow up your memory.

  Args:
    image: float tf.Tensor of shape [b, H, W, c], The source to differentiably
      resample from.
    size: The patch size.
    stride: The stride.
    padding: "SAME" or "VALID"

  Returns:
    A [b, n_H, n_W, size*size*c] tensor of patches, where n_H, n_W are the
      number of patches in the spatial dimensions.
  """
  channels = int(image.shape[-1])
  # We make a kernel of size
  # [filter_height, filter_width, in_channels, out_channels]
  # with filter_height=filter_width == `size`, in_channels == `channels` and
  # out_channels == `channels*size*size`
  # NOTE: We used to have a numpy kernel here but that causes XLA compilation
  # to crash when serializing the tensor. tf.eye does not have this problem.
  kernel = tf.reshape(
      tf.eye(size * size * channels, dtype=image.dtype),
      (size, size, channels, channels * size * size))
  return tf.nn.conv2d(image, kernel, strides=stride, padding=padding)


def _has_gpu():
  """Returns true iff a TPU is available on the current machine."""
  return bool(tf.config.list_logical_devices("GPU"))


def extract_patches_nonoverlapping(
    features,
    window_size,
    pad = True,
):
  """Wrapper around `window_partition` that returns same shape as other."""
  # Go from [B, nH, nW, wSize*wSize, c] to [b, n_H, n_W, size*size*c]
  patches = window_partition(features, window_size, pad=pad)
  _, n_h, n_w, seq_len, d = patches.shape
  return tf.reshape(patches, (-1, n_h, n_w, seq_len * d))


def extract_patches(
    image,
    size,
    stride = 1,
):
  """Extract patches of size `size x size` with `stride`.

  Note that this always does a VALID patch extraction. The caller is repsonsible
  for padding if SAME-type extraction is required!

  This function chooses the most efficient implementation based on the
  size/stride and device. On TPU, the tf.image.extract_patches function does
  not have a backward pass implementation, so we use the conv2d-based
  implementation. Sadly, the conv2d-based implementation is very slow on GPU
  for large patch sizes.

  Args:
    image: The image/tensor to patch, size [B, H, W, C].
    size: Size of the patch.
    stride: Stride.

  Returns:
    Tensor of shape [b, n_H, n_W, size*size*c].
  """
  if size == stride:
    # This function is reshape + transpose based and is always the fastest, but
    # of course only works if size == stride.
    return extract_patches_nonoverlapping(image, size, pad=False)
  return extract_patches_conv2d(image, size, stride, padding="VALID")
