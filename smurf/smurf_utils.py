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

"""SMURF utils.

This library contains the various util functions used in SMURF.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import gin
import tensorflow as tf
from tensorflow_addons import image as tfa_image


# Custom typing abbreviations.
_FlowKey = Tuple[int, int, str]
_FlowKeyDict = Dict[_FlowKey, List[tf.Tensor]]

# TFA is significantly faster on GPU but can cause compatibility problems.
USE_TFA = False


def resampler_flat_gather(data, warp, name='flat_resampler'):
  """Resampler that avoids gather_nd which can be expensive on TPU.

  Computing gradients of gather_nd requires calling scatter_nd
  which is very slow on TPU and causes a large memory blowup.
  Empirically, this resampler produces a much lower memory footprint
  and faster inference time on the TPU by avoding gather_nd and instead
  using a flat gather. See tfa.image.resampler for more documentation.

  Args:
    data: float tf Tensor of shape b H W c, The source to differentiably
      resample from.
    warp: float tf Tensor of shape b h w 2, The set of coordinates to sample
      from data.
    name: str scope to put operations under.
  Returns:
    resampled_data: float tf Tensor of shape b h w c, The result of sampling
      data with warp.
  """
  with tf.name_scope(name):
    b, data_h, data_w, c = tf.unstack(tf.shape(data))
    _, warp_h, warp_w, _ = tf.unstack(tf.shape(warp))
    warp_x, warp_y = tf.unstack(warp, axis=-1)

    warp_shape = tf.shape(warp_x)
    warp_batch = tf.range(warp_shape[0], dtype=tf.int32)
    warp_batch = tf.reshape(warp_batch, (warp_shape[0], 1, 1))
    warp_batch = tf.broadcast_to(warp_batch, (b, warp_h, warp_w))
    warp_batch = tf.reshape(warp_batch, [-1])
    warp_x = tf.reshape(warp_x, [-1])
    warp_y = tf.reshape(warp_y, [-1])
    warp_floor_x = tf.math.floor(warp_x)
    warp_floor_y = tf.math.floor(warp_y)

    right_warp_weight = warp_x - warp_floor_x
    down_warp_weight = warp_y - warp_floor_y
    left_warp_weight = tf.subtract(
        tf.convert_to_tensor(1.0, right_warp_weight.dtype), right_warp_weight)
    up_warp_weight = tf.subtract(
        tf.convert_to_tensor(1.0, down_warp_weight.dtype), down_warp_weight)

    warp_floor_x = tf.cast(warp_floor_x, tf.int32)
    warp_floor_y = tf.cast(warp_floor_y, tf.int32)
    warp_ceil_x = tf.cast(tf.math.ceil(warp_x), tf.int32)
    warp_ceil_y = tf.cast(tf.math.ceil(warp_y), tf.int32)

    left_warp_weight = tf.expand_dims(left_warp_weight, -1)
    right_warp_weight = tf.expand_dims(right_warp_weight, -1)
    up_warp_weight = tf.expand_dims(up_warp_weight, -1)
    down_warp_weight = tf.expand_dims(down_warp_weight, -1)

    def flatten_warp(warp_y, warp_x):
      """Converts the warps from a 2D index to a 1D index."""
      output = tf.reshape(
          warp_batch * data_w * data_h + warp_y * data_w + warp_x, [-1])
      # Get a mask of the coordinates which go out of bounds.
      mask_y = tf.cast(
          tf.logical_and(warp_y >= 0, warp_y <= data_h - 1), dtype=data.dtype)
      mask_x = tf.cast(
          tf.logical_and(warp_x >= 0, warp_x <= data_w - 1), dtype=data.dtype)
      output = tf.clip_by_value(output, 0, b * data_h * data_w - 1)
      return output, tf.expand_dims(mask_y * mask_x, -1)

    up_left_warp, mask_up_left = flatten_warp(warp_floor_y, warp_floor_x)
    up_right_warp, mask_up_right = flatten_warp(warp_floor_y, warp_ceil_x)
    down_left_warp, mask_down_left = flatten_warp(warp_ceil_y, warp_floor_x)
    down_right_warp, mask_down_right = flatten_warp(warp_ceil_y, warp_ceil_x)
    flat_data = tf.reshape(data, (-1, c))

    up_left = tf.gather(flat_data, up_left_warp, axis=0) * mask_up_left
    up_right = tf.gather(flat_data, up_right_warp, axis=0) * mask_up_right
    down_left = tf.gather(flat_data, down_left_warp, axis=0) * mask_down_left
    down_right = tf.gather(flat_data, down_right_warp, axis=0) * mask_down_right
    result = (up_left * left_warp_weight + up_right * right_warp_weight
             ) * up_warp_weight + (down_left * left_warp_weight + down_right *
                                   right_warp_weight) * down_warp_weight
    return tf.reshape(result, (b, warp_h, warp_w, c))


def resampler(source, coords):
  if USE_TFA:
    return tfa_image.resampler(source, coords)
  else:
    return resampler_flat_gather(source, coords)


def flow_to_warp(flow):
  """Compute the warp from the flow field.

  Args:
    flow: tf.tensor representing optical flow.

  Returns:
    The warp, i.e. the endpoints of the estimated flow.
  """

  # Construct a grid of the image coordinates.
  height, width = flow.shape.as_list()[-3:-1]
  i_grid, j_grid = tf.meshgrid(
      tf.linspace(0.0, height - 1.0, int(height)),
      tf.linspace(0.0, width - 1.0, int(width)),
      indexing='ij')
  grid = tf.stack([i_grid, j_grid], axis=2)

  # Potentially add batch dimension to match the shape of flow.
  if len(flow.shape) == 4:
    grid = grid[None]

  # Add the flow field to the image grid.
  if flow.dtype != grid.dtype:
    grid = tf.cast(grid, flow.dtype)
  warp = grid + flow
  return warp


def mask_invalid(coords, pad_h=0, pad_w=0):
  """Mask coordinates outside of the image.

  Valid = 1, invalid = 0.

  Args:
    coords: a 4D float tensor of image coordinates.
    pad_h: int, the amount of padding applied to the top of the image
    pad_w: int, the amount of padding applied to the left of the image

  Returns:
    The mask showing which coordinates are valid.
  """
  pad_h = float(pad_h)
  pad_w = float(pad_w)
  coords_rank = len(coords.shape)
  if coords_rank != 4:
    raise NotImplementedError()
  max_height = float(coords.shape[-3] - 1)
  max_width = float(coords.shape[-2] - 1)
  mask = tf.logical_and(
      tf.logical_and(coords[:, :, :, 0] >= pad_h,
                     coords[:, :, :, 0] <= max_height),
      tf.logical_and(coords[:, :, :, 1] >= pad_w,
                     coords[:, :, :, 1] <= max_width))
  mask = tf.cast(mask, dtype=tf.float32)[:, :, :, None]
  return mask


def resample(source, coords):
  """Resample the source image at the passed coordinates.

  Args:
    source: tf.tensor, batch of images to be resampled.
    coords: tf.tensor, batch of coordinates in the image. Coordinates should
      be between 0 and size - 1. Coordinates outside of this range are handled
      by interpolating with a background image filled with zeros in the same
      way that SAME size convolution works.

  Returns:
    The resampled image.
  """

  # Wrap this function because it uses a different order of height/width dims.
  orig_source_dtype = source.dtype
  if source.dtype != tf.float32:
    source = tf.cast(source, tf.float32)
  if coords.dtype != tf.float32:
    coords = tf.cast(coords, tf.float32)
  coords_rank = len(coords.shape)
  if coords_rank == 4:
    output = resampler(source, coords[:, :, :, ::-1])
    if orig_source_dtype != source.dtype:
      return tf.cast(output, orig_source_dtype)
    return output
  else:
    raise NotImplementedError()


def compute_range_map(flow,
                      downsampling_factor=1,
                      reduce_downsampling_bias=True,
                      resize_output=True):
  """Count how often each coordinate is sampled.

  Counts are assigned to the integer coordinates around the sampled coordinates
  using weights from bilinear interpolation.

  Args:
    flow: A float tensor of shape (batch size x height x width x 2) that
      represents a dense flow field.
    downsampling_factor: An integer, by which factor to downsample the output
      resolution relative to the input resolution. Downsampling increases the
      bin size but decreases the resolution of the output. The output is
      normalized such that zero flow input will produce a constant ones output.
    reduce_downsampling_bias: A boolean, whether to reduce the downsampling bias
      near the image boundaries by padding the flow field.
    resize_output: A boolean, whether to resize the output ot the input
      resolution.

  Returns:
    A float tensor of shape [batch_size, height, width, 1] that denotes how
    often each pixel is sampled.
  """

  # Get input shape.
  input_shape = flow.shape.as_list()
  if len(input_shape) != 4:
    raise NotImplementedError()
  batch_size, input_height, input_width, _ = input_shape

  flow_height = input_height
  flow_width = input_width

  # Apply downsampling (and move the coordinate frame appropriately).
  output_height = input_height // downsampling_factor
  output_width = input_width // downsampling_factor
  if downsampling_factor > 1:
    # Reduce the bias that comes from downsampling, where pixels at the edge
    # will get lower counts that pixels in the middle of the image, by padding
    # the flow field.
    if reduce_downsampling_bias:
      p = downsampling_factor // 2
      flow_height += 2 * p
      flow_width += 2 * p
      # Apply padding in multiple steps to padd with the values on the edge.
      for _ in range(p):
        flow = tf.pad(
            tensor=flow,
            paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],
            mode='SYMMETRIC')
      coords = flow_to_warp(flow) - p
    # Update the coordinate frame to the downsampled one.
    coords = (coords + (1 - downsampling_factor) * 0.5) / downsampling_factor
  elif downsampling_factor == 1:
    coords = flow_to_warp(flow)
  else:
    raise ValueError('downsampling_factor must be an integer >= 1.')

  # Split coordinates into an integer part and a float offset for interpolation.
  coords_floor = tf.floor(coords)
  coords_offset = coords - coords_floor
  coords_floor = tf.cast(coords_floor, 'int32')

  # Define a batch offset for flattened indexes into all pixels.
  batch_range = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
  idx_batch_offset = tf.tile(
      batch_range, [1, flow_height, flow_width]) * output_height * output_width

  # Flatten everything.
  coords_floor_flattened = tf.reshape(coords_floor, [-1, 2])
  coords_offset_flattened = tf.reshape(coords_offset, [-1, 2])
  idx_batch_offset_flattened = tf.reshape(idx_batch_offset, [-1])

  # Initialize results.
  idxs_list = []
  weights_list = []

  # Loop over differences di and dj to the four neighboring pixels.
  for di in range(2):
    for dj in range(2):

      # Compute the neighboring pixel coordinates.
      idxs_i = coords_floor_flattened[:, 0] + di
      idxs_j = coords_floor_flattened[:, 1] + dj
      # Compute the flat index into all pixels.
      idxs = idx_batch_offset_flattened + idxs_i * output_width + idxs_j

      # Only count valid pixels.
      mask = tf.reshape(
          tf.compat.v1.where(
              tf.logical_and(
                  tf.logical_and(idxs_i >= 0, idxs_i < output_height),
                  tf.logical_and(idxs_j >= 0, idxs_j < output_width))), [-1])
      valid_idxs = tf.gather(idxs, mask)
      valid_offsets = tf.gather(coords_offset_flattened, mask)

      # Compute weights according to bilinear interpolation.
      weights_i = (1. - di) - (-1)**di * valid_offsets[:, 0]
      weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 1]
      weights = weights_i * weights_j

      # Append indices and weights to the corresponding list.
      idxs_list.append(valid_idxs)
      weights_list.append(weights)

  # Concatenate everything.
  idxs = tf.concat(idxs_list, axis=0)
  weights = tf.concat(weights_list, axis=0)

  # Sum up weights for each pixel and reshape the result.
  counts = tf.math.unsorted_segment_sum(
      weights, idxs, batch_size * output_height * output_width)
  count_image = tf.reshape(counts, [batch_size, output_height, output_width, 1])

  if downsampling_factor > 1:
    # Normalize the count image so that downsampling does not affect the counts.
    count_image /= downsampling_factor**2
    if resize_output:
      count_image = resize(
          count_image, input_height, input_width, is_flow=False)

  return count_image


@tf.function
def resize(img, height, width, is_flow, mask=None):
  """Resize an image or flow field to a new resolution.

  In case a mask (per pixel {0,1} flag) is passed a weighted resizing is
  performed to account for missing flow entries in the sparse flow field. The
  weighting is based on the resized mask, which determines the 'amount of valid
  flow vectors' that contributed to each individual resized flow vector. Hence,
  multiplying by the reciprocal cancels out the effect of considering non valid
  flow vectors.

  Args:
    img: tf.tensor, image or flow field to be resized of shape [b, h, w, c]
    height: int, heigh of new resolution
    width: int, width of new resolution
    is_flow: bool, flag for scaling flow accordingly
    mask: tf.tensor, mask (optional) per pixel {0,1} flag

  Returns:
    Resized and potentially scaled image or flow field (and mask).
  """
  def _resize(image, mask=None):
    # _, orig_height, orig_width, _ = img.shape.as_list()
    orig_height = tf.shape(input=image)[1]
    orig_width = tf.shape(input=image)[2]

    if mask is not None:
      # multiply with mask, to ensure non-valid locations are zero
      image = tf.math.multiply(image, mask)
      # resize image
      img_resized = tf.compat.v2.image.resize(
          image, (int(height), int(width)), antialias=True)
      # resize mask (will serve as normalization weights)
      mask_resized = tf.compat.v2.image.resize(
          mask, (int(height), int(width)), antialias=True)
      # normalize sparse flow field and mask
      img_resized = tf.math.multiply(
          img_resized, tf.math.reciprocal_no_nan(mask_resized))
      mask_resized = tf.math.multiply(
          mask_resized, tf.math.reciprocal_no_nan(mask_resized))
    else:
      # normal resize without anti-alaising
      img_resized = tf.compat.v2.image.resize(image, (tf.cast(height,
                                                              tf.int32),
                                                      tf.cast(width,
                                                              tf.int32)))

    if is_flow:
      # If image is a flow image, scale flow values to be consistent with the
      # new image size.
      scaling = tf.reshape([
          float(height) / tf.cast(orig_height, tf.float32),
          float(width) / tf.cast(orig_width, tf.float32)
      ], [1, 1, 1, 2])
      img_resized *= scaling

    if mask is not None:
      return img_resized, mask_resized
    return img_resized

  # Apply resizing at the right shape.
  shape = img.shape.as_list()
  if img.shape.rank == 3:
    if mask is not None:
      img_resized, mask_resized = _resize(img[None], mask[None])
      return img_resized[0], mask_resized[0]
    else:
      return _resize(img[None])[0]
  if img.shape.rank == 4:
    # Input at the right shape.
    return _resize(img, mask)
  if img.shape.rank > 4:
    # Reshape input to [b, h, w, c], resize and reshape back.
    outer_shape = tf.shape(input=img)[:-3]
    required_shape = tf.concat([[-1], tf.shape(input=img)[-3:]], axis=0)
    img_flattened = tf.reshape(img, required_shape)
    if mask is not None:
      mask_flattened = tf.reshape(mask, required_shape)
      img_resized, mask_resized = _resize(img_flattened, mask_flattened)
    else:
      img_resized = _resize(img_flattened)
    final_shape = tf.concat(
        [outer_shape, tf.shape(input=img_resized)[-3:]], axis=0)
    result_img = tf.reshape(img_resized, final_shape)
    if mask is not None:
      final_mask_shape = tf.concat(
          [outer_shape, tf.shape(input=mask_resized)[-3:]], axis=0)
      result_mask = tf.reshape(mask_resized, final_mask_shape)
      return result_img, result_mask
    return result_img
  else:
    raise ValueError('Cannot resize an image of shape', shape)


def robust_l1(x):
  """Robust L1 metric."""
  return (x**2 + 0.001**2)**0.5


def abs_robust_loss(diff, eps=0.01, q=0.4):
  """The so-called robust loss used by DDFlow."""
  return tf.pow((tf.abs(diff) + eps), q)


def l2(x):
  return tf.norm(x, ord='euclidean', axis=-1, keepdims=True)


def first_order_smoothness_loss(
    image, flow,
    edge_weighting_fn):
  """Computes a first-order smoothness loss.

  Args:
    image: Image used for the edge-aware weighting [batch, height, width, 2].
    flow: Flow field for with to compute the smoothness loss [batch, height,
      width, 2].
    edge_weighting_fn: Function used for the edge-aware weighting.

  Returns:
    Average first-order smoothness loss.
  """
  img_gx, img_gy = image_grads(image)
  weights_x = edge_weighting_fn(img_gx)
  weights_y = edge_weighting_fn(img_gy)

  # Compute second derivatives of the predicted smoothness.
  flow_gx, flow_gy = image_grads(flow)

  # Compute weighted smoothness
  return ((tf.reduce_mean(input_tensor=weights_x * robust_l1(flow_gx)) +
           tf.reduce_mean(input_tensor=weights_y * robust_l1(flow_gy))) / 2.)


def second_order_smoothness_loss(
    image, flow,
    edge_weighting_fn):
  """Computes a second-order smoothness loss.

  Computes a second-order smoothness loss (only considering the non-mixed
  partial derivatives).

  Args:
    image: Image used for the edge-aware weighting [batch, height, width, 2].
    flow: Flow field for with to compute the smoothness loss [batch, height,
      width, 2].
    edge_weighting_fn: Function used for the edge-aware weighting.

  Returns:
    Average second-order smoothness loss.
  """
  img_gx, img_gy = image_grads(image, stride=2)
  weights_xx = edge_weighting_fn(img_gx)
  weights_yy = edge_weighting_fn(img_gy)

  # Compute second derivatives of the predicted smoothness.
  flow_gx, flow_gy = image_grads(flow)
  flow_gxx, _ = image_grads(flow_gx)
  _, flow_gyy = image_grads(flow_gy)

  # Compute weighted smoothness
  return ((tf.reduce_mean(input_tensor=weights_xx * robust_l1(flow_gxx)) +
           tf.reduce_mean(input_tensor=weights_yy * robust_l1(flow_gyy))) / 2.)


def image_grads(image_batch, stride=1):
  image_batch_gh = image_batch[:, stride:] - image_batch[:, :-stride]
  image_batch_gw = image_batch[:, :, stride:] - image_batch[:, :, :-stride]
  return image_batch_gh, image_batch_gw


def image_averages(image_batch):
  image_batch_ah = (image_batch[:, 1:] + image_batch[:, :-1]) / 2.
  image_batch_aw = (image_batch[:, :, 1:] + image_batch[:, :, :-1]) / 2
  return image_batch_ah, image_batch_aw


def compute_occlusions_brox(forward_flow,
                            backward_flow):
  """Compute an occlusion mask based on a forward-backward check.

  Args:
    forward_flow: Forward flow field of shape [batch, height, width, 2].
    backward_flow: Backward flow field of shape [batch, height, width, 2].

  Returns:
    Occlusion mask of shape [batch, height, width, 1], where 1 are occluded
    locations and 0 are non-occluded.
  """
  # Resampled backward flow at forward flow locations.
  warp = flow_to_warp(forward_flow)
  backward_flow_resampled = resample(backward_flow, warp)

  # Compute occlusions based on forward-backward consistency.
  fb_sq_diff = tf.reduce_sum(
      (forward_flow + backward_flow_resampled)**2, axis=-1, keepdims=True)
  fb_sum_sq = tf.reduce_sum(
      forward_flow**2 + backward_flow_resampled**2, axis=-1, keepdims=True)
  return tf.cast(fb_sq_diff > 0.01 * fb_sum_sq + 0.5, tf.float32)


def compute_occlusions_wang(backward_flow, downsampling_factor,
                            threshold):
  """Compute occlusion mask based on a rangemap.

  Args:
    backward_flow: Backward flow field of shape [batch, height, width, 2].
    downsampling_factor: Downsampling factor used for the range map computation.
    threshold: Indicates if thresholding should be used

  Returns:
    Occlusion mask of shape [batch, height, width, 1], where 1 are occluded
    locations and 0 are non-occluded.
  """
  range_map = compute_range_map(
      backward_flow,
      downsampling_factor=downsampling_factor,
      reduce_downsampling_bias=False,
      resize_output=False)
  if threshold:
    return 1.0 - tf.cast(range_map < 0.75, tf.float32)
  else:
    return 1.0 - tf.clip_by_value(range_map, 0.0, 1.0)


@tf.function
def compute_occlusions(forward_flow,
                       backward_flow,
                       occlusion_estimation = None,
                       occlusions_are_zeros = True,
                       occ_active = None,
                       boundaries_occluded = True):
  """Compute occlusion masks.

  Args:
    forward_flow: Forward flow field of shape [batch, height, width, 2].
    backward_flow: Backward flow field of shape [batch, height, width, 2].
    occlusion_estimation: Type of occlusion estimation that should be used.
    occlusions_are_zeros: Indicates if occlusions are indicated via 0 or 1.
    occ_active: Bool for each possible occlusion estimation type, indicating if
      occlusion estimation is active already or not.
    boundaries_occluded: If True, treat flow vectors pointing off the boundaries
      as occluded. Otherwise explicitly mark them as unoccluded.

  Returns:
    Occlusion mask of shape [batch, height, width, 1].
  """

  # Corresponding forward and backward flow.
  flow_ij = forward_flow
  flow_ji = backward_flow

  occlusion_mask = tf.zeros_like(flow_ij[Ellipsis, :1], dtype=tf.float32)

  if occlusion_estimation == 'none' or (occ_active is not None and
                                        not occ_active[occlusion_estimation]):
    occlusion_mask = tf.zeros_like(flow_ij[Ellipsis, :1], dtype=tf.float32)
  elif occlusion_estimation == 'brox':
    occlusion_mask = compute_occlusions_brox(flow_ij, flow_ji)
  elif occlusion_estimation == 'wang':
    occlusion_mask = compute_occlusions_wang(
        flow_ji, downsampling_factor=1, threshold=False)
  else:
    raise ValueError('Unknown value for occlusion_estimation:',
                     occlusion_estimation)

  if not boundaries_occluded:
    warp = flow_to_warp(flow_ij)
    occlusion_mask = tf.minimum(occlusion_mask, mask_invalid(warp))

  return 1. - occlusion_mask if occlusions_are_zeros else occlusion_mask


def unsupervised_loss(images,
                      flows,
                      weights,
                      occlusion_estimation_fn,
                      only_forward = False,
                      selfsup_transform_fn = None,
                      fb_sigma_teacher = 0.003,
                      fb_sigma_student = 0.03,
                      smoothness_edge_weighting = 'gaussian',
                      smoothness_edge_constant = 150.0,
                      stop_gradient_mask = True,
                      selfsup_mask = 'gaussian',
                      smoothness_at_level = 2,
                      full_size_images = None,
                      crop_h = None,
                      crop_w = None,
                      pad_h = None,
                      pad_w = None):
  """Computes unsupervised SMURF losses.

  Args:
    images: (Unaugmented) images for which the flow fields are passed of shape
      [batch, time, height, width, channels].
    flows: Dictionary of flow fields. The key is given by (i, j, t), where i =
      reference image index, j = second image index, t = augementation/model
      type (e.g. augmented-student).
    weights: Dictionary holding the weights for the different loss functions. If
      a weight is not in the dictionary the loss will not be computed.
    occlusion_estimation_fn: Function to compute occlusions masks.
    only_forward: Flag indicating if only losses for the forward flow estimation
      should be computed.
    selfsup_transform_fn: List of self-supervion transform functions.
    fb_sigma_teacher: Sigma used for the gaussian self-supervision masking.
    fb_sigma_student: Sigma used for the gaussian self-supervision masking.
    smoothness_edge_weighting: Defines which function should be used for the
      edge-aware smoothing, can be either gaussian or exponential.
    smoothness_edge_constant: Constant used within the edge weighting function.
    stop_gradient_mask: Flag indicating if gradients should be stopped for the
      occlusion masks.
    selfsup_mask: Indicates what type of masking to use for the self-supervision
      can be either gaussian or ddflow.
    smoothness_at_level: Resolution level at which the smoothness loss should be
      applied.
    full_size_images: Optional uncropped images to use for warping. If uncropped
      images, crop_h, and crop_w are provided, we will use the full scale images
      to perform a warp. This has the benefit of allowing a loss to be computed
      for many flow vectors which move off the edge of the image.
    crop_h: Optional upper left row of the bounding box used to crop the images
      from the full_size_images.
    crop_w: Optional upper left col of the bounding box used to crop the images
      from the full_size_images.
    pad_h: Optional upper padding applied to the full_size_images. The padding
      was applied after the image was cropped and is not reflected in crop_h.
    pad_w: Optional left padding applied to the full_size_images. The padding
      was applied after the image was cropped and is not reflected in crop_w.

  Returns:
    Dictionary holding the calculated losses.
  """
  # Initialize unsupervised losses with zero.
  losses = {}
  for key in weights:
    losses[key] = tf.constant(0.)

  compute_loss_for_these_flows = ['augmented-student']
  # Count number of non self-sup pairs, for which we will apply the losses.
  num_pairs = sum(
      [1.0 for (i, j, c) in flows if c in compute_loss_for_these_flows])

  # Ensure that smoothness_at_level is feasible, i.e. set smoothness_at_level
  # to be as close as possible to the chosen parameter. This means the current
  # default value of 2 will be modifed to 0 for raft with convex upsampling.
  smoothness_at_level = min(smoothness_at_level,
                            len(flows[(0, 1, 'augmented-student')]) - 1)

  # Ensure that smoothness_at_level is feasible, i.e. set
  # to be as close as possible to the chosen parameter. This means the
  # current default value of 2 will be modifed to 0 for raft with convex
  # upsampling.
  smoothness_at_level = min(smoothness_at_level,
                            len(flows[(0, 1, 'augmented-student')]) - 1)
  # Always self supervise with the full resolution flows.
  selfsup_at_level = 0

  # Iterate over image pairs.
  for key in flows:
    time_i, time_j, c = key
    key_rev = (time_j, time_i, c)
    if (c not in compute_loss_for_these_flows or
        (only_forward and time_i > time_j)):
      continue

    if full_size_images is not None:
      flow = flows[key][0]
      height = flow.shape[-3]
      width = flow.shape[-2]
      full_height = full_size_images.shape[-3]
      full_width = full_size_images.shape[-2]
      batch_size = flow.shape[0]
      # TODO(smurf): Make work for batch size > 1
      with tf.control_dependencies([
          tf.compat.v1.assert_equal(batch_size, 1),
          tf.compat.v1.assert_greater_equal(full_height, height),
          tf.compat.v1.assert_greater_equal(full_width, width),
      ]):
        flow = tf.pad(flow, [[
            0, 0
        ], [crop_h[0] + pad_h[0], full_height - height - crop_h[0] - pad_h[0]
           ], [crop_w[0] + pad_w[0], full_width - width - crop_w[0] - pad_w[0]],
                             [0, 0]])
        flow.set_shape((batch_size, full_height, full_width, 2))
      warp = flow_to_warp(flow)
      valid_warp_mask = mask_invalid(warp, pad_h, pad_w)
      warped_image = resample(
          tf.stop_gradient(full_size_images[:, time_j]), warp)
      warped_image = tf.image.crop_to_bounding_box(warped_image,
                                                   pad_h[0] + crop_h[0],
                                                   pad_w[0] + crop_w[0], height,
                                                   width)
      valid_warp_mask = tf.image.crop_to_bounding_box(valid_warp_mask,
                                                      pad_h[0] + crop_h[0],
                                                      pad_w[0] + crop_w[0],
                                                      height, width)
    else:
      warp = flow_to_warp(flows[key][0])
      valid_warp_mask = mask_invalid(warp)
      warped_image = resample(tf.stop_gradient(images[:, time_j]), warp)

    occlusion_mask = occlusion_estimation_fn(
        forward_flow=flows[key][0], backward_flow=flows[key_rev][0])

    if stop_gradient_mask:
      mask_level0 = tf.stop_gradient(occlusion_mask * valid_warp_mask)
    else:
      mask_level0 = occlusion_mask * valid_warp_mask

    if 'census' in weights:
      # Loss based on the census transform.
      cen_loss = census_loss(
          image_a_bhw3=images[:, time_i],
          image_b_bhw3=warped_image,
          mask_bhw3=mask_level0)
      losses['census'] += weights['census'] * cen_loss / num_pairs

    # Compute smoothness losses.
    if 'smooth2' in weights or 'smooth1' in weights:
      # Configure function for the edge-aware weighting.
      def edge_weighting_fn(x):
        if smoothness_edge_weighting == 'gaussian':
          return tf.exp(-tf.reduce_mean(
              input_tensor=((smoothness_edge_constant * x)**2),
              axis=-1,
              keepdims=True))
        elif smoothness_edge_weighting == 'exponential':
          return tf.exp(-tf.reduce_mean(
              input_tensor=(abs(smoothness_edge_constant * x)),
              axis=-1,
              keepdims=True))
        else:
          raise ValueError('Only gaussian or exponential edge weighting '
                           'implemented.')

      # Resize multiple times for a smoother result.
      images_at_smoothness_level = images[:, time_i]
      for _ in range(smoothness_at_level):
        height = tf.shape(images_at_smoothness_level)[-3]
        width = tf.shape(images_at_smoothness_level)[-2]
        images_at_smoothness_level = resize(
            images_at_smoothness_level, (height) // 2, (width) // 2,
            is_flow=False)

      if 'smooth1' in weights:
        # Compute first-order smoohtness term loss.
        smooth_loss_1st = first_order_smoothness_loss(
            image=images_at_smoothness_level,
            flow=flows[key][smoothness_at_level],
            edge_weighting_fn=edge_weighting_fn)
        losses['smooth1'] += weights['smooth1'] * smooth_loss_1st / num_pairs

      if 'smooth2' in weights:
        # Compute second-order smoohtness term loss.
        smooth_loss_2nd = second_order_smoothness_loss(
            image=images_at_smoothness_level,
            flow=flows[key][smoothness_at_level],
            edge_weighting_fn=edge_weighting_fn)
        losses['smooth2'] += weights['smooth2'] * smooth_loss_2nd / num_pairs

    # Compute self-supervision loss.
    if 'selfsup' in weights:
      teacher_key = (time_i, time_j, 'original-teacher')
      student_key = (time_i, time_j, 'transformed-student')
      teacher_key_rev = (time_j, time_i, 'original-teacher')
      student_key_rev = (time_j, time_i, 'transformed-student')
      selfsup_loss = self_supervision_loss(
          teacher_flow=flows[teacher_key][selfsup_at_level],
          student_flow=flows[student_key][selfsup_at_level],
          teacher_backward_flow=flows[teacher_key_rev][selfsup_at_level],
          student_backward_flow=flows[student_key_rev][selfsup_at_level],
          selfsup_mask=selfsup_mask,
          selfsup_transform_fn=selfsup_transform_fn,
          fb_sigma_student=fb_sigma_student,
          fb_sigma_teacher=fb_sigma_teacher)
      losses['selfsup'] += weights['selfsup'] * selfsup_loss / num_pairs

  return losses


def self_supervision_loss(teacher_flow, student_flow,
                          teacher_backward_flow,
                          student_backward_flow, selfsup_mask,
                          selfsup_transform_fn,
                          fb_sigma_student,
                          fb_sigma_teacher):
  """Computes self-supervision based on a given teacher and student flow.

  Args:
    teacher_flow: Flow field computed by the teacher model [batch, height,
      width, 2].
    student_flow: Flow field computed by the student model [batch, height,
      width, 2].
    teacher_backward_flow: Backward flow field computed by the teacher model
      [batch, height, width, 2].
    student_backward_flow: Backward flow field computed by the student model
      [batch, height, width, 2].
    selfsup_mask: Indicates what type of masking to use for the self-supervision
      can be either gaussian or ddflow.
    selfsup_transform_fn: Transform function used for the self-supervision. This
      function allows to transform images and flow fields accordingly.
    fb_sigma_student: Sigma used for the gaussian self-supervision masking.
    fb_sigma_teacher: Sigma used for the gaussian self-supervision masking.

  Returns:
    Average self-supervision loss.
  """
  if selfsup_transform_fn is None:
    raise ValueError('Self-supervision transform should not be none if '
                     'self-supervision loss is used.')

  h = tf.cast(tf.shape(input=teacher_flow)[-3], tf.float32)
  w = tf.cast(tf.shape(input=teacher_flow)[-2], tf.float32)

  # Resampled backward flows at forward flow locations.
  student_warp = flow_to_warp(student_flow)
  student_backward_flow_resampled = resample(student_backward_flow,
                                             student_warp)
  teacher_warp = flow_to_warp(teacher_flow)
  teacher_backward_flow_resampled = resample(teacher_backward_flow,
                                             teacher_warp)

  # Compute valid warp masks, i.e. a mask indicating if a pixel was warped
  # outside the image frame.
  student_valid_warp_masks = mask_invalid(student_warp)
  teacher_valid_warp_masks = mask_invalid(teacher_warp)

  student_fb_sq_diff = tf.reduce_sum(
      (student_flow + student_backward_flow_resampled)**2,
      axis=-1,
      keepdims=True)
  teacher_fb_sq_diff = tf.reduce_sum(
      (teacher_flow + teacher_backward_flow_resampled)**2,
      axis=-1,
      keepdims=True)
  if selfsup_mask == 'gaussian':
    student_fb_consistency = tf.exp(-student_fb_sq_diff / (fb_sigma_student**2 *
                                                           (h**2 + w**2)))
    teacher_fb_consistency = tf.exp(-teacher_fb_sq_diff / (fb_sigma_teacher**2 *
                                                           (h**2 + w**2)))
  elif selfsup_mask == 'ddflow':
    student_fb_sum_sq = tf.reduce_sum(
        student_flow**2 + student_backward_flow_resampled**2,
        axis=-1,
        keepdims=True)
    teacher_fb_sum_sq = tf.reduce_sum(
        teacher_flow**2 + teacher_backward_flow_resampled**2,
        axis=-1,
        keepdims=True)

    threshold_student = 0.01 * student_fb_sum_sq + 0.5
    threshold_teacher = 0.01 * teacher_fb_sum_sq + 0.5
    student_fb_consistency = tf.cast(student_fb_sq_diff < threshold_student,
                                     tf.float32)
    teacher_fb_consistency = tf.cast(teacher_fb_sq_diff < threshold_teacher,
                                     tf.float32)
  elif selfsup_mask != 'none':
    raise ValueError('Unknown selfsup_mask', selfsup_mask)

  if selfsup_mask == 'none':
    student_mask = tf.ones_like(student_valid_warp_masks)
    teacher_mask = tf.ones_like(teacher_valid_warp_masks)
  else:
    student_mask = 1. - (student_fb_consistency * student_valid_warp_masks)
    teacher_mask = teacher_fb_consistency * teacher_valid_warp_masks
  teacher_mask = selfsup_transform_fn(teacher_mask, is_flow=False)
  teacher_flow = selfsup_transform_fn(teacher_flow, is_flow=True)

  error = robust_l1(tf.stop_gradient(teacher_flow) - student_flow)
  mask = tf.stop_gradient(teacher_mask * student_mask)

  return tf.reduce_mean(input_tensor=mask * error)


@gin.configurable
def supervised_loss(ground_truth_flow,
                    ground_truth_valid,
                    predicted_flows,
                    weights,
                    multi_level_loss=False,
                    multi_level_decay=0.8,
                    resize_gt_flow=False,
                    loss_type='l2',
                    hinge_loss_boundary=1.0,
                    normalization='mask'):
  """Computes a supervised loss."""
  losses = {}
  losses['supervision'] = 0.0

  levels = [0]  # level 0 corresponds to highest flow resolution
  if multi_level_loss:
    levels = range(len(predicted_flows))

  # loop over all flow field that should be considered in the loss
  for level in levels:
    flow = predicted_flows[level]
    _, height, width, _ = flow.get_shape().as_list()

    if resize_gt_flow:
      _, height, width, _ = flow.get_shape().as_list()
      flow_gt, mask_gt = resize(
          ground_truth_flow,
          height,
          width,
          is_flow=True,
          mask=ground_truth_valid)
    else:
      _, height, width, _ = ground_truth_flow.get_shape().as_list()
      flow = resize(flow, height, width, is_flow=True)
      flow_gt = ground_truth_flow
      mask_gt = ground_truth_valid

    # compute error/loss metric
    if loss_type == 'robust':
      error = robust_l1(flow_gt - flow)
    elif loss_type == 'hinge':
      error = tf.math.maximum(0.0, l2(flow_gt - flow) - hinge_loss_boundary)
    else:
      error = l2(flow_gt - flow)

    level_weight = weights['supervision'] * pow(multi_level_decay, level)
    if normalization == 'mask':
      losses['supervision'] += level_weight * (
          tf.reduce_sum(input_tensor=mask_gt * error) /
          (tf.reduce_sum(input_tensor=mask_gt) + 1e-16))
    elif normalization == 'dimension':
      losses['supervision'] += level_weight * (
          tf.reduce_sum(input_tensor=mask_gt * error) /
          (height*width))
    else:
      raise ValueError('Supervised loss only supports: mask and dimension '
                       'normalization.')
  return losses


@gin.configurable
def supervised_sequence_loss(ground_truth_flow,
                             ground_truth_valid,
                             predicted_flows,
                             weights,
                             max_flow_threshold=400,
                             max_flow_norm='l2',
                             loss_decay=.8):
  """Computes a supervised sequence loss."""
  del weights
  losses = {}
  def fn(accum, predicted_flow):
    height = tf.shape(ground_truth_flow)[-3]
    width = tf.shape(ground_truth_flow)[-2]
    upsampled_flow = resize(predicted_flow, height, width, is_flow=True)

    if max_flow_norm == 'l1':
      # Version used in earlier RAFT implementation with a max_flow_threshold
      # of 1000 and later 500.
      max_flow_mask = tf.cast(
          tf.norm(ground_truth_flow, ord=1, axis=-1, keepdims=True) <
          max_flow_threshold, tf.float32)
    elif max_flow_norm == 'l2':
      max_flow_mask = tf.cast(
          tf.norm(ground_truth_flow, ord='euclidean', axis=-1, keepdims=True) <
          max_flow_threshold, tf.float32)
    else:
      raise ValueError('The value of max_flow_norm within the supervised '
                       'sequence loss must be either l1 or l2.')

    loss = tf.math.abs(ground_truth_flow - upsampled_flow)
    loss = tf.reduce_mean(loss * max_flow_mask * ground_truth_valid)
    return accum * loss_decay + loss

  loss_array = tf.scan(
      fn,
      predicted_flows,
      initializer=tf.convert_to_tensor([0], dtype=tf.float32),
      parallel_iterations=1)

  final_accum_loss = loss_array[-1, 0]
  losses['supervised_sequence_loss'] = final_accum_loss
  return losses


def compute_flows_for_unsupervised_loss(
    feature_model,
    flow_model,
    batch,
    batch_without_aug,
    training,
    selfsup_transform_fn = None,
    return_sequence = False,
    perform_selfsup = True):
  """Computes all flow fields required for the unsupervised loss.

  Args:
    feature_model: Model that computes features given an image.
    flow_model: Model that takes features and ouputs flow estimates at different
      resolutions.
    batch: Image batch with shape [batch, 2, height, width, channels].
    batch_without_aug: Image batch without photometric augmentation applied of
      shape [batch, 2, height, width, channels].
    training: Indicates if training mode is used (this effects the dropout).
    selfsup_transform_fn: Augmentation function applied for realizing the self-
      supervision term.
    return_sequence: Flag if flow sequence should be returned instead of multi-
      resolution flows. This must be supported by the flow model.
    perform_selfsup: Whether or not to compute the additional flows for
      selfsupervision

  Returns:
    Returns multiple flow fields required for the unsupervised loss.
  """
  del training
  # Initialize empty dicts.
  flows = dict()
  features = dict()

  # Compute features of the augmented images.
  features['augmented-student'] = feature_model(batch[:, 0],
                                                batch[:, 1],
                                                bidirectional=True)
  if perform_selfsup:
    # Compute teacher features.
    features['original-teacher'] = feature_model(batch_without_aug[:, 0],
                                                 batch_without_aug[:, 1],
                                                 training=False,
                                                 bidirectional=True)
    # Create further augmentented images for self-supervision.
    tranformed_image_1 = selfsup_transform_fn(batch[:, 0], is_flow=False)
    tranformed_image_2 = selfsup_transform_fn(batch[:, 1], is_flow=False)
    # Compute student features on the further augmentented images.
    features['transformed-student'] = feature_model(tranformed_image_1,
                                                    tranformed_image_2,
                                                    bidirectional=True)

  # Compute all forward backward flow fields.
  def compute_fw_bw_flow(version, training=True):
    """Helper function computes forward and backward flow fields."""
    output = dict()
    # Compute forward flow.
    key = (0, 1, version)
    flow = flow_model(features[version], training=training, backward=False)
    fw_flows = flow_model.get_flow_sequence() if return_sequence else flow[:3]
    output.update({key: fw_flows})
    # Compute backward flow.
    key = (1, 0, version)
    flow = flow_model(features[version], training=training, backward=True)
    bw_flows = flow_model.get_flow_sequence() if return_sequence else flow[:3]
    output.update({key: bw_flows})
    return output

  flows.update(compute_fw_bw_flow('augmented-student'))
  if perform_selfsup:
    flows.update(compute_fw_bw_flow('transformed-student', training=True))
    flows.update(compute_fw_bw_flow('original-teacher', training=False))

  # Restructure to allow simple reusage for the unsupervised sequence loss.
  if return_sequence:
    flows_sequence = []
    for i in range(flow_model.get_flow_sequence_length()):
      flows_sequence.append({key: [flows[key][i]] for key in flows})
    # Always use the final prediction as self-supervision for all prior
    # predictions.
    if perform_selfsup:
      for flow in flows_sequence:
        for key in [(0, 1, 'original-teacher'), (1, 0, 'original-teacher')]:
          flow[key] = flows_sequence[-1][key]
    return flows_sequence
  return flows


def compute_flow_for_supervised_loss(
    feature_model,
    flow_model,
    batch,
    training
):
  """Compute flow for an image batch.

  Args:
    feature_model: A model to compute features for flow.
    flow_model: A model to compute flow.
    batch: A tf.tensor of shape [b, seq, h, w, c] holding a batch of triplets.
    training: bool that tells the model to use training or inference code.

  Returns:
    A tuple consisting of the images, the extracted features, the estimated
    flows, and the upsampled refined flows.
  """
  feature_dict = feature_model(batch[:, 0],
                               batch[:, 1],
                               training=training)
  return flow_model(feature_dict, training=training)


def compute_flow_for_sequence_loss(
    feature_model,
    flow_model,
    batch,
    training):
  """Compute flow for an image batch and return a flow sequence.

  Model must provide get_flow_sequence (RAFT).

  Args:
    feature_model: A model to compute features for flow.
    flow_model: A model to compute flow.
    batch: A tf.tensor of shape [b, seq, h, w, c] holding a batch of triplets.
    training: bool that tells the model to use training or inference code.

  Returns:
    A tuple consisting of the images, the extracted features, the estimated
    flows, and the upsampled refined flows.
  """
  # Check if model provides get_flow_sequence.
  get_flow_sequence_fn = getattr(flow_model, 'get_flow_sequence', None)
  if not callable(get_flow_sequence_fn):
    raise NotImplementedError('Model must provide get_flow_sequence to work '
                              'with sequence-loss.')

  feature_dict = feature_model(batch[:, 0],
                               batch[:, 1],
                               training=training)
  _ = flow_model(feature_dict, training=training)
  return flow_model.get_flow_sequence()


def zero_mask_border(mask_bhw3, patch_size):
  """Used to ignore border effects from census_transform."""
  mask_padding = patch_size // 2
  mask = mask_bhw3[:, mask_padding:-mask_padding, mask_padding:-mask_padding, :]
  return tf.pad(
      tensor=mask,
      paddings=[[0, 0], [mask_padding, mask_padding],
                [mask_padding, mask_padding], [0, 0]])


def census_transform(image, patch_size):  # pylint:disable=missing-function-docstring
  intensities = tf.image.rgb_to_grayscale(image) * 255
  kernel = tf.reshape(
      tf.eye(patch_size * patch_size),
      (patch_size, patch_size, 1, patch_size * patch_size))
  neighbors = tf.nn.conv2d(
      input=intensities, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
  diff = neighbors - intensities
  # Magic numbers taken from DDFlow
  diff_norm = diff / tf.sqrt(.81 + tf.square(diff))
  return diff_norm


# pylint:disable=g-doc-args
# pylint:disable=g-doc-return-or-yield
def soft_hamming(a_bhwk, b_bhwk, thresh=.1):
  """A soft hamming distance between tensor a_bhwk and tensor b_bhwk.

  Returns a tensor with approx. 1 in (h, w) locations that are significantly
  more different than thresh and approx. 0 if significantly less
  different than thresh.
  """
  sq_dist_bhwk = tf.square(a_bhwk - b_bhwk)
  soft_thresh_dist_bhwk = sq_dist_bhwk / (thresh + sq_dist_bhwk)
  return tf.reduce_sum(
      input_tensor=soft_thresh_dist_bhwk, axis=3, keepdims=True)


def census_loss(image_a_bhw3,
                image_b_bhw3,
                mask_bhw3,
                patch_size=7,
                distance_metric_fn=abs_robust_loss):
  """Compare the similarity of the census transform of two images."""
  census_image_a_bhwk = census_transform(image_a_bhw3, patch_size)
  census_image_b_bhwk = census_transform(image_b_bhw3, patch_size)

  hamming_bhw1 = soft_hamming(census_image_a_bhwk,
                              census_image_b_bhwk)

  # set borders of mask to zero to ignore edge effects
  padded_mask_bhw3 = zero_mask_border(mask_bhw3, patch_size)
  diff = distance_metric_fn(hamming_bhw1)
  diff *= padded_mask_bhw3
  diff_sum = tf.reduce_sum(input_tensor=diff)
  loss_mean = diff_sum / (tf.reduce_sum(
      input_tensor=tf.stop_gradient(padded_mask_bhw3) + 1e-6))
  return loss_mean


def time_it(f, num_reps=1, execute_once_before=False):
  """Time a tensorflow function in eager mode.

  Args:
    f: function with no arguments that should be timed.
    num_reps: int, number of repetitions for timing.
    execute_once_before: boolean, whether to execute the function once before
      timing in order to not count the tf.function compile time.
  Returns: tuple of the average time in ms and the functions output.
  """
  assert num_reps >= 1
  # Execute f once before timing it to allow tf.function to compile the graph.
  if execute_once_before:
    x = f()
  # Make sure that there is nothing still running on the GPU by waiting for the
  # completion of a bogus command.
  _ = tf.square(tf.random.uniform([1])).numpy()
  # Time f for a number of repetitions.
  start_in_s = time.time()
  for _ in range(num_reps):
    x = f()
    # Make sure that f has finished and was not just enqueued by using another
    # bogus command. This will overestimate the computing time of f by waiting
    # until the result has been copied to main memory. Calling reduce_sum
    # reduces that overestimation.
    if isinstance(x, tuple) or isinstance(x, list):
      _ = [tf.reduce_sum(input_tensor=xi).numpy() for xi in x]
    else:
      _ = tf.reduce_sum(input_tensor=x).numpy()
  end_in_s = time.time()
  # Compute the average time in ms.
  avg_time = (end_in_s - start_in_s) * 1000. / float(num_reps)
  return avg_time, x


@gin.configurable
def unsupervised_sequence_loss(
    images,
    flows_sequence,
    unsupervised_loss_fn,  # pylint:disable=g-bare-generic
    loss_decay = .8,
    supervision_weight = 0.05,
    mode = 'unsup_per_update',
    full_size_images = None,
    crop_h = None,
    crop_w = None,
    pad_h = None,
    pad_w = None):
  """Computes a unsupervised sequence loss."""
  loss_dict = {}

  def add_loss_dicts(old_dict, new_dict, decay):
    """Adds all losses in the dict considering a decay factor."""
    for key, value in new_dict.items():
      if key not in old_dict:
        old_dict[key] = value
      else:
        old_dict[key] = value + old_dict[key] * decay

  if mode == 'unsup_per_update':
    # Applies the same unsupervised loss for each update iteration.
    for flows in flows_sequence:
      # Compute the losses.
      loss_dict_one_flow = unsupervised_loss_fn(
          images=images,
          flows=flows,
          full_size_images=full_size_images,
          crop_h=crop_h,
          crop_w=crop_w,
          pad_h=pad_h,
          pad_w=pad_w)
      add_loss_dicts(loss_dict, loss_dict_one_flow, loss_decay)

  elif mode == 'unsup_final_update':
    # Applies the unsupervised loss at the final update iteration and uses a
    # supervised loss for all preceding update iterations (using the final flow
    # output as supervision).
    final_flow_fw = flows_sequence[-1][(0, 1, 'augmented-student')][0]
    final_flow_bw = flows_sequence[-1][(1, 0, 'augmented-student')][0]
    dummy_mask = tf.ones_like(final_flow_fw[Ellipsis, :1])
    weights = {'supervision': supervision_weight}

    # Apply supervised loss for all previous iterations.
    for flows in flows_sequence[:-1]:
      # Compute the losses fw flow.
      flow_fw = flows[(0, 1, 'augmented-student')]
      flow_bw = flows[(1, 0, 'augmented-student')]

      loss_dict_fw_bw = {}
      for flow, final in [(flow_fw, final_flow_fw), (flow_bw, final_flow_bw)]:
        # Compute supervised loss.
        loss_dict_one_direction = supervised_loss(
            final, dummy_mask, flow, weights, loss_type='robust')
        # Add intermediate losses for the same update iteration.
        add_loss_dicts(loss_dict_fw_bw, loss_dict_one_direction, decay=1.0)

      # Add losses to the final loss with decay.
      add_loss_dicts(loss_dict, loss_dict_fw_bw, loss_decay)

    # Apply unsupervised loss for the final update iteration.
    flows = flows_sequence[-1]
    loss_dict_one_flow = unsupervised_loss_fn(
        images=images,
        flows=flows,
        full_size_images=full_size_images,
        crop_h=crop_h,
        crop_w=crop_w,
        pad_h=pad_h,
        pad_w=pad_w)
    add_loss_dicts(loss_dict, loss_dict_one_flow, loss_decay)
  else:
    raise ValueError('Unknown mode for unsupervised_sequence_loss.')
  return loss_dict

