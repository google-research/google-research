# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Library for aligning stacks of images.

Given a fixed number of images, defines a free parameter of control
points for a bspline warp that aligns all images.
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.resampler as contrib_resampler
from factorize_a_city.libs import bspline


def interpolate_2d(knots, positions, degree, cyclical):
  """Interpolates the knot values at positions of a bspline surface warp.

  Sparse mode b-spline warp so the memory usage is efficient. This is a
  2D version of tfg.math.interpolation.bspline.interpolate.

  Args:
    knots: A tensor with shape [bsz, KH, KW, KCh] representing the values to be
      interpolated over. In warping these are control_points.
    positions: A tensor with shape [bsz, H, W, 2] that defines the desired
      positions to interpolate. Positions must be between [0, KHW - D) for
      non-cyclical and [0, KHW) for cyclical splines, where KHW is the number of
      knots on the height or width dimension and D is the spline degree. The
      last dimension of positions record [y, x] coordinates.
    degree: An int describing the degree of the spline. There must be at least D
      + 1 horizontal and vertical knots.
    cyclical: A length-two tuple bool describing whether the spline is cyclical
      in the height and width dimension respectively.

  Returns:
    A tensor of shape '[bsz, H, W, KCh]' with the interpolated value based on
    the control points at various positions.

  Raises:
    ValueError: If degree is greater than 4 or num_knots - 1, or less than 0.
    InvalidArgumentError: If positions are not in the right range.
  """
  batch_size, knots_height, knots_width, knots_ch = knots.shape.as_list()
  y_weights, y_ind = bspline.knot_weights(
      positions[Ellipsis, 0], knots_height, degree, cyclical[0], sparse_mode=True)

  x_weights, x_ind = bspline.knot_weights(
      positions[Ellipsis, 1], knots_width, degree, cyclical[1], sparse_mode=True)
  if cyclical[0]:
    stacked_y_inds = []
    for i in range(-degree // 2, degree // 2 + 1):
      stacked_y_inds.append(y_ind + i)
    stacked_y = tf.stack(stacked_y_inds, axis=-1)
    stacked_y = tf.floormod(stacked_y, knots_height)
  else:
    stacked_y = tf.stack([y_ind + i for i in range(degree + 1)], axis=-1)
  if cyclical[1]:
    stacked_x_inds = []
    for i in range(-degree // 2, degree // 2 + 1):
      stacked_x_inds.append(x_ind + i)
    stacked_x = tf.stack(stacked_x_inds, axis=-1)
    stacked_x = tf.floormod(stacked_x, knots_width)
  else:
    stacked_x = tf.stack([x_ind + i for i in range(degree + 1)], axis=-1)

  stacked_y = stacked_y[:, :, :, :, tf.newaxis]
  stacked_x = stacked_x[:, :, :, tf.newaxis, :]

  stacked_y += tf.zeros_like(stacked_x)
  stacked_x += tf.zeros_like(stacked_y)

  batch_ind = tf.range(
      0, batch_size, 1, dtype=tf.int32)[:, tf.newaxis, tf.newaxis, tf.newaxis,
                                        tf.newaxis]
  batch_ind += tf.zeros_like(stacked_y)

  # tf.gather process dimensions left to right which means (batch, H, W)
  gather_idx = tf.stack([stacked_y, stacked_x], axis=-1)
  original_shape_no_channel = gather_idx.shape.as_list()[:-1]
  gather_nd_indices = tf.reshape(gather_idx, [batch_size, -1, 2])

  relevant_cp = tf.gather_nd(knots, gather_nd_indices, batch_dims=1)
  reshaped_cp = tf.reshape(relevant_cp, original_shape_no_channel + [knots_ch])

  mixed = y_weights[:, :, :, :, tf.newaxis] * x_weights[:, :, :, tf.newaxis, :]
  mixed = mixed[Ellipsis, tf.newaxis]
  return tf.reduce_sum(reshaped_cp * mixed, axis=[-2, -3])


def bspline_warp(cps, image, degree, regularization=0, pano_pad=False):
  """Differentiable 2D alignment of a stack of nearby panoramas.

  Entry point for regularized b-spline surface warp with appropriate handling
  for boundary padding of panoramas. Includes the image resampling operation.

  Args:
    cps: Control points [bsz, H_CP, W_CP, d] defining the deformations.
    image: An image tensor [bsz, H, W, 3] from which we sample deformed
      coordinates.
    degree: Defines the degree of the b-spline interpolation.
    regularization: A float ranging from [0, 1] that smooths the extremes of the
      control points. The effect is that the network has some leeway in fitting
      the original control points exactly.
    pano_pad: When true pads the image and uses a cyclical horizontal warp.
      Useful for warping panorama images.

  Returns:
    A warped image based on deformations specified by control points at various
    positions. Has shape [bsz, H, W, d]

  Raises:
    ValueError: If degree is greater than 4 or num_knots - 1, or less than 0.
    InvalidArgumentError: If positions are not in the right range.
  """

  if regularization < 0 or regularization > 1:
    raise ValueError("b-spline regularization must be between [0, 1]")

  if regularization > 0.:
    # Regularizing constraint on the local structure of control points.
    #   New control points is:
    #     regularization * ave_neighbor + (1-regularization) * cp
    cps_down = tf.concat([cps[:, 1:], cps[:, -1:]], axis=1)
    cps_up = tf.concat([cps[:, :1], cps[:, :-1]], axis=1)
    if pano_pad:
      cps_left = tf.roll(cps, shift=1, axis=2)
      cps_right = tf.roll(cps, shift=-1, axis=2)
    else:
      cps_left = tf.concat([cps[:, :, :1], cps[:, :, :-1]], axis=2)
      cps_right = tf.concat([cps[:, :, 1:], cps[:, :, -1:]], axis=2)
    cps_reg = (cps_left + cps_right + cps_up + cps_down) / 4.
    cps = cps * (1 - regularization) + cps_reg * (regularization)
  tf.summary.image("cps_h", cps[Ellipsis, :1])
  tf.summary.image("cps_w", cps[Ellipsis, 1:])

  batch_size, small_h, small_w, unused_d = cps.shape.as_list()
  unused_batch_size, big_h, big_w, unused_d = image.shape.as_list()

  # Control points are "normalized" in the sense that they're agnostic to the
  # resolution of the image being warped.
  cps = cps * np.array([big_h, big_w])

  y_coord = tf.linspace(0., small_h - 3 - 1e-4, big_h - 4)
  y_coord = tf.concat(
      [tf.zeros([2]), y_coord,
       tf.ones([2]) * (small_h - 3 - 1e-4)], axis=0)
  y_coord = y_coord[:, tf.newaxis]
  if pano_pad:
    x_coord = tf.linspace(0., small_w + 1 - 1e-4, big_w)[tf.newaxis, :]
  else:
    x_coord = tf.linspace(0., small_w - 3 - 1e-4, big_w - 4)
    x_coord = tf.concat(
        [tf.zeros([
            2,
        ]), x_coord,
         tf.ones([
             2,
         ]) * (small_w - 3 - 1e-4)], axis=0)
    x_coord = x_coord[tf.newaxis, :]
  y_coord += tf.zeros_like(x_coord)
  x_coord += tf.zeros_like(y_coord)

  stacked_coords = tf.stack([y_coord, x_coord], axis=-1)[tf.newaxis]
  stacked_coords = tf.tile(stacked_coords, [batch_size, 1, 1, 1])
  estimated_offsets = interpolate_2d(cps, stacked_coords, degree,
                                     [False, pano_pad])
  tf.summary.image("y_flowfield", estimated_offsets[Ellipsis, :1])
  tf.summary.image("x_flowfield", estimated_offsets[Ellipsis, 1:])

  y_coord_sample = tf.range(0., big_h, 1)[:, tf.newaxis]
  x_coord_sample = tf.range(0., big_w, 1)[tf.newaxis, :]

  y_coord_sample += tf.zeros_like(x_coord_sample)
  x_coord_sample += tf.zeros_like(y_coord_sample)

  y_coord_sample += estimated_offsets[Ellipsis, 0]
  x_coord_sample += estimated_offsets[Ellipsis, 1]
  y_clipped = tf.clip_by_value(y_coord_sample, 0, big_h - 1)
  if pano_pad:
    x_clipped = tf.floormod(x_coord_sample, big_w)
    image = tf.concat([image, image[:, :, :1]], axis=2)
  else:
    x_clipped = tf.clip_by_value(x_coord_sample, 0, big_w - 1)

  stacked_resampler_coords = tf.stack([x_clipped, y_clipped], axis=-1)
  return contrib_resampler.resampler(image, stacked_resampler_coords)


class ImageAlignment(object):
  """A class for aligning a set of images using bspline warps."""

  def __init__(self,
               regularization=0.3,
               clip_margin=32,
               pano_pad=True,
               spline_degree=3):
    """Initializes a layer that warp images based on alignment parameters.

    Args:
      regularization: (float): A regularization, ranging from [0, 1], for
        alignment control points
      clip_margin (int): For non-panoramic padding dimensions, controls how many
        edge pixels to crop. Useful for removing warping artifacts near the
        boundary due to missing pixels.
      pano_pad (bool): If true, performs a panoramic spline warp in the
        horizontal dimension.
      spline_degree (int): Degree of the spline
    """
    self.regularization = regularization

    self.clip_margin = clip_margin
    self.pano_pad = pano_pad
    self.spline_degree = spline_degree

  def align_images(self, image, thetas):
    """Performs a warp on the images with the sub_theta control points.

    Warped images are clipped along the edges to prevent missing pixels
    from being visible as a result of warping.

    Args:
      image: [num_ims, H, W, d] image tensor
      thetas: [num_ims, h_cp, w_cp, 2] control points to warp

    Returns:
      clipped: [num_ims, new_H, new_W, d] where new_H and new_W depend on
        pano_pad and clip_margin.
    """
    aligned_results = bspline_warp(
        thetas,
        image,
        self.spline_degree,
        regularization=self.regularization,
        pano_pad=self.pano_pad)
    if self.pano_pad:
      clipped_results = aligned_results[:, self.clip_margin:-self.clip_margin]
    else:
      clipped_results = aligned_results[:, self.clip_margin:-self.clip_margin,
                                        self.clip_margin:-self.clip_margin, :]
    return clipped_results
