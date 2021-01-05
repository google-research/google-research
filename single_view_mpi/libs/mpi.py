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

# -*- coding: utf-8 -*-
"""Utilities for working with Multiplane Images (MPIs).

A multiplane image is a set of RGB + alpha textures, positioned as fronto-
parallel planes at specific depths from a reference camera. It represents a
lightfield and can be used to render new views from nearby camera positions
by warping each texture according to its plane homography and combining the
results with an over operation. More detail at:
   https://people.eecs.berkeley.edu/~tinghuiz/projects/mpi/

In this code, an MPI is represented by a tensor of layer textures and a tensor
of depths:
  layers: [..., L, H, W, 4] -- L is the number of layers, last dimension is
          typically RGBA but it can be any number of channels as long as the
          last channel is alpha.
  depths: [..., L] -- distances of the planes from the reference camera.

Layers and depths are stored back-to-front, i.e. farthest layer ("layer 0")
comes first. Typically the depths are chosen so that the corresponding
disparities (inverse depths) form an arithmetic sequence.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from single_view_mpi.libs import geometry
from single_view_mpi.libs import utils


@utils.name_scope
def layer_visibility(alphas):
  """Compute visibility for each pixel in each layer.

  Visibility says how unoccluded each pixel is by the corresponding pixels in
  front of it (i.e. those pixels with the same (x,y) position in subsequent
  layers). The front layer has visibility 1 everywhere since nothing can occlude
  it. Each other layer has visibility equal to the product of (1 - alpha) for
  all the layers in front of it.

  Args:
    alphas: [..., L, H, W, 1] Alpha channels for L layers, back to front.

  Returns:
    [..., L, H, W, 1] visibilities.
  """
  return tf.math.cumprod(
      1.0 - alphas, axis=-4, exclusive=True, reverse=True)


@utils.name_scope
def layer_weights(alphas):
  """Compute contribution weights for each layer from a set of alpha channels.

  The weights w_i for each layer are determined from the layer alphas so that
  to composite the layers we simple multiply each by its weight and add them
  up. In other words, the weight says how much each layer contributes to the
  final composed image.

  For alpha-blending, the weight of a layer at a point is its visibility at that
  point times its alpha at that point, i.e:
       alpha_i * (1 - alpha_i+1) * (1 - alpha_i+2) * ... (1 - alpha_n-1)
  If the first (i.e. the back) layer has alpha=1 everywhere, then the output
  weights will sum to 1 at each point.

  Args:
     alphas: [..., L, H, W, 1] Alpha channels for L layers, back to front.

  Returns:
     [..., L, H, W, 1] The resulting layer weights.
  """
  return alphas * layer_visibility(alphas)


@utils.name_scope
def compose_back_to_front(images):
  """Compose a set of images (for example, RGBA), back to front.

  Args:
    images: [..., L, H, W, C+1] Set of L images, with alpha in the last channel.

  Returns:
    [..., H, W, C] Composed image.
  """
  weights = layer_weights(images[Ellipsis, -1:])
  return tf.reduce_sum(images[Ellipsis, :-1] * weights, axis=-4)


@utils.name_scope
def disparity_from_layers(layers, depths):
  """Compute disparity map from a set of MPI layers.

  From reference view.

  Args:
    layers: [..., L, H, W, C+1] MPI layers, back to front.
    depths: [..., L] depths for each layer.

  Returns:
    [..., H, W, 1] Single-channel disparity map from reference viewpoint.
  """
  disparities = 1.0 / depths
  # Add height, width and channel axes to disparities, so it can broadcast.
  disparities = disparities[Ellipsis, tf.newaxis, tf.newaxis, tf.newaxis]
  weights = layer_weights(layers[Ellipsis, -1:])

  # Weighted sum of per-layer disparities:
  return tf.reduce_sum(disparities * weights, axis=-4)


@utils.name_scope
def make_depths(front_depth, back_depth, num_planes):
  """Returns a list of MPI plane depths, back to front.

  The first element in the list will be back_depth, and last will be
  near-depth, and in between there will be num_planes intermediate
  depths, which are interpolated linearly in disparity.

  Args:
    front_depth: The depth of the front-most MPI plane.
    back_depth: The depth of the back-most MPI plane.
    num_planes: The total number of planes to create.

  Returns:
    [num_planes] A tensor of depths sorted in descending order (so furthest
    first). This order is useful for back to front compositing.
  """
  assert front_depth < back_depth

  front_disparity = 1.0 / front_depth
  back_disparity = 1.0 / back_depth
  disparities = tf.linspace(back_disparity, front_disparity, num_planes)
  return 1.0 / disparities


@utils.name_scope
def render_layers(layers,
                  depths,
                  pose,
                  intrinsics,
                  target_pose,
                  target_intrinsics,
                  height=None,
                  width=None,
                  clamp=True):
  """Render target layers from MPI representation.

  Args:
    layers: [..., L, H, W, C] MPI layers, back to front.
    depths: [..., L] MPI plane depths, back to front.
    pose: [..., 3, 4] reference camera pose.
    intrinsics: [..., 4] reference intrinsics.
    target_pose: [..., 3, 4] target camera pose.
    target_intrinsics: [..., 4] target intrinsics.
    height: height to render to in pixels (or None for input height).
    width: width to render to in pixels (or None for input width).
    clamp: whether to clamp image coordinates (see geometry.sample_image doc),
      i.e. extending the image beyond its size or not.

  Returns:
    [..., L, height, width, C] The layers warped to the target view by applying
    an appropriate homography to each one.
  """

  source_to_target_pose = geometry.mat34_product(
      target_pose, geometry.mat34_pose_inverse(pose))

  # Add a dimension to correspond to L in the poses and intrinsics.
  pose = pose[Ellipsis, tf.newaxis, :, :]  # [..., 1, 3, 4]
  target_pose = target_pose[Ellipsis, tf.newaxis, :, :]  # [..., 1, 3, 4]
  intrinsics = intrinsics[Ellipsis, tf.newaxis, :]  # [..., 1, 4]
  target_intrinsics = target_intrinsics[Ellipsis, tf.newaxis, :]  # [..., 1, 4]

  # Fronto-parallel plane equations at the given depths, in the reference
  # camera's frame.
  normals = tf.constant([0.0, 0.0, 1.0], shape=[1, 3])
  depths = -depths[Ellipsis, tf.newaxis]  # [..., L, 1]
  normals, depths = utils.broadcast_to_match(normals, depths, ignore_axes=1)
  planes = tf.concat([normals, depths], axis=-1)  # [..., L, 4]

  homographies = geometry.inverse_homography(pose, intrinsics, target_pose,
                                             target_intrinsics,
                                             planes)  # [..., L, 3, 3]
  # Each of the resulting [..., L] homographies knows how to inverse-warp one
  # of the [..., (H,W), L] images into a new [... (H',W')] target images.
  target_layers = geometry.homography_warp(
      layers, homographies, height=height, width=width, clamp=clamp)

  # The next few lines implement back-face culling.
  #
  # We don't want to render content that is behind the camera. (If we did, we
  # might see upside-down images of the layers.) A typical graphics approach
  # would be to test each pixel of each layer against a near-plane and discard
  # those that are in front of it. Here we implement something cheaper:
  # back-face culling. If the target camera sees the "back" of a layer then we
  # set that layer's alpha to zero. This is simple and sufficient in practice
  # to avoid nasty artefacts.

  # Convert planes to target camera space. target_planes is [..., L, 4]
  target_planes = geometry.mat34_transform_planes(source_to_target_pose, planes)

  # Fourth coordinate of plane is negative distance in front of the camera.
  # target_visible is [..., L]
  target_visible = tf.cast(target_planes[Ellipsis, -1] < 0.0, dtype=tf.float32)
  # per_layer_alpha is [..., L, 1, 1, 1]
  per_layer_alpha = target_visible[Ellipsis, tf.newaxis, tf.newaxis, tf.newaxis]
  # Multiply alpha channel by per_layer_alpha:
  non_alpha_channels = target_layers[Ellipsis, :-1]
  alpha = target_layers[Ellipsis, -1:] * per_layer_alpha

  target_layers = tf.concat([non_alpha_channels, alpha], axis=-1)
  return target_layers


@utils.name_scope
def render(layers,
           depths,
           pose,
           intrinsics,
           target_pose,
           target_intrinsics,
           height=None,
           width=None,
           clamp=True):
  """Render target image from MPI representation.

  Args:
    layers: [..., L, H, W, C+1] MPI layers back to front, alpha in last channel.
    depths: [..., L] MPI plane depths, back to front
    pose: [..., 3, 4] reference camera pose
    intrinsics: [..., 4] reference intrinsics
    target_pose: [..., 3, 4] target camera pose
    target_intrinsics: [..., 4] target intrinsics
    height: height to render to in pixels (or None for input height)
    width: width to render to in pixels (or None for input width)
    clamp: whether to clamp image coordinates (see geometry.sample_image doc).
      i.e. extending the image beyond its size or not

  Returns:
    [...., height, width, C] Rendered image at the target view.
  """
  target_layers = render_layers(
      layers,
      depths,
      pose,
      intrinsics,
      target_pose,
      target_intrinsics,
      height=height,
      width=width,
      clamp=clamp)
  return compose_back_to_front(target_layers)

