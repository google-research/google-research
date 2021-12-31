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

"""Evaluation functions for IPDF models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
from absl import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg


@tf.function
def _get_probabilities(vision_model, model_head, images, so3_grid):
  vision_description = vision_model(images, training=False)
  query_rotations = tf.reshape(so3_grid, [-1, model_head.len_rotation])
  query_rotations = model_head.positional_encoding(query_rotations)
  logits = model_head.implicit_model(
      [vision_description, query_rotations], training=False)[Ellipsis, 0]
  return tf.nn.softmax(logits, axis=-1)


def eval_spread_and_loglikelihood(vision_model, model_head, dataset,
                                  batch_size=32, eval_grid_size=None,
                                  skip_spread_evaluation=False,
                                  number_eval_iterations=None):
  """Distribution-based evaluation functions for pose estimation.

  Args:
    vision_model: The model which produces a feature vector to hand to IPDF.
    model_head: The IPDF model.
    dataset: The dataset of images paired with single-valued ground truth
      rotations.
    batch_size: batch size for chunking up the evaluation operations.
    eval_grid_size: if supplied, sets the resolution of the grid to use for
      evaluation.
    skip_spread_evaluation: Whether to skip the spread calculation, which can
      take a while and is uninformative for the three shapes (tetX, cylO,
      sphereX) without the full set of ground truth annotations.
    number_eval_iterations: stop evaluation after this number of steps.

  Returns:
    Average log likelihood and average spread (in degrees)
  """
  spreads_all = []
  loglikelihoods_all = []
  so3_grid = model_head.get_closest_available_grid(eval_grid_size)

  for step, (images, rotation_matrices_gt) in enumerate(
      dataset.batch(batch_size)):
    if step % 100 == 0:
      logging.info('Eval step %d', step)
    if number_eval_iterations is not None and step >= number_eval_iterations:
      break
    tf_probabilities = _get_probabilities(
        vision_model, model_head, images, so3_grid)
    np_probabilities = np.float32(tf_probabilities)
    max_inds = find_closest_rot_inds_rotmat(
        so3_grid, rotation_matrices_gt)
    max_inds = np.array(max_inds)
    probabilities_gt = np.float32(
        [np_probabilities[i][max_inds[i]] for i in range(max_inds.shape[0])])
    # Divide by the volume of each cell, pi**2/N, to get probability density.
    loglikelihood = np.log(probabilities_gt * so3_grid.shape[0] / np.pi**2)
    loglikelihoods_all.append(loglikelihood)
    if skip_spread_evaluation:
      spread = [0]
    else:
      spread = [compute_spread_tf(
          so3_grid, tf_probabilities, rotation_matrices_gt)]
    spreads_all.append(np.float32(spread))
  loglikelihoods_all = np.concatenate(loglikelihoods_all, 0)
  spreads_all = np.concatenate(spreads_all, 0)
  return np.mean(loglikelihoods_all), np.rad2deg(np.mean(spreads_all))


def compute_spread(rotations, probabilities, rotations_gt):
  """Measures the spread of a distribution (or mode) around ground truth(s).

  In the case of one ground truth, this is the expected angular error.
  When there are multiple ground truths, the only related quantity that makes
  sense is the expected angular error to the nearest ground truth.

  Args:
    rotations: The grid of rotation matrices for which the probabilities were
      evaluated.
    probabilities: The probability for each rotation.
    rotations_gt: The set of ground truth rotation matrices.
  Returns:
    A scalar, the spread (in radians).
  """
  assert len(probabilities) == len(rotations)
  assert np.allclose(probabilities.sum(), 1.)
  dists = geodesic_distance_rotmats_pairwise_np(rotations, rotations_gt)
  min_distance_to_gt = np.min(dists, axis=1)
  return (probabilities * min_distance_to_gt).sum()


@tf.function
def compute_spread_tf(rotations, probabilities, rotations_gt):
  """TensorFlow version of compute_spread."""
  min_distance_to_gt = min_geodesic_distance_rotmats_pairwise_tf(
      rotations, rotations_gt)
  return tf.reduce_sum(probabilities * min_distance_to_gt, axis=-1)


def eval_single_estimate_accuracy(vision_model,
                                  model_head,
                                  dataset,
                                  batch_size=32,
                                  gradient_ascent=False):
  """Evaluate an IPDF model using single-valued estimates.

  We don't use this for symsol because the multi-valued ground truth makes
  single pose estimates uninformative. For a pose estimation task where the
  ground truth is taken to be single-valued, like Pascal3D+, this is the method
  to use.

  Args:
    vision_model: The model which produces a feature vector to hand to IPDF.
    model_head: The IPDF model.
    dataset: The dataset of images paired with single-valued ground truth
      rotations.
    batch_size: batch size for chunking up the evaluation operations.
    gradient_ascent: Whether to further optimize each rotation prediction
      through gradient ascent on the predicted probability distribution, or just
      use the argmax over a set of sampled rotations as the estimate.
  Returns:
    median_angular_error: The median error, in degrees
    accuracy15: The accuracy at 15 degrees (the proportion of estimates within
      15 degrees from the ground truth).
    accuracy30: The same, but for 30 degrees.
  """
  geodesic_errors = []
  for images, rotation_matrices_gt in dataset.batch(batch_size):
    vision_description = vision_model(images, training=False)
    rotation_matrices_pred = model_head.predict_rotation(
        vision_description, gradient_ascent=gradient_ascent)
    geodesic_errors.append(geodesic_distance_rotmats(rotation_matrices_pred,
                                                     rotation_matrices_gt))

  geodesic_errors = np.rad2deg(np.concatenate(geodesic_errors, 0))
  median_angular_error = np.median(geodesic_errors)
  accuracy15 = np.average(geodesic_errors <= 15)
  accuracy30 = np.average(geodesic_errors <= 30)

  return median_angular_error, accuracy15, accuracy30


def geodesic_distance_rotmats(r1s, r2s):
  """Computes batched geodesic distances between two corresponding sets of rotations.

  Args:
    r1s: [N, 3, 3] tensor
    r2s: [N, 3, 3] tensor

  Returns:
    N angular distances.
  """
  prod = tf.matmul(r1s, r2s, transpose_b=True)
  tr_prod = tf.linalg.trace(prod)
  angs = tf.math.acos(tf.clip_by_value((tr_prod-1.0)/2.0, -1.0, 1.0))
  return angs


def geodesic_distance_rotmats_pairwise_np(r1s, r2s):
  """Computes pairwise geodesic distances between two sets of rotation matrices.

  Args:
    r1s: [N, 3, 3] numpy array
    r2s: [M, 3, 3] numpy array

  Returns:
    [N, M] angular distances.
  """
  rot_rot_transpose = np.einsum('aij,bkj->abik', r1s, r2s, optimize=True)
  tr = np.trace(rot_rot_transpose, axis1=-2, axis2=-1)
  return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def geodesic_distance_rotmats_pairwise_tf(r1s, r2s):
  """TensorFlow version of `geodesic_distance_rotmats_pairwise_np`."""
  # These are the traces of R1^T R2
  trace = tf.einsum('aij,bij->ab', r1s, r2s)
  return tf.acos(tf.clip_by_value((trace - 1.0) / 2.0, -1.0, 1.0))


def min_geodesic_distance_rotmats_pairwise_tf(r1s, r2s):
  """Compute min geodesic distance for each R1 wrt R2."""
  # These are the traces of R1^T R2
  trace = tf.einsum('...aij,...bij->...ab', r1s, r2s)
  # closest rotation has max trace
  max_trace = tf.reduce_max(trace, axis=-1)
  return tf.acos(tf.clip_by_value((max_trace - 1.0) / 2.0, -1.0, 1.0))


@tf.function
def find_closest_rot_inds(grid_quats, gt_quats):
  """Dot the sets of quaternions, and take the absolute value because -q=+q.

  Args:
    grid_quats: tensor, shape [N, 4]
    gt_quats: tensor, shape [bs, M, 4]

  Returns:
    [bs, M] indices, one for each of the gt_quats
  """
  if tf.rank(gt_quats) == 2:
    gt_quats = gt_quats[None]
  dotps = tf.abs(tf.einsum('ij,lkj->ilk', grid_quats, gt_quats))
  # shape is [N, bs, M]
  max_inds = tf.argmax(dotps, axis=0)
  return max_inds


@tf.function
def find_closest_rot_inds_rotmat(grid_rot, gt_rot):
  """Same as find_closest_rot_inds, but with rotation matrices."""
  if tf.rank(gt_rot) == 2:
    gt_rot = gt_rot[None]
  # These are traces of R1^T R2
  traces = tf.einsum('gij,lkij->glk', grid_rot, gt_rot)
  max_inds = tf.argmax(traces, axis=0)
  return max_inds


def plot_to_image(figure):
  """Converts matplotlib fig to a png for logging with tf.summary.image."""
  buffer = io.BytesIO()
  plt.savefig(buffer, format='png', dpi=100)
  plt.close(figure)
  buffer.seek(0)
  image = tf.image.decode_png(buffer.getvalue(), channels=4)
  return image[tf.newaxis]


def visualize_model_output(vision_model,
                           model_head,
                           images,
                           rotation_matrices_gt=None):
  """Display distributions over SO(3).

  Args:
    vision_model: The model which produces a feature vector to hand to IPDF.
    model_head: The IPDF model.
    images: A list of images.
    rotation_matrices_gt: A list of [N, 3, 3] tensors, representing the ground
      truth rotation matrices corresponding to the images.

  Returns:
    A tensor of images to output via Tensorboard.
  """
  return_images = []
  num_to_display = 5

  query_rotations = model_head.get_closest_available_grid(
      model_head.number_eval_queries)
  probabilities = []
  for image in images:
    probabilities.append(_get_probabilities(vision_model,
                                            model_head,
                                            image[None],
                                            query_rotations))
  probabilities = tf.concat(probabilities, 0)

  inches_per_subplot = 4
  canonical_rotation = np.float32(tfg.rotation_matrix_3d.from_euler([0.2]*3))
  for image_index in range(num_to_display):
    fig = plt.figure(figsize=(3*inches_per_subplot, inches_per_subplot),
                     dpi=100)
    gs = fig.add_gridspec(1, 3)
    fig.add_subplot(gs[0, 0])
    plt.imshow(images[image_index])
    plt.axis('off')
    ax2 = fig.add_subplot(gs[0, 1:], projection='mollweide')
    return_fig = visualize_so3_probabilities(
        query_rotations,
        probabilities[image_index],
        rotation_matrices_gt[image_index],
        ax=ax2,
        fig=fig,
        display_threshold_probability=1e-2 / query_rotations.shape[0],
        canonical_rotation=canonical_rotation)
    return_images.append(return_fig)
  return tf.concat(return_images, 0)


def visualize_so3_probabilities(rotations,
                                probabilities,
                                rotations_gt=None,
                                ax=None,
                                fig=None,
                                display_threshold_probability=0,
                                to_image=True,
                                show_color_wheel=True,
                                canonical_rotation=np.eye(3)):
  """Plot a single distribution on SO(3) using the tilt-colored method.

  Args:
    rotations: [N, 3, 3] tensor of rotation matrices
    probabilities: [N] tensor of probabilities
    rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
    ax: The matplotlib.pyplot.axis object to paint
    fig: The matplotlib.pyplot.figure object to paint
    display_threshold_probability: The probability threshold below which to omit
      the marker
    to_image: If True, return a tensor containing the pixels of the finished
      figure; if False return the figure itself
    show_color_wheel: If True, display the explanatory color wheel which matches
      color on the plot with tilt angle
    canonical_rotation: A [3, 3] rotation matrix representing the 'display
      rotation', to change the view of the distribution.  It rotates the
      canonical axes so that the view of SO(3) on the plot is different, which
      can help obtain a more informative view.

  Returns:
    A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
  """
  def _show_single_marker(ax, rotation, marker, edgecolors=True,
                          facecolors=False):
    eulers = tfg.euler.from_rotation_matrix(rotation)
    xyz = rotation[:, 0]
    tilt_angle = eulers[0]
    longitude = np.arctan2(xyz[0], -xyz[1])
    latitude = np.arcsin(xyz[2])

    color = cmap(0.5 + tilt_angle / 2 / np.pi)
    ax.scatter(longitude, latitude, s=2500,
               edgecolors=color if edgecolors else 'none',
               facecolors=facecolors if facecolors else 'none',
               marker=marker,
               linewidth=4)

  if ax is None:
    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_subplot(111, projection='mollweide')
  if rotations_gt is not None and len(tf.shape(rotations_gt)) == 2:
    rotations_gt = rotations_gt[tf.newaxis]

  display_rotations = rotations @ canonical_rotation
  cmap = plt.cm.hsv
  scatterpoint_scaling = 4e3
  eulers_queries = tfg.euler.from_rotation_matrix(display_rotations)
  xyz = display_rotations[:, :, 0]
  tilt_angles = eulers_queries[:, 0]

  longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
  latitudes = np.arcsin(xyz[:, 2])

  which_to_display = (probabilities > display_threshold_probability)

  if rotations_gt is not None:
    # The visualization is more comprehensible if the GT
    # rotation markers are behind the output with white filling the interior.
    display_rotations_gt = rotations_gt @ canonical_rotation

    for rotation in display_rotations_gt:
      _show_single_marker(ax, rotation, 'o')
    # Cover up the centers with white markers
    for rotation in display_rotations_gt:
      _show_single_marker(ax, rotation, 'o', edgecolors=False,
                          facecolors='#ffffff')

  # Display the distribution
  ax.scatter(
      longitudes[which_to_display],
      latitudes[which_to_display],
      s=scatterpoint_scaling * probabilities[which_to_display],
      c=cmap(0.5 + tilt_angles[which_to_display] / 2. / np.pi))

  ax.grid()
  ax.set_xticklabels([])
  ax.set_yticklabels([])

  if show_color_wheel:
    # Add a color wheel showing the tilt angle to color conversion.
    ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
    theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
    radii = np.linspace(0.4, 0.5, 2)
    _, theta_grid = np.meshgrid(radii, theta)
    colormap_val = 0.5 + theta_grid / np.pi / 2.
    ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
    ax.set_yticklabels([])
    ax.set_xticklabels([r'90$\degree$', None,
                        r'180$\degree$', None,
                        r'270$\degree$', None,
                        r'0$\degree$'], fontsize=14)
    ax.spines['polar'].set_visible(False)
    plt.text(0.5, 0.5, 'Tilt', fontsize=14,
             horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)

  if to_image:
    return plot_to_image(fig)
  else:
    return fig
