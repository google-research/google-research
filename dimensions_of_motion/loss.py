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

# -*- coding: utf-8 -*-
"""Functions to compute loss for Dimensions of Motion."""

import flow_basis
import geometry
import solvers
import tensorflow as tf
import utils


def basis_from_embedding(embedding, basis):
  """Take an embedding and a basis from camera movement and return the product.

  The product is the outer product in which each dimension of the basis
  is multiplied (pixel-wise) by each dimension of the embedding.
  (Equation 15 in the paper.)

  Args:
    embedding: [B, N, H, W]: N dimensions of per-pixel embedding
    basis: [B, M, H, W, 2]: a flow basis

  Returns:
    [B, M*N, H, W, 2]  a new basis of size N*M, consisting of all
    pairwise products of a scalar field from the embedding and a flow field
    from the basis.
  """

  basis = (
      embedding[:, tf.newaxis, Ellipsis, tf.newaxis] *
      basis[:, :, tf.newaxis, Ellipsis])
  basis = utils.collapse_dim(basis, 2)
  return basis


def project_flow(basis, flow):
  """Project the flow into the column space of 'basis'.

  Args:
    basis: [B, N, H, W, 2]: N-dimensional basis for flow
    flow: [B, H, W, 2]: observed flow

  Returns:
    reconstructed_flow: [B, H, W, 2] the projection of flow into the basis
    weights: [B,N] the column weights that produce reconstructed_flow
  """

  # Flatten flow and basis to the shape expected by solver code.
  flat_flow = utils.collapse_dim(flow, 2)
  flat_flow = utils.collapse_dim(flat_flow, 2)
  flat_flow = flat_flow[Ellipsis, None]

  flat_basis = utils.collapse_dim(basis, 3)
  flat_basis = utils.collapse_dim(flat_basis, 3)
  flat_basis = tf.transpose(flat_basis, (0, 2, 1))

  singular_threshold = 1e-5
  reconstructed_flow, weights = solvers.project_svd(
      flat_basis, flat_flow, singular_threshold)

  # Undo flattening.
  reconstructed_flow = tf.reshape(reconstructed_flow, flow.shape)
  return reconstructed_flow, weights


def flow_loss_and_summaries(reconstructed_flow, flow, suffix, loss_weight,
                            loss_dictionary, add_summary):
  """Produce summaries and losses for a reconstructed flow.

  Args:
    reconstructed_flow: [B, H, W, 2] Projected flow
    flow: [B, H, W, 2] Ground-truth flow
    suffix: A string to append to loss and summary names
    loss_weight: Weight for this loss
    loss_dictionary: A LossDictionary, which will be updated with the loss
    add_summary: function(name, image) to add summary images
  """
  flow_error = flow - reconstructed_flow
  # We use L2 norm of flow error. Because we represent flow as a proportion
  # of image width and height, this loss is anisotropic for non-square images.
  # We haven't seen this cause problems in practice.
  flow_loss = tf.norm(flow_error, ord=2, axis=-1, keepdims=True)
  loss_dictionary.add(f'flow_reconstruction_{suffix}',
                      utils.batched_mean(flow_loss), loss_weight)

  add_summary(f'reconstructed_flow_{suffix}', reconstructed_flow)
  add_summary(f'flow_error_{suffix}', flow_error)
  add_summary(f'flow_loss_{suffix}', flow_loss)


class LossDictionary:
  """A dictionary for storing (loss, weight) pairs with a convenience add().
  """

  def __init__(self):
    self._losses_and_weights = {}

  def add(self, key, loss, weight):
    assert key not in self._losses_and_weights, f'Duplicate key {key}'
    self._losses_and_weights[key] = (loss, weight)

  def as_dict(self):
    return self._losses_and_weights


def compute_motion_loss(
    source_image, flow, disparity, embeddings,
    principal_point=None):
  """Uses flow to compute losses from a predicted scene representation.

  disparity is used to compute basis fields for the different directions of
  camera movement. If embeddings is present, it is used in conjunction with
  disparity to compute a basis allowing for object translation and camera
  rotation.

  Args:
    source_image: [B, H, W, 3] Source RGB, so we can summarize it.
    flow: [B, H, W, 2] optical flow (texture coord offsets) source to target.
    disparity: [B, H, W, 1] disparity prediction
    embeddings: [B, A, H, W] predicted object instance embeddings, or None.
    principal_point: [B, 2] Principal point, or None (defaults to center).
  Returns:
    A dictionary of (loss, weight) pairs, in which the losses have shape [B],
    and a dictionary of [B,H,W,C] images / flows / disparities (C=3/2/1) to
    to help visualize what's going on.
  """
  batch_size = source_image.shape.as_list()[0]

  if principal_point is None:
    # Default to center of image
    principal_point = tf.tile([[0.5, 0.5]], [batch_size, 1])

  # Collect summary images
  summary_images = {}
  def add_summary(name, image):
    summary_images[name] = image

  # A map to which we'll add our various losses.
  loss_dictionary = LossDictionary()

  # Begin with image summaries
  add_summary('flow_actual', flow)
  add_summary('warp_gt_flow', geometry.flow_forward_warp(source_image, flow))

  # Generate camera translation and rotation bases.
  (height, width) = disparity.shape[-3:-1]
  translation_basis = flow_basis.camera_translation_basis(
      height=height,
      width=width,
      principal_point=principal_point,
      disparity=disparity)

  rotation_basis = flow_basis.camera_rotation_basis(
      height=height,
      width=width,
      principal_point=principal_point)

  add_summary('basis_rx1', rotation_basis[:, 0])
  add_summary('basis_rx2', rotation_basis[:, 1])
  add_summary('basis_ry1', rotation_basis[:, 2])
  add_summary('basis_ry2', rotation_basis[:, 3])
  add_summary('basis_rz', rotation_basis[:, 4])

  add_summary('basis_tx', translation_basis[:, 0])
  add_summary('basis_ty', translation_basis[:, 1])
  add_summary('basis_tz', translation_basis[:, 2])

  # Compute the nonrotation basis, which is either just the translation basis or
  # an appropriate outer product.
  if embeddings is not None:
    visualize_embedding_pca(embeddings, add_summary, True)
    nonrotation_basis = basis_from_embedding(embeddings, translation_basis)
    for i in range(nonrotation_basis.shape[1]):
      add_summary(f'basis_move_{i}', nonrotation_basis[:, i])
  else:
    nonrotation_basis = translation_basis

  # Main reconstruction loss.
  full_basis = tf.concat((nonrotation_basis, rotation_basis), axis=-4)
  reconstructed_flow, _ = project_flow(full_basis, flow)
  flow_loss_and_summaries(
      reconstructed_flow, flow, 'main',
      1.0, loss_dictionary, add_summary)

  if embeddings is not None:
    # Secondary reconstruction loss allowing camera movement only.
    full_basis_no_embedding = tf.concat(
        (translation_basis, rotation_basis), axis=-4)
    reconstructed_flow_no_embedding, _ = project_flow(
        full_basis_no_embedding, flow)
    flow_loss_and_summaries(
        reconstructed_flow_no_embedding, flow, 'no_embedding',
        0.5, loss_dictionary, add_summary)

  return loss_dictionary.as_dict(), summary_images


def visualize_embedding_pca(embeddings, add_summary, demean):
  """Visualize the embedding space after PCA."""
  embeddings = tf.transpose(embeddings, (0, 2, 3, 1))  # [B, H, W, A]
  embeddings_image_space_shape = embeddings.shape
  embeddings = utils.collapse_dim(embeddings, 2)  # [B, NPIX, A]

  if demean:
    s, u, _ = tf.linalg.svd(embeddings -
                            tf.reduce_mean(embeddings, axis=1, keepdims=True))
  else:
    s, u, _ = tf.linalg.svd(embeddings)
  components = tf.matmul(u, tf.linalg.diag(s))  # [B, NPIX, A]
  components = tf.reshape(components,
                          embeddings_image_space_shape)  # [B, H, W, A]
  components = components - tf.reduce_min(
      components, axis=(1, 2, 3), keepdims=True)
  components = components / (
      1e-6 + tf.reduce_max(components, axis=(1, 2, 3), keepdims=True))
  components = components[Ellipsis, None]  # [B,H,W,A,1]
  for (i, data) in enumerate(tf.unstack(components, axis=3)):
    add_summary(f'embedding_pca_{"" if demean else "no"}demean_{i}',
                data)
