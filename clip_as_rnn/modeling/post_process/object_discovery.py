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

"""Find objects."""

# pylint: disable=g-importing-member
import numpy as np
import scipy
from scipy import ndimage
from scipy.linalg import eigh
from scipy.ndimage import label
import torch
import torch.nn.functional as F


def ncut(
    feats,
    dims,
    scales,
    init_image_size,
    tau=0,
    eps=1e-5,
    no_binary_graph=False,
):
  """Implementation of NCut Method.

  Args:
    feats: the pixel/patche features of an image
    dims: dimension of the map from which the features are used
    scales: from image to map scale
    init_image_size: size of the image
    tau: thresold for graph construction
    eps: graph edge weight
    no_binary_graph: ablation study for using similarity score as graph
      edge weight
  Returns:
    TODO
  """
  feats = feats[0, 1:, :]
  feats = F.normalize(feats, p=2)
  a = feats @ feats.transpose(1, 0)
  a = a.cpu().numpy()
  if no_binary_graph:
    a[a < tau] = eps
  else:
    a = a > tau
    a = np.where(a.astype(float) == 0, eps, a)
  d_i = np.sum(a, axis=1)
  d = np.diag(d_i)

  # Print second and third smallest eigenvector
  _, eigenvectors = eigh(d - a, d, subset_by_index=[1, 2])
  eigenvec = np.copy(eigenvectors[:, 0])

  # Using average point to compute bipartition
  second_smallest_vec = eigenvectors[:, 0]
  avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
  bipartition = second_smallest_vec > avg

  seed = np.argmax(np.abs(second_smallest_vec))

  if bipartition[seed] != 1:
    eigenvec = eigenvec * -1
    bipartition = np.logical_not(bipartition)
  bipartition = bipartition.reshape(dims).astype(float)

  # predict BBox
  # We only extract the principal object BBox
  pred, _, objects, cc = detect_box(
      bipartition,
      seed,
      dims,
      scales=scales,
      initial_im_size=init_image_size[1:],
  )
  mask = np.zeros(dims)
  mask[cc[0], cc[1]] = 1

  return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)


def grad_obj_discover_on_attn(attn, gradcam, dims, topk=1, threshold=0.6):
  """Get the gradcam and attn map, then find the seed, then use LOST algorithm to find the potential points.

  Args:
      attn: attention map from ViT averaged across all heads, shape: [1,
        (1+num_patches), (1+num_patches)].
      gradcam: gradcam map from ViT, shape: [1, 1, H, W].
      dims:
      topk:
      threshold:
  Returns:
      th_attn:
  """

  w_featmap, h_featmap = dims
  # nh = attn.shape[1]
  attn = attn.squeeze()

  seeds = torch.argsort(gradcam.flatten(), descending=True)[:topk]

  # We keep only the output patch attention
  # Get the attentions corresponding to [CLS] token
  patch_attn = attn[1:, 1:]
  topk_attn = patch_attn[seeds]
  nh = topk_attn.shape[0]
  # attentions = attn[0, :, 0, 1:].reshape(nh, -1)

  # we keep only a certain percentage of the mass
  val, idx = torch.sort(topk_attn)
  val /= torch.sum(val, dim=1, keepdim=True)
  cumval = torch.cumsum(val, dim=1)
  th_attn = cumval > (1 - threshold)
  idx2 = torch.argsort(idx)
  for h in range(nh):
    th_attn[h] = th_attn[h][idx2[h]]
  th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
  th_attn = th_attn.sum(0)
  th_attn[th_attn > 1] = 1
  return th_attn[None, None]


def grad_obj_discover(feats, gradcam, dims):
  """Using gradient heatmap to find the seed, then use LOST algorithm to find the potential points.

  Args:
      feats: the pixel/patche features of an image. Shape: [1, HW, C]
      gradcam: the grad cam map
      dims: dimension of the map from which the features are used

  Returns:
      pred: box predictions
      A: binary affinity matrix
      scores: lowest degree scores for all patches
      seed: selected patch corresponding to an object
  """
  # Compute the similarity
  a = (feats @ feats.transpose(1, 2)).squeeze()

  # Compute the inverse degree centrality measure per patch
  # sorted_patches, scores = patch_scoring(a)

  # Select the initial seed
  # seed = sorted_patches[0]
  seed = gradcam.argmax()
  mask = a[seed]
  mask = mask.view(1, 1, *dims)

  return mask


def lost(feats, dims, scales, init_image_size, k_patches=100):
  """Implementation of LOST method.

  Args:
      feats: the pixel/patche features of an image. Shape: [1, C, H, W]
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      k_patches: number of k patches retrieved that are compared to the seed
          at seed expansion.
  Returns:
      pred: box predictions
      A: binary affinity matrix
      scores: lowest degree scores for all patches
      seed: selected patch corresponding to an object
  """
  # Compute the similarity
  feats = feats.flatten(2).transpose(1, 2)
  a = (feats @ feats.transpose(1, 2)).squeeze()

  # Compute the inverse degree centrality measure per patch
  sorted_patches, _ = patch_scoring(a)

  # Select the initial seed
  seed = sorted_patches[0]

  # Seed expansion
  potentials = sorted_patches[:k_patches]
  similars = potentials[a[seed, potentials] > 0.0]
  m = torch.sum(a[similars, :], dim=0)

  # Box extraction
  _, _, _, mask = detect_box(
      m, seed, dims, scales=scales, initial_im_size=init_image_size[1:]
  )

  return mask
  # return np.asarray(bbox), A, scores, seed


def patch_scoring(m, threshold=0.0):
  """Patch scoring based on the inverse degree."""
  # Cloning important
  a = m.clone()

  # Zero diagonal
  a.fill_diagonal_(0)

  # Make sure symmetric and non nul
  a[a < 0] = 0
  # C = A + A.t()

  # Sort pixels by inverse degree
  cent = -torch.sum(a > threshold, dim=1).type(torch.float32)
  sel = torch.argsort(cent, descending=True)

  return sel, cent


def detect_box(
    bipartition,
    seed,
    dims,
    initial_im_size=None,
    scales=None,
    principle_object=True,
):
  """Extract a box corresponding to the seed patch."""

  # Among connected components extract from the affinity matrix, select the one
  # corresponding to the seed patch.

  # w_featmap, h_featmap = dims
  objects, _ = ndimage.label(bipartition)
  cc = objects[np.unravel_index(seed, dims)]

  if principle_object:
    mask = np.where(objects == cc)
    # Add +1 because excluded max
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1
    # Rescale to image size
    r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
    r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    # Check not out of image size (used when padding)
    if initial_im_size:
      pred[2] = min(pred[2], initial_im_size[1])
      pred[3] = min(pred[3], initial_im_size[0])

    # Coordinate predictions for the feature space
    # Axis different then in image space
    pred_feats = [ymin, xmin, ymax, xmax]

    return pred, pred_feats, objects, mask
  else:
    raise NotImplementedError


# This function is modified from
# https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
# Ref: https://github.com/facebookresearch/dino.
def dino_seg(attn, dims, patch_size, head=0):
  """Extraction of boxes based on the DINO segmentation method proposed in DINO."""
  w_featmap, h_featmap = dims
  nh = attn.shape[1]
  official_th = 0.6

  # We keep only the output patch attention
  # Get the attentions corresponding to [CLS] token
  attentions = attn[0, :, 0, 1:].reshape(nh, -1)

  # we keep only a certain percentage of the mass
  val, idx = torch.sort(attentions)
  val /= torch.sum(val, dim=1, keepdim=True)
  cumval = torch.cumsum(val, dim=1)
  th_attn = cumval > (1 - official_th)
  idx2 = torch.argsort(idx)
  for h in range(nh):
    th_attn[h] = th_attn[h][idx2[h]]
  th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()

  # Connected components
  labeled_array, _ = scipy.ndimage.label(th_attn[head].cpu().numpy())

  # Find the biggest component
  size_components = [
      np.sum(labeled_array == c) for c in range(np.max(labeled_array))
  ]

  if len(size_components) > 1:
    # Select the biggest component avoiding component 0 corresponding
    # to background
    biggest_component = np.argmax(size_components[1:]) + 1
  else:
    # Cases of a single component
    biggest_component = 0

  # Mask corresponding to connected component
  mask = np.where(labeled_array == biggest_component)

  # Add +1 because excluded max
  ymin, ymax = min(mask[0]), max(mask[0]) + 1
  xmin, xmax = min(mask[1]), max(mask[1]) + 1

  # Rescale to image
  r_xmin, r_xmax = xmin * patch_size, xmax * patch_size
  r_ymin, r_ymax = ymin * patch_size, ymax * patch_size
  pred = [r_xmin, r_ymin, r_xmax, r_ymax]

  return pred


def get_feats(feat_out, shape):
  # Batch size, Number of heads, Number of tokens
  nb_im, nh, nb_tokens = shape[0:3]
  qkv = (
      feat_out["qkv"]
      .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
      .permute(2, 0, 3, 1, 4)
  )
  k = qkv[1]
  k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
  return k


def get_instances(masks, return_largest=False):
  return [
      get_instances_single(m[None], return_largest=return_largest)
      for m in masks
  ]


def get_instances_single(mask, return_largest=False):
  """Get the mask of a single instance."""
  labeled_array, _ = label(mask.cpu().numpy())
  instances = np.concatenate(
      [labeled_array == c for c in range(np.max(labeled_array) + 1)], axis=0
  )
  if return_largest:
    size_components = np.sum(instances, axis=(1, 2))
    if len(size_components) > 1:
      # Select the biggest component avoiding component 0 corresponding
      # to background
      biggest_component = np.argmax(size_components[1:]) + 1
    else:
      # Cases of a single component
      biggest_component = 0
    # Mask corresponding to connected component
    return torch.from_numpy(labeled_array == biggest_component).float()
  return torch.from_numpy(instances[1:]).float()
