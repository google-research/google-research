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

"""Model that process images by extracting only "important" patches."""

import enum
import functools
from typing import Any, Dict, Optional, Tuple

import chex
import einops
import flax.deprecated.nn as nn
import jax
import jax.numpy as jnp
from lib import utils
from lib.layers import transformer
from lib.ops import perturbed_topk
from lib.ops import topk


class SelectionMethod(str, enum.Enum):
  SINKHORN_TOPK = "topk"
  PERTURBED_TOPK = "perturbed-topk"
  HARD_TOPK = "hard-topk"
  RANDOM = "random"


class AggregationMethod(str, enum.Enum):
  TRANSFORMER = "transformer"
  MEANPOOLING = "meanpooling"
  MAXPOOLING = "maxpooling"
  SUM_LAYERNORM = "sum-layernorm"


class SqueezeExciteLayer(nn.Module):
  """Squeeze and Exite layer from https://arxiv.org/abs/1709.01507."""

  def apply(self, x, reduction = 16):
    num_channels = x.shape[-1]
    y = x.mean(axis=(1, 2))
    y = nn.Dense(y, features=num_channels // reduction, bias=False)
    y = nn.relu(y)
    y = nn.Dense(y, features=num_channels, bias=False)
    y = nn.sigmoid(y)
    return x * y[:, None, None, :]


class Scorer(nn.Module):
  """Scorer function."""

  def apply(self, x, use_squeeze_excite = False):
    x = nn.Conv(x, features=8, kernel_size=(3, 3), padding="VALID")
    x = nn.relu(x)
    x = nn.Conv(x, features=16, kernel_size=(3, 3), padding="VALID")
    x = nn.relu(x)
    if use_squeeze_excite:
      x = SqueezeExciteLayer(x)
    x = nn.Conv(x, features=32, kernel_size=(3, 3), padding="VALID")
    x = nn.relu(x)
    if use_squeeze_excite:
      x = SqueezeExciteLayer(x)
    x = nn.Conv(x, features=1, kernel_size=(3, 3), padding="VALID")
    scores = nn.max_pool(x, window_shape=(8, 8), strides=(8, 8))[Ellipsis, 0]
    return scores

  @classmethod
  def compute_output_size(cls, height, width):
    return ((height - 8) // 8, (width - 8) // 8)


class PatchNet(nn.Module):
  """Model that process images by extracting only "important" patches."""

  def apply(self,
            x,
            *,
            patch_size,
            k,
            downscale,
            scorer_has_se,
            normalization_str = "identity",
            selection_method,
            selection_method_kwargs = None,
            selection_method_inference = None,
            patch_dropout = 0.,
            hard_topk_probability = 0.,
            random_patch_probability = 0.,
            use_iterative_extraction,
            append_position_to_input,
            feature_network,
            aggregation_method,
            aggregation_method_kwargs = None,
            train):
    """Process a high resolution image by selecting a subset of useful patches.

    This model processes the input as follow:
    1. Compute scores per patch on a downscaled version of the input.
    2. Select "important" patches using sampling or top-k methods.
    3. Extract the patches from the high-resolution image.
    4. Compute representation vector for each patch with a feature network.
    5. Aggregate the patch representation to obtain an image representation.

    Args:
      x: Input tensor of shape (batch, height, witdh, channels).
      patch_size: Size of the (squared) patches to extract.
      k: Number of patches to extract per image.
      downscale: Downscale multiplier for the input of the scorer network.
      scorer_has_se: Whether scorer network has Squeeze-excite layers.
      normalization_str: String specifying the normalization of the scores.
      selection_method: Method that selects which patches should be extracted,
        based on their scores. Either returns indices (hard selection) or
        indicators vectors (which could yield interpolated patches).
      selection_method_kwargs: Keyword args for the selection_method.
      selection_method_inference: Selection method used at inference.
      patch_dropout: Probability to replace a patch by 0 values.
      hard_topk_probability: Probability to use the true topk on the scores to
        select the patches. This operation has no gradient so scorer's weights
        won't be trained.
      random_patch_probability: Probability to replace each patch by a random
        patch in the image during training.
      use_iterative_extraction: If True, uses a for loop instead of patch
        indexing for memory efficiency.
      append_position_to_input: Append normalized (height, width) position to
        the channels of the input.
      feature_network: Network to be applied on each patch individually to
        obtain patch representation vectors.
      aggregation_method: Method to aggregate the representations of the k
        patches of each image to obtain the image representation.
      aggregation_method_kwargs: Keywords arguments for aggregation_method.
      train: If the model is being trained. Disable dropout otherwise.

    Returns:
      A representation vector for each image in the batch.
    """
    selection_method = SelectionMethod(selection_method)
    aggregation_method = AggregationMethod(aggregation_method)
    if selection_method_inference:
      selection_method_inference = SelectionMethod(selection_method_inference)

    selection_method_kwargs = selection_method_kwargs or {}
    aggregation_method_kwargs = aggregation_method_kwargs or {}

    stats = {}

    # Compute new dimension of the scoring image.
    b, h, w, c = x.shape
    scoring_shape = (b, h // downscale, w // downscale, c)

    # === Compute the scores with a small CNN.
    if selection_method == SelectionMethod.RANDOM:
      scores_h, scores_w = Scorer.compute_output_size(h // downscale,
                                                      w // downscale)
      num_patches = scores_h * scores_w
    else:
      # Downscale input to run scorer on.
      scoring_x = jax.image.resize(x, scoring_shape, method="bilinear")
      scores = Scorer(scoring_x, use_squeeze_excite=scorer_has_se,
                      name="scorer")
      flatten_scores = einops.rearrange(scores, "b h w -> b (h w)")
      num_patches = flatten_scores.shape[-1]
      scores_h, scores_w = scores.shape[1:3]

      # Compute entropy before normalization
      prob_scores = jax.nn.softmax(flatten_scores)
      stats["entropy_before_normalization"] = jax.scipy.special.entr(
          prob_scores).sum(axis=1).mean(axis=0)

      # Normalize the flatten scores
      normalization_fn = create_normalization_fn(normalization_str)
      flatten_scores = normalization_fn(flatten_scores)
      scores = flatten_scores.reshape(scores.shape)
      stats["scores"] = scores[Ellipsis, None]

    # Concatenate height and width position to the input channels.
    if append_position_to_input:
      coords = utils.create_grid([h, w], value_range=(0., 1.))
      x = jnp.concatenate([x, coords[jnp.newaxis, Ellipsis].repeat(b, axis=0)],
                          axis=-1)
      c += 2

    # Overwrite the selection method at inference
    if selection_method_inference and not train:
      selection_method = selection_method_inference

    # === Patch selection

    # Select the patches by sampling or top-k. Some methods returns the indices
    # of the selected patches, other methods return indicator vectors.
    extract_by_indices = selection_method in [SelectionMethod.HARD_TOPK,
                                              SelectionMethod.RANDOM]
    if selection_method is SelectionMethod.SINKHORN_TOPK:
      indicators = select_patches_sinkhorn_topk(
          flatten_scores, k=k, **selection_method_kwargs)
    elif selection_method is SelectionMethod.PERTURBED_TOPK:
      sigma = selection_method_kwargs["sigma"]
      num_samples = selection_method_kwargs["num_samples"]
      sigma *= self.state("sigma_mutiplier", shape=(),
                          initializer=nn.initializers.ones).value
      stats["sigma"] = sigma
      indicators = select_patches_perturbed_topk(
          flatten_scores, k=k, sigma=sigma, num_samples=num_samples)
    elif selection_method is SelectionMethod.HARD_TOPK:
      indices = select_patches_hard_topk(flatten_scores, k=k)
    elif selection_method is SelectionMethod.RANDOM:
      batch_random_indices_fn = jax.vmap(functools.partial(
          jax.random.choice, a=num_patches, shape=(k,), replace=False))
      indices = batch_random_indices_fn(jax.random.split(nn.make_rng(), b))

    # Compute scores entropy for regularization
    if selection_method not in [SelectionMethod.RANDOM]:
      prob_scores = flatten_scores
      # Normalize the scores if it is not already done.
      if "softmax" not in normalization_str:
        prob_scores = jax.nn.softmax(prob_scores)
      stats["entropy"] = jax.scipy.special.entr(
          prob_scores).sum(axis=1).mean(axis=0)

    # Randomly use hard topk at training.
    if (train and
        hard_topk_probability > 0 and
        selection_method not in [SelectionMethod.HARD_TOPK,
                                 SelectionMethod.RANDOM]):
      true_indices = select_patches_hard_topk(flatten_scores, k=k)
      random_values = jax.random.uniform(nn.make_rng(), (b,))
      use_hard = random_values < hard_topk_probability
      if extract_by_indices:
        indices = jnp.where(use_hard[:, None], true_indices, indices)
      else:
        true_indicators = make_indicators(true_indices, num_patches)
        indicators = jnp.where(use_hard[:, None, None],
                               true_indicators, indicators)

    # Sample some random patches during training with random_patch_probability.
    if (train and
        random_patch_probability > 0 and
        selection_method is not SelectionMethod.RANDOM):
      single_random_patches = functools.partial(
          jax.random.choice, a=num_patches, shape=(k,), replace=False)
      random_indices = jax.vmap(single_random_patches)(
          jax.random.split(nn.make_rng(), b))
      random_values = jax.random.uniform(nn.make_rng(), (b, k))
      use_random = random_values < random_patch_probability
      if extract_by_indices:
        indices = jnp.where(use_random, random_indices, indices)
      else:
        random_indicators = make_indicators(random_indices,
                                            num_patches)
        indicators = jnp.where(use_random[:, None, :],
                               random_indicators, indicators)

    # === Patch extraction
    if extract_by_indices:
      patches = extract_patches_from_indices(
          x, indices, patch_size=patch_size,
          grid_shape=(scores_h, scores_w))
      indicators = make_indicators(indices, num_patches)
    else:
      patches = extract_patches_from_indicators(
          x, indicators, patch_size, grid_shape=(scores_h, scores_w),
          iterative=use_iterative_extraction, patch_dropout=patch_dropout,
          train=train)

    chex.assert_shape(patches, (b, k, patch_size, patch_size, c))

    stats["extracted_patches"] = einops.rearrange(patches,
                                                  "b k i j c -> b i (k j) c")
    # Remove position channels for plotting.
    if append_position_to_input:
      stats["extracted_patches"] = (stats["extracted_patches"][Ellipsis, :-2])

    # === Compute patch features
    flatten_patches = einops.rearrange(patches, "b k i j c -> (b k) i j c")
    representations = feature_network(flatten_patches, train=train)
    if representations.ndim > 2:
      collapse_axis = tuple(range(1, representations.ndim - 1))
      representations = representations.mean(axis=collapse_axis)
    representations = einops.rearrange(representations,
                                       "(b k) d -> b k d", k=k)

    stats["patch_representations"] = representations

    # === Aggregate the k patches

    # - for sampling we are forced to take an expectation
    # - for topk we have multiple options: mean, max, transformer.
    if aggregation_method is AggregationMethod.TRANSFORMER:
      patch_pos_encoding = nn.Dense(einops.rearrange(indicators,
                                                     "b d k -> b k d"),
                                    features=representations.shape[-1])

      chex.assert_equal_shape([representations, patch_pos_encoding])
      representations += patch_pos_encoding
      representations = transformer.Transformer(
          representations, **aggregation_method_kwargs, is_training=train)

    elif aggregation_method is AggregationMethod.MEANPOOLING:
      representations = representations.mean(axis=1)
    elif aggregation_method is AggregationMethod.MAXPOOLING:
      representations = representations.max(axis=1)
    elif aggregation_method is AggregationMethod.SUM_LAYERNORM:
      representations = representations.sum(axis=1)
      representations = nn.LayerNorm(representations)

    representations = nn.Dense(representations,
                               features=representations.shape[-1],
                               name="classification_dense1")
    representations = nn.swish(representations)

    return representations, stats


def select_patches_perturbed_topk(flatten_scores,
                                  sigma,
                                  *,
                                  k,
                                  num_samples = 1000):
  """Select patches using a differentiable top-k based on perturbation.

  Uses https://q-berthet.github.io/papers/BerBloTeb20.pdf,
  see off_the_grid.lib.ops.perturbed_topk for more info.

  Args:
    flatten_scores: The flatten scores of shape (batch, num_patches).
    sigma: Standard deviation of the noise.
    k: The number of patches to extract.
    num_samples: Number of noisy inputs used to compute the output expectation.

  Returns:
    Indicator vectors of the selected patches (batch, num_patches, k).
  """
  batch_size = flatten_scores.shape[0]

  batch_topk_fn = jax.vmap(
      functools.partial(perturbed_topk.perturbed_sorted_topk_indicators,
                        num_samples=num_samples,
                        sigma=sigma,
                        k=k))

  rng_keys = jax.random.split(nn.make_rng(), batch_size)
  indicators = batch_topk_fn(flatten_scores, rng_keys)
  topk_indicators_flatten = einops.rearrange(indicators, "b k d -> b d k")
  return topk_indicators_flatten


def select_patches_sinkhorn_topk(flatten_scores,
                                 *,
                                 k,
                                 epsilon,
                                 num_iterations):
  """Select patches using a differentiable top-k based on sinkhorn.

  Uses https://arxiv.org/abs/2002.06504, see lib.ops.topk for more
  info.

  Args:
    flatten_scores: The flatten scores of shape (batch, num_patches).
    k: The number of patches to extract.
    epsilon: Temperature of sinkhorn.
    num_iterations: Number of iterations of sinkhorn.

  Returns:
    Indicator vectors of the selected patches (batch, num_patches, k).
  """
  batch_topk_fn = jax.vmap(
      functools.partial(topk.differentiable_smooth_sorted_top_k,
                        k=k, epsilon=epsilon, num_iterations=num_iterations))

  topk_indicators_flatten = batch_topk_fn(flatten_scores)
  return topk_indicators_flatten


def select_patches_hard_topk(flatten_scores, *, k):
  """Return the indices of the k patches with highest `flatten_scores`."""
  indices = jax.lax.top_k(flatten_scores, k)[1]
  # Naive sorting commented below was ~10% slower on TPU.
  # indices = jnp.argsort(flatten_scores)[:, -k:]
  return indices


def extract_patches_from_indicators(x,
                                    indicators,
                                    patch_size,
                                    patch_dropout,
                                    grid_shape,
                                    train,
                                    iterative = False):
  """Extract patches from a batch of images.

  Args:
    x: The batch of images of shape (batch, height, width, channels).
    indicators: The one hot indicators of shape (batch, num_patches, k).
    patch_size: The size of the (squared) patches to extract.
    patch_dropout: Probability to replace a patch by 0 values.
    grid_shape: Pair of height, width of the disposition of the num_patches
      patches.
    train: If the model is being trained. Disable dropout if not.
    iterative: If True, etracts the patches with a for loop rather than
      instanciating the "all patches" tensor and extracting by dotproduct with
      indicators. `iterative` is more memory efficient.

  Returns:
    The patches extracted from x with shape
      (batch, k, patch_size, patch_size, channels).

  """
  batch_size, height, width, channels = x.shape
  scores_h, scores_w = grid_shape
  k = indicators.shape[-1]
  indicators = einops.rearrange(indicators, "b (h w) k -> b k h w",
                                h=scores_h, w=scores_w)

  scale_height = height // scores_h
  scale_width = width // scores_w
  padded_height = scale_height * scores_h + patch_size - 1
  padded_width = scale_width * scores_w + patch_size - 1
  top_pad = (patch_size - scale_height) // 2
  left_pad = (patch_size - scale_width) // 2
  bottom_pad = padded_height - top_pad - height
  right_pad = padded_width - left_pad - width

  # TODO(jbcdnr): assert padding is positive.

  padded_x = jnp.pad(x,
                     [(0, 0),
                      (top_pad, bottom_pad),
                      (left_pad, right_pad),
                      (0, 0)])

  # Extract the patches. Iterative fits better in memory as it does not
  # instanciate the "all patches" tensor but iterate over them to compute the
  # weighted sum with the indicator variables from topk.
  if not iterative:
    assert patch_dropout == 0., "Patch dropout not implemented."
    patches = utils.extract_images_patches(
        padded_x,
        window_size=(patch_size, patch_size),
        stride=(scale_height, scale_width))

    shape = (batch_size, scores_h, scores_w, patch_size, patch_size, channels)
    chex.assert_shape(patches, shape)

    patches = jnp.einsum("b k h w, b h w i j c -> b k i j c",
                         indicators, patches)

  else:
    mask = jnp.ones((batch_size, scores_h, scores_w))
    mask = nn.dropout(mask, patch_dropout, deterministic=not train)

    def accumulate_patches(acc, index_i_j):
      i, j = index_i_j
      patch = jax.lax.dynamic_slice(
          padded_x,
          (0, i * scale_height, j * scale_width, 0),
          (batch_size, patch_size, patch_size, channels))
      weights = indicators[:, :, i, j]

      is_masked = mask[:, i, j]
      weighted_patch = jnp.einsum("b, bk, bijc -> bkijc",
                                  is_masked, weights, patch)
      chex.assert_equal_shape([acc, weighted_patch])
      acc += weighted_patch
      return acc, None

    indices = jnp.stack(
        jnp.meshgrid(jnp.arange(scores_h), jnp.arange(scores_w), indexing="ij"),
        axis=-1)
    indices = indices.reshape((-1, 2))
    init_patches = jnp.zeros((batch_size, k, patch_size, patch_size, channels))
    patches, _ = jax.lax.scan(accumulate_patches, init_patches, indices)

  return patches


def extract_patches_from_indices(x,
                                 indices,
                                 patch_size,
                                 grid_shape):
  """Extract patches from a batch of images.

  Args:
    x: The batch of images of shape (batch, height, width, channels).
    indices: The indices of the flatten patches to extract of shape
      (batch, k).
    patch_size: The size of the (squared) patches to extract.
    grid_shape: Pair of height, width of the disposition of the num_patches
      patches.

  Returns:
    The patches extracted from x with shape
      (batch, k, patch_size, patch_size, channels).

  """
  _, height, width, _ = x.shape
  scores_h, scores_w = grid_shape

  scale_height = height // scores_h
  scale_width = width // scores_w

  h_padding = (patch_size - scale_height) // 2
  w_padding = (patch_size - scale_width) // 2

  height_indices = (indices // scores_w) * scale_height
  width_indices = (indices % scores_w) * scale_width

  padded_x = jnp.pad(x,
                     [(0, 0),
                      (h_padding, h_padding),
                      (w_padding, w_padding),
                      (0, 0)])

  @jax.vmap
  @functools.partial(jax.vmap, in_axes=(None, 0, 0))
  def patch(image, i, j):
    # Equivalent to image[i:i+patch_size, j:j+patch_size, :]
    return jax.lax.dynamic_slice(image,
                                 (i, j, 0),
                                 (patch_size, patch_size, image.shape[-1]))

  patches = patch(padded_x, height_indices, width_indices)
  return patches


def _get_available_normalization_fns():
  """Defines functions available in normalization function strings."""
  def smoothing(s):
    def smoothing_fn(x):
      uniform = 1. / x.shape[-1]
      x = x * (1 - s) + uniform * s
      return x
    return smoothing_fn

  def zeroone(scores):
    scores -= scores.min(axis=1, keepdims=True)
    scores /= scores.max(axis=1, keepdims=True)
    return scores

  def zerooneeps(eps):
    def zerooneeps_fn(scores):
      scores_min = scores.min(axis=-1, keepdims=True)
      scores_max = scores.max(axis=-1, keepdims=True)
      return (scores - scores_min) / (scores_max - scores_min + eps)
    return zerooneeps_fn

  return dict(identity=lambda x: x,
              softmax=jax.nn.softmax,
              smoothing=smoothing,
              zeroone=zeroone,
              sigmoid=jax.nn.sigmoid,
              layernorm=nn.LayerNorm,
              zerooneeps=zerooneeps)


def create_normalization_fn(fn_str):
  """Create a normalization function from a string representation.

  The syntax is similar to data preprocessing strings. Functions are specified
  by name with parameters and chained with | character. Available functions are
  specified in _get_available_normalization_fns.

  Example:
    "softmax|smoothing(0.1)" will give `smoothing(softamax(x), 0.1)`.

  Args:
    fn_str: The function definition string.

  Returns:
    The function specified by the string.
  """
  functions = [eval(f, _get_available_normalization_fns())  # pylint:disable=eval-used
               for f in fn_str.split("|") if f]
  def chain(x):
    for f in functions:
      x = f(x)
    return x
  return chain


@jax.vmap
def batch_gather(x, indices):
  return x[indices, Ellipsis]


@functools.partial(jax.vmap, in_axes=[0, None])
@functools.partial(jax.vmap, in_axes=[0, None], out_axes=1)
def make_indicators(indices, num_classes):
  """Create one hot associated to indices.

  Args:
    indices: Tensor of indices of dimension (batch, k).
    num_classes: The number of classes to represent in the one hot vectors.

  Returns:
    The one hot indicators associated to indices of shape
    (batch, num_classes, k).
  """
  return jax.nn.one_hot(indices, num_classes)
