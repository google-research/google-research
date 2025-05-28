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

"""Generalizable patch-based neural rendering model with three transformers and rotation canonicalization."""

from einshape import jax_einshape as einshape
from flax import linen as nn
from jax import random
import jax.numpy as jnp

from gen_patch_neural_rendering.src.models import efficient_conv
from gen_patch_neural_rendering.src.models import projector
from gen_patch_neural_rendering.src.models import transformer
from gen_patch_neural_rendering.src.utils import config_utils
from gen_patch_neural_rendering.src.utils import lf_utils
from gen_patch_neural_rendering.src.utils import model_utils


class GPNR(nn.Module):
  """Generalized Patch-based Neural Renderer."""
  render_config: config_utils.RenderParams
  encoding_config: config_utils.EncodingParams
  lf_config: config_utils.LightFieldParams
  epipolar_transformer_config: config_utils.TransformerParams
  view_transformer_config: config_utils.TransformerParams
  epipolar_config: config_utils.EpipolarParams
  return_attn: bool

  def setup(self):
    """Setup Model."""

    precision = self.epipolar_config.precision

    #--------------------------------------------------------------------
    # Light Field Object
    self.lightfield = lf_utils.get_lightfield_obj(self.lf_config)

    #--------------------------------------------------------------------
    # Projector
    self.projector = projector.RayProjector(self.epipolar_config)

    #--------------------------------------------------------------------
    # Transformers
    self.vision_transfromer = transformer.SelfAttentionTransformer(
        self.epipolar_transformer_config)
    self.epipolar_transformer = transformer.SelfAttentionTransformer(
        self.epipolar_transformer_config)
    self.view_transformer = transformer.SelfAttentionTransformer(
        self.view_transformer_config)

    #--------------------------------------------------------------------
    # Layer to predict the attention weight for each point on epipolar line
    self.epipolar_correspondence = nn.DenseGeneral(1)
    self.view_correspondence = nn.DenseGeneral(1)

    #--------------------------------------------------------------------
    # Color prediction (Optional)
    self.rgb_dense = nn.Dense(
        self.render_config.num_rgb_channels, precision=precision)

    #--------------------------------------------------------------------
    # Layer to transform key and query to same dim for conatenation
    self.key_transform = nn.DenseGeneral(
        self.epipolar_transformer_config.qkv_params, precision=precision)
    self.query_transform = nn.DenseGeneral(
        self.epipolar_transformer_config.qkv_params, precision=precision)

    # Layer to transform key and query to same dim for conatenation
    self.key_transform2 = nn.DenseGeneral(
        self.view_transformer_config.qkv_params)
    self.query_transform2 = nn.DenseGeneral(
        self.view_transformer_config.qkv_params, precision=precision)

    self.conv_layer = efficient_conv.SplitConvModel(
        features=self.epipolar_config.conv_feature_dim,
        kernel_size=self.epipolar_config.patch_size,
    )
    self.feature_activation = nn.elu

    self.mean = einshape("x->111x", jnp.array([0.485, 0.456, 0.406]))
    self.std = einshape("x->111x", jnp.array([0.229, 0.224, 0.225]))

  def _get_query(self, rays):
    """Get the light field encoding for the target rays.

    Args:
      rays: data_types.Batch

    Returns:
      q_samples : light field query samples
      q_samples_enc: encoded light field query samples
      q_mask: mask for query
    """
    q_samples, q_samples_enc, q_mask = self.lightfield.get_lf_encoding(rays)
    return q_samples, q_samples_enc, q_mask

  def _get_key(
      self,
      projected_rays,
      projected_rgb_and_feat,
      wcoords,
  ):
    """Get the light field encoding for the rays through epipolar projections.

    Add the world coordinates of the projected point along the query ray.
    Args:
      projected_rays: data_types.Rays
      projected_rgb_and_feat: color and (or) features at the epipolar
        projections.
      wcoords: The world coordinate for points along the query ray.

    Returns:
      k_samples: key
      k_samples_enc: encoded keys
      k_mask: optional mask for keys
      learned_embedding: learned embedding for reference view
    """
    k_samples, k_samples_enc, k_mask = self.lightfield.get_lf_encoding(
        projected_rays)

    # Add the 3d woord coordinates.
    wcoords_enc = model_utils.posenc(
        wcoords,
        self.encoding_config.min_deg_point,
        self.encoding_config.max_deg_point,
    )
    # wcoords_enc is of shape (B, P, EmbeddingDim). To concatenate to keys we
    # need to broadcast it to (B, N, P, EmbeddingDim)
    wcoords_enc = jnp.broadcast_to(
        wcoords_enc[:, None],
        k_samples_enc.shape[:-1] + (wcoords_enc.shape[-1],))

    input_k = jnp.concatenate(
        [k_samples_enc, wcoords_enc, projected_rgb_and_feat], axis=-1)

    return k_samples, input_k, k_mask

  def _get_pixel_projection(self, projected_coordinates, ref_images):
    """Get the rgb or features from the image at the projected coordinates."""
    ref_features = self.feature_activation(self.conv_layer(ref_images))

    projected_features = self.projector.get_interpolated_rgb(
        projected_coordinates, ref_features)
    # Also project rgb
    del ref_features
    projected_rgb = self.projector.get_interpolated_rgb(projected_coordinates,
                                                        ref_images)
    projected_features = jnp.concatenate([projected_features, projected_rgb],
                                         axis=-1)

    return projected_features

  def _get_avg_features(self, input_q, input_k, randomized):
    """Function that aggregate feature over the projection on the epipolar line.

    Args:
      input_q: query, with shape (bs, 1, q_feature_dim)
      input_k: key, with shape (bs, near_cam, projections, k_feature_dim)
      randomized: True during training

    Returns:
      out: Average features (bs, near_cam, _)
      epipolar_attn_weights: attention weights
    """
    # Change shape of query from (BS, Q) -> (BS, NearCam, 1, Q)
    input_q = jnp.tile(input_q[:, None, None], (1, input_k.shape[1], 1, 1))
    input_q = self.query_transform(input_q)
    input_k = self.key_transform(input_k)

    # Concatenate the query to the keys
    input_k = jnp.concatenate([input_q, input_k], axis=-2)

    input_k = einshape("bnpc->bpnc", input_k)
    out = self.vision_transfromer(input_k, deterministic=not randomized)
    out = einshape("bpnc->bnpc", out)

    out = self.epipolar_transformer(
        out,
        deterministic=not randomized,
    )
    refined_query = out[Ellipsis, 0:1, :]  # Get refined query
    refined_key = out[Ellipsis, 1:, :]

    refined_query = jnp.tile(refined_query, (1, 1, refined_key.shape[-2], 1))
    # Predict attetion weights for averaging the key
    concat_query_key = jnp.concatenate([refined_query, refined_key], axis=-1)
    epipolar_attn_weights = self.epipolar_correspondence(concat_query_key)
    epipolar_attn_weights = nn.softmax(epipolar_attn_weights, axis=-2)
    out = (epipolar_attn_weights * refined_key).sum(-2)

    return out, epipolar_attn_weights

  def _predict_color(self, input_q, input_k, randomized):  # pylint: disable=arguments-differ
    """Function to predict the color by aggregating information from neighbouring views.

    Args:
      input_q: query, with shape (bs, 1, q_feature_dim)
      input_k: key, with shape (bs, near_cam, k_feature_dim')
      randomized: True during training.

    Returns:
      rgb: color prediction (bs, 3)
      neighbor_attn_weights: attention weights
    """
    input_q = self.query_transform2(input_q[:, None])

    input_k = self.key_transform2(input_k)
    input_q = jnp.concatenate([input_q, input_k], axis=-2)

    out = self.view_transformer(
        input_q,
        deterministic=not randomized,
    )

    refined_query = out[:, 0:1]
    refined_key = out[:, 1:]
    refined_query = jnp.tile(refined_query, (1, refined_key.shape[-2], 1))

    concat_key_query = jnp.concatenate([refined_query, refined_key], axis=-1)
    neighbor_attn_weights = self.view_correspondence(concat_key_query)
    neighbor_attn_weights = nn.softmax(neighbor_attn_weights, axis=-2)

    raw_rgb = self.rgb_dense((refined_key * neighbor_attn_weights).sum(-2))
    rgb = self.render_config.rgb_activation(raw_rgb)

    return rgb, neighbor_attn_weights

  def _get_reg_prediction(self, ref_features, epipolar_attn_weights,
                          neighbor_attn_weights):
    """Get regularization prediction."""
    ref_rgb = ref_features[Ellipsis, -3:]  # rgb concatenated at the end
    neighbor_rgb = (ref_rgb * epipolar_attn_weights).sum(-2)
    coarse_rgb = (neighbor_rgb * neighbor_attn_weights).sum(-2)
    if self.epipolar_config.normalize_ref_image:
      coarse_rgb = jnp.clip(coarse_rgb * self.std[0, 0] + self.mean[0, 0], 0, 1)
    return coarse_rgb

  def __call__(self, rng_0, rng_1, batch, randomized):
    """Generalizale Patch-Based Neural Rendering Model.

    Args:
      rng_0: jnp.ndarray, random number generator for coarse model sampling.
      rng_1: jnp.ndarray, random number generator for fine model sampling.
      batch: data batch. data_types.Batch
      randomized: bool, use randomized stratified sampling.

    Returns:
      ret: list, [(rgb, None, Optional[acc])]
    """
    del rng_1

    # Get the batch rays
    batch_rays = batch.target_view.rays

    #---------------------------------------------------------------------------------------
    # Get image height and width. To be use to clip projection
    # outside the image.
    image_height, image_width = batch.reference_views.rgb.shape[1:3]
    # The the min and max depth as intergers.
    min_depth = batch.reference_views.min_depth[0][0]
    max_depth = batch.reference_views.max_depth[0][0]

    projected_coordinates, _, wcoords = self.projector.epipolar_projection(
        rng_0,
        batch_rays,
        batch.reference_views.ref_worldtocamera,
        batch.reference_views.intrinsic_matrix,
        image_height,
        image_width,
        min_depth,
        max_depth,
    )

    #---------------------------------------------------------------------------------------
    # Add a [0, 0, 0, 1] row to the ref cam to world
    bottom = jnp.tile(
        jnp.array([[[0, 0, 0, 1.]]]),
        (batch.reference_views.ref_cameratoworld.shape[0], 1, 1))
    ref_cameratoworld = jnp.concatenate(
        [batch.reference_views.ref_cameratoworld[Ellipsis, :3, :4], bottom], -2)

    #--------------------------------------------------------------------------------------
    # Compute the canonical transformation.
    # get the cam to world of the target rays.
    camtoworld = jnp.linalg.inv(
        batch.reference_views.target_worldtocam)  # (1, 4, 4)
    # Get the up vector
    upv = camtoworld[Ellipsis, :3, 1]
    # Get the ray direction.
    raydir = batch_rays.directions
    rdotup = (raydir * upv).sum(-1, keepdims=True)
    orthoup = upv - rdotup * raydir
    orthoup = orthoup / (orthoup**2).sum(-1, keepdims=True)
    vec0 = jnp.cross(orthoup, raydir)
    vec0 = vec0 / (vec0**2).sum(-1, keepdims=True)
    # Stack it such that the matrix is the transpose (inverse).
    r_relative = jnp.stack([vec0, orthoup, raydir], axis=1)
    #--------------------------------------------------------------------------------------

    translation_relative = -jnp.matmul(r_relative, batch_rays.origins[Ellipsis,
                                                                      None])

    canonical_transform = jnp.concatenate([r_relative, translation_relative],
                                          axis=-1)
    bottom = jnp.tile(jnp.array([[[0, 0, 0, 1.]]]), (r_relative.shape[0], 1, 1))

    canonical_transform = jnp.concatenate([canonical_transform, bottom],
                                          axis=-2)

    wcoords = jnp.pad(wcoords, ((0, 0), (0, 0), (0, 1)), constant_values=1)
    wcoords = jnp.einsum("nij,nkj->nki", canonical_transform, wcoords)
    wcoords = wcoords[Ellipsis, :3]
    canonical_ref_cameratoworld = jnp.einsum("bxz, nzy->nbxy",
                                             canonical_transform,
                                             ref_cameratoworld)

    projected_rays = self.projector.get_near_rays_canonical(
        projected_coordinates, canonical_ref_cameratoworld,
        batch.reference_views.intrinsic_matrix)

    # Next we need to get the rgb values and the rays corresponding to these
    # projections.
    ref_images = model_utils.uint2float(batch.reference_views.rgb)
    if self.epipolar_config.normalize_ref_image:
      ref_images = (ref_images - self.mean) / self.std

    projected_rgb_and_feat = self._get_pixel_projection(projected_coordinates,
                                                        ref_images)

    batch.reference_views.rgb = None
    #----------------------------------------------------------------------------------------
    # Canonicalize the batch rays.
    # Mutiply the directions with the inverse of the target camera to world
    # matrix

    batch_rays.directions = (
        r_relative[Ellipsis, :3, :3] @ batch_rays.directions[Ellipsis, None])[Ellipsis, 0]
    batch_rays.origins = jnp.zeros_like(batch_rays.origins)

    # Get LF representation of the batch and the projected rays.
    # Below we consider the representation extracted from the batch rays as the
    # query and representation extracted from the projected rays as keys and the
    # projected rgb as the values.
    _, input_q, _ = self._get_query(batch_rays)
    _, input_k, _ = self._get_key(projected_rays, projected_rgb_and_feat,
                                  wcoords)

    #----------------------------------------------------------------------------------------
    # Append the relative orientation between the target and reference cameras.
    rel_camera_rot = einshape(
        "nxy->bnp(xy)",
        jnp.matmul(batch.reference_views.target_worldtocam[Ellipsis, :3, :3],
                   batch.reference_views.ref_cameratoworld[Ellipsis, :3, :3]),
        b=input_k.shape[0],
        p=input_k.shape[2])
    rel_camera_tran = einshape(
        "nx->bnpx", (batch.reference_views.target_worldtocam[Ellipsis, :3, -1] -
                     batch.reference_views.ref_cameratoworld[Ellipsis, :3, -1]),
        b=input_k.shape[0],
        p=input_k.shape[2])
    input_k = jnp.concatenate([input_k, rel_camera_rot, rel_camera_tran],
                              axis=-1)

    # Get the average feature over each epipolar line
    avg_projection_features, e_attn = self._get_avg_features(
        input_q, input_k, randomized=randomized)

    rgb, n_attn = self._predict_color(input_q, avg_projection_features,
                                      randomized)
    rgb_coarse = self._get_reg_prediction(projected_rgb_and_feat, e_attn,
                                          n_attn)

    ret = [(rgb_coarse, None, None)]
    ret.append((rgb, None, None))

    if self.return_attn:
      return ret, {
          "e_attn": e_attn,
          "n_attn": n_attn,
          "p_coord": projected_coordinates.swapaxes(0, 1)
      }
    else:
      return ret


def construct_model(key, example_batch, args):
  """Construct a  Generalizable Patch-Based Neural Renderig Model.

  Args:
    key: jnp.ndarray. Random number generator.
    example_batch: dict, an example of a batch of data.
    args: FLAGS class. Hyperparameters of nerf.

  Returns:
    model: nn.Model. model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  rgb_activation = getattr(nn, str(args.model.rgb_activation))
  sigma_activation = getattr(nn, str(args.model.sigma_activation))

  # Assert that rgb_activation always produces outputs in [0, 1], and
  # sigma_activation always produce non-negative outputs.
  x = jnp.exp(jnp.linspace(-90, 90, 1024))
  x = jnp.concatenate([-x[::-1], x], 0)

  rgb = rgb_activation(x)  # pylint: disable=not-callable
  if jnp.any(rgb < 0) or jnp.any(rgb > 1):
    raise NotImplementedError(
        "Choice of rgb_activation `{}` produces colors outside of [0, 1]"
        .format(args.rgb_activation))

  sigma = sigma_activation(x)  # pylint: disable=not-callable
  if jnp.any(sigma < 0):
    raise NotImplementedError(
        "Choice of sigma_activation `{}` produces negative densities".format(
            args.sigma_activation))

  # We have defined some wrapper functions to extract the relavant cofiguration
  # so are to alow for efficient reuse
  render_config = config_utils.get_render_params(args, rgb_activation,
                                                 sigma_activation)
  encoding_config = config_utils.get_encoding_params(args)
  lf_config = config_utils.get_lightfield_params(args)
  epipolar_config = config_utils.get_epipolar_params(args)
  epipolar_config.use_learned_embedding = False

  epipolar_transformer_config = config_utils.get_epipolar_transformer_params(
      args)
  view_transformer_config = config_utils.get_view_transformer_params(args)

  if epipolar_config.use_learned_embedding:
    assert epipolar_transformer_config.qkv_params == view_transformer_config.qkv_params, "Currently the learned embedding are shared so the transformers need to have same qkv dim"

  model = GPNR(
      render_config=render_config,
      encoding_config=encoding_config,
      lf_config=lf_config,
      epipolar_config=epipolar_config,
      epipolar_transformer_config=epipolar_transformer_config,
      view_transformer_config=view_transformer_config,
      return_attn=args.model.return_attn)

  key1, key2, key3 = random.split(key, num=3)

  init_variables = model.init(  # pylint: disable=no-member
      key1,
      rng_0=key2,
      rng_1=key3,
      batch=example_batch,
      randomized=args.model.randomized)

  return model, init_variables
