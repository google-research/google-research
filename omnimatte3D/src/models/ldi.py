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

"""Model to predict per-frame LDI."""

import dataclasses

from einops import rearrange
from flax import linen as nn
import jax.numpy as jnp

from omnimatte3D.src.models import unet
from omnimatte3D.src.utils import data_types
from omnimatte3D.src.utils import model_utils
from omnimatte3D.src.utils import train_utils


@dataclasses.dataclass
class LDIParams:
  """Parameters for the UNet model."""

  num_layers: int  # Number of layers in LDI.
  min_depth: float  # Min depth for scene.
  max_depth: float  # Max depth for scene.
  image_height: int
  image_width: int
  white_bkgd: bool
  use_rts: bool


def get_ldi_params(config):
  return LDIParams(
      num_layers=1,
      min_depth=config.dataset.min_depth,
      max_depth=config.dataset.max_depth,
      image_height=config.dataset.image_height,
      image_width=config.dataset.image_width,
      white_bkgd=config.dataset.white_bkgd,
      use_rts=config.model.use_rts,
  )


class LDI(nn.Module):
  """LDI prediction model."""

  ldi_params: LDIParams
  unet_params: unet.UnetConfig
  rgb_unet_params: unet.UnetConfig
  num_objects: int
  crop_projection: bool

  def get_rgb_layers(self, features, nsrc, nlayer):
    """Predict rgb layers.

    Args:
      features: feature to predict layers from.
      nsrc: number of source images. (to reshape)
      nlayer: number of layers.

    Returns:
      rgb_layers: background prediction.
      fg_layers: foreground rgb layers.
      fg_alpha: forgoround alpha layers.
    """
    rgb_layer_list = []
    fg_layer_list = []
    features = rearrange(
        features, "(b n l) h w c -> (b n) l h w c", n=nsrc, l=nlayer
    )
    for idx in range(self.ldi_params.num_layers):
      out = unet.UNet(self.rgb_unet_params, name="rgb_net_{}".format(idx))(
          features[:, -1]
      )
      rgb_layer_list.append(nn.sigmoid(out))

      multi_fg_list = []
      for midx in range(self.num_objects):
        out_fg = nn.Conv(
            3 + 1, kernel_size=(3, 3), name="fg_net_{}_{}".format(midx, idx)
        )(features[:, midx])
        multi_fg_list.append(nn.sigmoid(out_fg))

      fg_layer_list.append(jnp.stack(multi_fg_list, axis=1))

    rgb_layers = jnp.stack(rgb_layer_list, axis=1)  # (b n) l h w c
    rgb_layers = rearrange(
        rgb_layers,
        "(b n) l h w c -> b n l h w c",
        n=nsrc,
    )

    fg_layers = jnp.stack(fg_layer_list, axis=1)  # (b n) 2 l h w c
    fg_layers = rearrange(
        fg_layers,
        "(b n) l m h w c -> b n l m h w c",
        n=nsrc,
        l=1,
    )

    return rgb_layers, fg_layers[Ellipsis, :3], fg_layers[Ellipsis, 3:4]

  def get_disp_layers(self, features, nsrc, nlayer):
    features = rearrange(
        features, "(b n l) h w c -> (b n) l h w c", n=nsrc, l=nlayer
    )
    disp_layer_list = []
    for idx in range(self.ldi_params.num_layers):
      out = nn.Conv(1, kernel_size=(3, 3), name="disp_net_{}".format(idx))(
          features[:, -1]
      )
      disp_layer_list.append(nn.sigmoid(out))

    disp_layers = jnp.stack(disp_layer_list, axis=1)
    disp_layers = rearrange(
        disp_layers,
        "(b n) l h w c -> b n l h w c",
        n=nsrc,
    )
    disp_layers = model_utils.scale_disp(
        disp_layers,
        self.ldi_params.min_depth,
        self.ldi_params.max_depth,
    )
    return disp_layers

  def get_mask_layers(self, features, nsrc):
    mask_layer_list = []
    for idx in range(self.ldi_params.num_layers - 1):
      out = nn.Conv(1, kernel_size=(3, 3), name="mask_net_{}".format(idx))(
          features
      )
      mask_layer_list.append(nn.sigmoid(out))

    bg_mask = jnp.ones_like(out)
    mask_layer_list.append(bg_mask)

    mask_layers = jnp.stack(mask_layer_list, axis=1)
    mask_layers = rearrange(
        mask_layers,
        "(b n) l h w c -> b n l h w c",
        n=nsrc,
    )
    return mask_layers

  def get_layer_pred(self, batch):
    nsrc = batch["src_rgb"].shape[-4]  # (b, n , h, w ,c )

    # --------------------------------------------------------------------
    # Get mask input.
    # 2 images for the main source images.
    # 1 image from further away timestep.
    mask_input = jnp.concatenate(
        [
            batch["src_mask_layer"],
            rearrange(batch["bg_src_layer_mask"], "b l h w c -> b 1 l h w c"),
            rearrange(
                batch["bg_src_tgt_layer_mask"], "b l h w c -> b 1 l h w c"
            ),
        ],
        axis=1,
    )

    # --------------------------------------------------------------------
    # disp input.
    # 2 images for the main source images.
    # 1 image from further away timestep.
    disp_input = jnp.concatenate(
        [
            batch["src_disp_input_layer"],
            rearrange(
                batch["bg_src_layer_input_disp"], "b l h w c -> b 1 l h w c"
            ),
            rearrange(
                batch["bg_src_tgt_layer_input_disp"], "b l h w c -> b 1 l h w c"
            ),
        ],
        axis=1,
    )

    # --------------------------------------------------------------------
    # rgb input.
    # 2 images for the main source images.
    # 1 image from further away timestep.
    rgb_input = jnp.concatenate(
        [
            batch["src_rgb_input_layer"],
            rearrange(
                batch["bg_src_layer_input_rgb"], "b l h w c -> b 1 l h w c"
            ),
            rearrange(
                batch["bg_src_tgt_layer_input_rgb"], "b l h w c -> b 1 l h w c"
            ),
        ],
        axis=1,
    )
    # --------------------------------------------------------------------
    # Prepare source input.
    x = jnp.concatenate([rgb_input, disp_input, mask_input], axis=-1)
    nsrc, nlayer = x.shape[1:3]
    x = rearrange(x, "b n l h w c -> (b n l) h w c")

    # features = self.unet(x)
    features = unet.UNet(self.unet_params, name="feature_extractor")(x)

    # -------------------------------------------------------------------------
    # Get layer predictions.
    # Get rgb layers
    rgb_layers, fg_layers, fg_alphas = self.get_rgb_layers(
        features, nsrc, nlayer
    )

    # Get disp layers
    disp_layers = self.get_disp_layers(features, nsrc, nlayer)

    # Get mask layers.
    mask_layers = jnp.ones_like(disp_layers)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # For the LDI data type.
    # Source 1 and 2 are to project to target.

    src_ldi = data_types.LDI(
        rgb_layers=rgb_layers[:, [0, 1]],
        fg_layers=fg_layers[:, [0, 1]],
        fg_alphas=fg_alphas[:, [0, 1]],
        disp_layers=disp_layers[:, [0, 1]],
        mask_layers=mask_layers[:, [0, 1]],
        camtoworld=batch["src_camtoworld"],
        intrinsic=batch["intrinsic"],
    )

    # source 0 and source T''
    # only store bg stuff.
    bg_ldi = data_types.LDI(
        rgb_layers=rgb_layers[:, [0, 2]],
        fg_layers=fg_layers[:, [0, 2]],
        fg_alphas=fg_alphas[:, [0, 2]],
        disp_layers=disp_layers[:, [0, 2]],
        mask_layers=mask_layers[:, [0, 2]],
        camtoworld=jnp.stack(
            [batch["src_camtoworld"][:, 0], batch["bg_src_camtoworld"]], axis=1
        ),
        intrinsic=batch["intrinsic"],
    )

    # -------------------------------------------------------------------------
    ft_ldi = data_types.LDI(
        rgb_layers=rgb_layers[:, 3],
        fg_layers=fg_layers[:, 3],
        fg_alphas=fg_alphas[:, 3],
        disp_layers=disp_layers[:, 3],
        mask_layers=mask_layers[:, 3],
        camtoworld=batch["bg_src_tgt_camtoworld"],
        intrinsic=batch["intrinsic"],
    )

    return src_ldi, bg_ldi, ft_ldi

  @nn.compact
  def __call__(self, batch):
    """LDI predictor model.

    Args:
      batch: data_typs.batch, contatining batch elements.

    Returns:
      out: data_types.RenderResult.
    """
    src_ldi, bg_ldi, ft_ldi = self.get_layer_pred(batch)

    # src composed image.
    src_composed_rgb = src_ldi.compose_rgb(self.ldi_params.white_bkgd)

    # ---------------------------------------------------------------------
    # RGB  projections.
    # ---------------------------------------------------------------------
    rgb_tgt_composed, alpha_tgt_composed = src_ldi.project_rgb(
        batch["camtoworld"],
        batch["intrinsic"],
        self.ldi_params.white_bkgd,
        merge=True,
        use_rts=self.ldi_params.use_rts,
    )

    # ---------------------------------------------------------------------
    # Project to a far away time step.
    # ---------------------------------------------------------------------
    rgb_proj_ft, alpha_proj_ft = bg_ldi.project_rgb(
        batch["bg_src_tgt_camtoworld"],
        batch["intrinsic"],
        self.ldi_params.white_bkgd,
        merge=False,
        use_rts=self.ldi_params.use_rts,
    )  # (b n h w c)

    out = data_types.RenderResult(
        ldi=src_ldi,
        bg_ldi=bg_ldi,
        ft_ldi=ft_ldi,
        # RGB prediction for source.
        pred_src_rgb=src_composed_rgb,
        # RGB Projection to time T.
        pred_tgt_rgb=rgb_tgt_composed,
        pred_tgt_alpha=(alpha_tgt_composed > 0.98) * 1.0,
        # RGB projection to T'
        pred_rgb_proj_ft=rgb_proj_ft,
        pred_alpha_proj_ft=(alpha_proj_ft > 0.98) * 1.0,
        # Coord projection at T'
        crop_projection=self.crop_projection,
    )
    return out


def create_model(key, example_batch, config):
  """Function to create and init a unet model."""
  unet_config = unet.get_unet_config(config)
  ldi_config = get_ldi_params(config)

  rgb_unet_params = unet.UnetConfig(
      out_features=3,
      num_res_blocks=config.model.unet_num_res_blocks,
      feat_scales=config.model.unet_feat_scales,
  )

  model = LDI(
      ldi_params=ldi_config,
      unet_params=unet_config,
      rgb_unet_params=rgb_unet_params,
      num_objects=config.dataset.num_objects,
      crop_projection=config.train.crop_projection,
  )
  init_variables = model.init(key, example_batch)

  return model, init_variables, train_utils.TrainMetrics
