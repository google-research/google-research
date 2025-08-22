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

"""Decoder module library."""
import functools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from flax import linen as nn

import jax.numpy as jnp

from invariant_slot_attention.lib import utils
from invariant_slot_attention.modules import misc

Shape = Tuple[int]

DType = Any
Array = Any  # jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class SpatialBroadcastDecoder(nn.Module):
  """Spatial broadcast decoder for a set of slots (per frame)."""

  resolution: Sequence[int]
  backbone: Callable[[], nn.Module]
  pos_emb: Callable[[], nn.Module]
  early_fusion: bool = False  # Fuse slot features before constructing targets.
  target_readout: Optional[Callable[[], nn.Module]] = None

  # Vmapped application of module, consumes time axis (axis=1).
  @functools.partial(utils.time_distributed, in_axes=(1, None))
  @nn.compact
  def __call__(self, slots, train = False):

    batch_size, n_slots, n_features = slots.shape

    # Fold slot dim into batch dim.
    x = jnp.reshape(slots, (batch_size * n_slots, n_features))

    # Spatial broadcast with position embedding.
    x = utils.spatial_broadcast(x, self.resolution)
    x = self.pos_emb()(x)

    # bb_features.shape = (batch_size * n_slots, h, w, c)
    bb_features = self.backbone()(x, train=train)
    spatial_dims = bb_features.shape[-3:-1]

    alpha_logits = nn.Dense(
        features=1, use_bias=True, name="alpha_logits")(bb_features)
    alpha_logits = jnp.reshape(
        alpha_logits, (batch_size, n_slots) + spatial_dims + (-1,))

    alphas = nn.softmax(alpha_logits, axis=1)
    if not train:
      # Define intermediates for logging / visualization.
      self.sow("intermediates", "alphas", alphas)

    if self.early_fusion:
      # To save memory, fuse the slot features before predicting targets.
      # The final target output should be equivalent to the late fusion when
      # using linear prediction.
      bb_features = jnp.reshape(
          bb_features, (batch_size, n_slots) + spatial_dims + (-1,))
      # Combine backbone features by alpha masks.
      bb_features = jnp.sum(bb_features * alphas, axis=1)

    targets_dict = self.target_readout()(bb_features, train)  # pylint: disable=not-callable

    preds_dict = dict()
    for target_key, channels in targets_dict.items():
      if self.early_fusion:
        # decoded_target.shape = (batch_size, h, w, c) after next line.
        decoded_target = channels
      else:
        # channels.shape = (batch_size, n_slots, h, w, c)
        channels = jnp.reshape(
            channels, (batch_size, n_slots) + (spatial_dims) + (-1,))

        # masked_channels.shape = (batch_size, n_slots, h, w, c)
        masked_channels = channels * alphas

        # decoded_target.shape = (batch_size, h, w, c)
        decoded_target = jnp.sum(masked_channels, axis=1)  # Combine target.
      preds_dict[target_key] = decoded_target

      if not train:
      # Define intermediates for logging / visualization.
        self.sow("intermediates", f"{target_key}_slots", channels)
        if not self.early_fusion:
          self.sow("intermediates", f"{target_key}_masked", masked_channels)
        self.sow("intermediates", f"{target_key}_combined", decoded_target)

    preds_dict["segmentations"] = jnp.argmax(alpha_logits, axis=1)

    return preds_dict


class SiameseSpatialBroadcastDecoder(nn.Module):
  """Siamese spatial broadcast decoder for a set of slots (per frame).

  Similar to the decoders used in IODINE: https://arxiv.org/abs/1903.00450
  and in Slot Attention: https://arxiv.org/abs/2006.15055.
  """

  resolution: Sequence[int]
  backbone: Callable[[], nn.Module]
  pos_emb: Callable[[], nn.Module]
  pass_intermediates: bool = False
  alpha_only: bool = False  # Predict only alpha masks.
  concat_attn: bool = False
  # Readout after backbone.
  target_readout_from_slots: bool = False
  target_readout: Optional[Callable[[], nn.Module]] = None
  early_fusion: bool = False  # Fuse slot features before constructing targets.
  # Readout on slots.
  attribute_readout: Optional[Callable[[], nn.Module]] = None
  remove_background_attribute: bool = False
  attn_key: Optional[str] = None
  attn_width: Optional[int] = None
  # If True, expects slot embeddings to contain slot positions.
  relative_positions: bool = False
  # Slot positions and scales.
  relative_positions_and_scales: bool = False
  relative_positions_rotations_and_scales: bool = False

  # Vmapped application of module, consumes time axis (axis=1).
  @functools.partial(utils.time_distributed, in_axes=(1, None))
  @nn.compact
  def __call__(self,
               slots,
               train = False):

    if self.remove_background_attribute and self.attribute_readout is None:
      raise NotImplementedError(
          "Background removal is only supported for attribute readout.")

    if self.relative_positions:
      # Assume slot positions were concatenated to slot embeddings.
      # E.g. an output of SlotAttentionTranslEquiv.
      slots, positions = slots[Ellipsis, :-2], slots[Ellipsis, -2:]
      # Reshape positions to [B * num_slots, 2]
      positions = positions.reshape(
          (positions.shape[0] * positions.shape[1], positions.shape[2]))
    elif self.relative_positions_and_scales:
      # Assume slot positions and scales were concatenated to slot embeddings.
      # E.g. an output of SlotAttentionTranslScaleEquiv.
      slots, positions, scales = (slots[Ellipsis, :-4],
                                  slots[Ellipsis, -4: -2],
                                  slots[Ellipsis, -2:])
      positions = positions.reshape(
          (positions.shape[0] * positions.shape[1], positions.shape[2]))
      scales = scales.reshape(
          (scales.shape[0] * scales.shape[1], scales.shape[2]))
    elif self.relative_positions_rotations_and_scales:
      slots, positions, scales, rotm = (slots[Ellipsis, :-8],
                                        slots[Ellipsis, -8: -6],
                                        slots[Ellipsis, -6: -4],
                                        slots[Ellipsis, -4:])
      positions = positions.reshape(
          (positions.shape[0] * positions.shape[1], positions.shape[2]))
      scales = scales.reshape(
          (scales.shape[0] * scales.shape[1], scales.shape[2]))
      rotm = rotm.reshape(
          rotm.shape[0] * rotm.shape[1], 2, 2)

    batch_size, n_slots, n_features = slots.shape

    preds_dict = {}
    # Fold slot dim into batch dim.
    x = jnp.reshape(slots, (batch_size * n_slots, n_features))

    # Attribute readout.
    if self.attribute_readout is not None:
      if self.remove_background_attribute:
        slots = slots[:, 1:]
      attributes_dict = self.attribute_readout()(slots, train)  # pylint: disable=not-callable
      preds_dict.update(attributes_dict)

    # Spatial broadcast with position embedding.
    # See https://arxiv.org/abs/1901.07017.
    x = utils.spatial_broadcast(x, self.resolution)

    if self.relative_positions:
      x = self.pos_emb()(inputs=x, slot_positions=positions)
    elif self.relative_positions_and_scales:
      x = self.pos_emb()(inputs=x, slot_positions=positions, slot_scales=scales)
    elif self.relative_positions_rotations_and_scales:
      x = self.pos_emb()(
          inputs=x, slot_positions=positions, slot_scales=scales,
          slot_rotm=rotm)
    else:
      x = self.pos_emb()(x)

    # bb_features.shape = (batch_size*n_slots, h, w, c)
    bb_features = self.backbone()(x, train=train)
    spatial_dims = bb_features.shape[-3:-1]
    alphas = nn.Dense(features=1, use_bias=True, name="alphas")(bb_features)
    alphas = jnp.reshape(
        alphas, (batch_size, n_slots) + spatial_dims + (-1,))
    alphas_softmaxed = nn.softmax(alphas, axis=1)
    preds_dict["segmentation_logits"] = alphas
    preds_dict["segmentations"] = jnp.argmax(alphas, axis=1)
    # Define intermediates for logging.
    _ = misc.Identity(name="alphas_softmaxed")(alphas_softmaxed)
    if self.alpha_only or self.target_readout is None:
      assert alphas.shape[-1] == 1, "Alpha masks need to be one-dimensional."
      return preds_dict, {"segmentation_logits": alphas}

    if self.early_fusion:
      # To save memory, fuse the slot features before predicting targets.
      # The final target output should be equivalent to the late fusion when
      # using linear prediction.
      bb_features = jnp.reshape(
          bb_features, (batch_size, n_slots) + spatial_dims + (-1,))
      # Combine backbone features by alpha masks.
      bb_features = jnp.sum(bb_features * alphas_softmaxed, axis=1)

    if self.target_readout_from_slots:
      targets_dict = self.target_readout()(slots, train)  # pylint: disable=not-callable
    else:
      targets_dict = self.target_readout()(bb_features, train)  # pylint: disable=not-callable

    targets_dict_new = dict()
    targets_dict_new["targets_masks"] = alphas_softmaxed
    targets_dict_new["targets_logits_masks"] = alphas

    for target_key, channels in targets_dict.items():
      if self.early_fusion:
        # decoded_target.shape = (batch_size, h, w, c) after next line.
        decoded_target = channels
      else:
        # channels.shape = (batch_size, n_slots, h, w, c) after next line.
        channels = jnp.reshape(
            channels, (batch_size, n_slots) +
            (spatial_dims if not self.target_readout_from_slots else
             (1, 1)) + (-1,))
        # masked_channels.shape = (batch_size, n_slots, h, w, c) at next line.
        masked_channels = channels * alphas_softmaxed
        # decoded_target.shape = (batch_size, h, w, c) after next line.
        decoded_target = jnp.sum(masked_channels, axis=1)  # Combine target.
        targets_dict_new[target_key + "_channels"] = channels
        # Define intermediates for logging.
        _ = misc.Identity(name=f"{target_key}_channels")(channels)
        _ = misc.Identity(name=f"{target_key}_masked_channels")(masked_channels)

      targets_dict_new[target_key] = decoded_target
      # Define intermediates for logging.
      _ = misc.Identity(name=f"decoded_{target_key}")(decoded_target)

    preds_dict.update(targets_dict_new)
    return preds_dict
