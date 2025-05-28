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

"""Helper multimodal layers."""

import collections
import functools
from typing import Sequence

from absl import logging
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp

from imp.max.core import constants
from imp.max.core import utils
from imp.max.modeling import embeddings as embeds
from imp.max.modeling import heads
from imp.max.modeling import linear
from imp.max.modeling import normalization
from imp.max.modeling import stochastic
from imp.max.utils import sharding
from imp.max.utils import typing

Modality = constants.Modality
_default_special_token_init = jax.nn.initializers.glorot_normal()
_default_mask_token_init = jax.nn.initializers.normal()


def extract_volume_patches(inputs,
                           patch_sizes,
                           output_shardings = (),
                           flatten = True):
  """Transforms raw inputs into volume patches.

  Args:
    inputs: Array of shape `[batch, instance, time, height, width, channels]`
    patch_sizes: Tuple of `[time_patch, height_patch, width_patch]`
    output_shardings: Sharding annotations to shard the patched array.
    flatten: Flattens the spatio-temporal tokens into a single dimension.

  Returns:
    Array of shape `[batch, instance, time_patches, height_patches,
        width_patches, time_patch, height_patch, width_patch, channels]`
  """
  inputs = jnp.reshape(
      inputs,
      (inputs.shape[0], inputs.shape[1], inputs.shape[2] // patch_sizes[0],
       patch_sizes[0], inputs.shape[3] // patch_sizes[1], patch_sizes[1],
       inputs.shape[4] // patch_sizes[2], patch_sizes[2], inputs.shape[5]))

  inputs = jnp.transpose(inputs, (0, 1, 2, 4, 6, 3, 5, 7, 8))
  inputs = jnp.reshape(inputs,
                       (inputs.shape[0], inputs.shape[1], inputs.shape[2],
                        inputs.shape[3], inputs.shape[4], -1))

  if flatten:
    inputs = jnp.reshape(
        inputs, (inputs.shape[0], inputs.shape[1], -1, inputs.shape[-1]))

  inputs = sharding.shard_array(inputs, output_shardings)
  return inputs


class PerModalityDense(nn.Module):
  """Per-Modality Dense Projections."""

  features: int
  dtype: jax.typing.DTypeLike = jnp.float32
  kernel_shardings: typing.ShardingAxes = ()
  dot_general: typing.DotGeneral = lax.dot_general
  lora_rank: int = 4
  lora_scale: float = 0.

  def setup(self):
    self.projections = {
        Modality.VISION: linear.Dense(
            features=self.features,
            use_bias=True,
            dtype=self.dtype,
            kernel_shardings=self.kernel_shardings,
            dot_general=self.dot_general,
            lora_rank=self.lora_rank,
            lora_scale=self.lora_scale,
            name='vision_dense'
        ),
        Modality.WAVEFORM: linear.Dense(
            features=self.features,
            use_bias=True,
            dtype=self.dtype,
            kernel_shardings=self.kernel_shardings,
            dot_general=self.dot_general,
            lora_rank=self.lora_rank,
            lora_scale=self.lora_scale,
            name='waveform_dense'
        ),
        Modality.SPECTROGRAM: linear.Dense(
            features=self.features,
            use_bias=True,
            dtype=self.dtype,
            kernel_shardings=self.kernel_shardings,
            dot_general=self.dot_general,
            lora_rank=self.lora_rank,
            lora_scale=self.lora_scale,
            name='spectrogram_dense'
        ),
        Modality.TEXT: linear.Dense(
            features=self.features,
            use_bias=True,
            dtype=self.dtype,
            kernel_shardings=self.kernel_shardings,
            dot_general=self.dot_general,
            lora_rank=self.lora_rank,
            lora_scale=self.lora_scale,
            name='text_dense'
        ),
    }

  def __call__(self, inputs, modality):
    return self.projections[modality](inputs)


class PerModalityCLS(nn.Module):
  """Per-Modality Classification Heads.

  This module can be configured as below:
  ```
  cls_model = PerModalityCLS(
      vision_classes=(('k400', 400), ('jft', 10)),
      spectrogram_classes=100,
      text_classes=(('c4', 43),)
  )
  ```

  and the user can expect the following when calling it wrt each modality:
  ```
  Parameters:
  vision_cls_k400/kernel -> shape: (16, 400)
  vision_cls_k400/bias -> shape: (400,)
  vision_cls_jft/kernel -> shape: (16, 10)
  vision_cls_jft/bias -> shape: (10,)
  spectrogram_cls/kernel -> shape: (16, 100)
  spectrogram_cls/bias -> shape: (100,)
  text_cls_c4/kernel -> shape: (16, 43)
  text_cls_c4/bias -> shape: (43,)

  Inputs:
  vision -> shape: (1, 3, 16)
  spectrogram -> shape: (1, 3, 16)
  text -> shape: (1, 3, 16)

  Outputs:
  vision/logits_k400 (1, 3, 400)
  vision/logits_jft (1, 3, 10)
  spectrogram/logits (1, 3, 100)
  text/logits_c4 (1, 3, 43)
  ```

  Attributes:
    vision_classes: An integer or a sequence of (str, int) pairs.
    waveform_classes: An integer or a sequence of (str, int) pairs.
    spectrogram_classes: An integer or a sequence of (str, int) pairs.
    text_classes: An integer or a sequence of (str, int) pairs.
    predictions_key: The predictions (logits) prefix key. If integer is used
      to configure the classes, the `predictions_key` will be solely used to
      refer to the logits. Otherwise, it will be used as a prefix followed by
      the name of the classes. For example, if classes=('dataset_1', 23), then
      all logits for this very classification will have the name
      f'{predictions_key}_dataset_1'.
    kernel_shardings: Sharding annotations for the dense kernels.
    dtype: The activation dtype of the entire module (across all heads).
    dense_dot_general: The function that performs dot product between the
      weights and their corresponding inputs.
  """

  vision_classes: typing.ClassificationHead = None
  waveform_classes: typing.ClassificationHead = None
  spectrogram_classes: typing.ClassificationHead = None
  text_classes: typing.ClassificationHead = None
  predictions_key: str = constants.DataFeatureName.LOGITS
  kernel_shardings: typing.ShardingAxes = ()
  dtype: jax.typing.DTypeLike = jnp.float32
  dense_dot_general: typing.DotGeneral = lax.dot_general

  def _fetch_classification_config(
      self,
      classes
      ):
    """Verifies and returns the cls config as a tuple of (str, int)."""
    if isinstance(classes, tuple):
      for classes_name, num_classes in classes:
        if (not classes_name
            or not isinstance(num_classes, int)
            or not utils.is_sub_np_dtype(num_classes, int)):
          raise ValueError(
              'The `classes` configuration should contain valid (str, int) '
              f'pairs. Instead, received ({classes_name}, {num_classes}).')
      return classes

    elif isinstance(classes, int) or utils.is_sub_np_dtype(classes, int):
      return (('', int(classes)),)

    else:
      raise ValueError(
          'The `classes` config should be either an integer or a tuple of '
          f'(str, int) pairs. Instead, received {classes=}')

  def _setup_modality_cls(
      self,
      modality,
      cls_cfg,
  ):
    """Constructs the modality-specific multi-head classification.

    Args:
      modality: The modality for which we want to construct the classification
        heads.
      cls_cfg: A sequence of (str, int) pairs that define the classification
        heads for the modality-of-interest.

    Returns:
      A nested dictionary that contains per-modality projections heads. This
      dictionary has the following structure:
          projections: {modality: {name_1: flax_module_1,
                                   name_2: flax_module_2,
                                   ...
                                   name_n: flax_module_n}}
    """
    projections = collections.defaultdict(dict)
    dense_class = functools.partial(
        linear.Dense,
        use_bias=True,
        dtype=self.dtype,
        kernel_shardings=self.kernel_shardings,
        dot_general=self.dense_dot_general)
    for classes_name, num_classes in cls_cfg:
      layer_name = f'{modality}_cls'
      if classes_name:
        layer_name += f'_{classes_name}'
      projections[modality][classes_name] = dense_class(
          features=num_classes, name=layer_name)

    return projections

  def setup(self):
    projections = {}
    if self.vision_classes is not None:
      vision_cls_cfg = self._fetch_classification_config(self.vision_classes)
      projections.update(
          self._setup_modality_cls(
              modality=Modality.VISION, cls_cfg=vision_cls_cfg))

    if self.waveform_classes is not None:
      waveform_cls_cfg = self._fetch_classification_config(
          self.waveform_classes)
      projections.update(
          self._setup_modality_cls(
              modality=Modality.WAVEFORM, cls_cfg=waveform_cls_cfg))

    if self.spectrogram_classes is not None:
      spectrogram_cls_cfg = self._fetch_classification_config(
          self.spectrogram_classes)
      projections.update(
          self._setup_modality_cls(
              modality=Modality.SPECTROGRAM, cls_cfg=spectrogram_cls_cfg))

    if self.text_classes is not None:
      text_cls_cfg = self._fetch_classification_config(self.text_classes)
      projections.update(
          self._setup_modality_cls(
              modality=Modality.TEXT, cls_cfg=text_cls_cfg))
    self.projections = projections

  def __call__(self, inputs, modality):
    outputs = {}
    cls_layers = self.projections.get(modality, None)
    if cls_layers is not None:
      for cls_name, cls_layer in cls_layers.items():
        output_name = self.predictions_key
        if cls_name:
          output_name += f'_{cls_name}'
        outputs[output_name] = cls_layer(inputs)
    return outputs


class FineAndCoarseCommonSpace(nn.Module):
  """Bridge audio, text and vision with a FAC style.

  This common-space projection method was originally proposed in
  https://arxiv.org/abs/2006.16228
  """

  d_va: int
  d_vat: int
  dtype: jax.typing.DTypeLike = jnp.float32
  kernel_shardings: typing.ShardingAxes = ()
  dense_dot_general: typing.DotGeneral = lax.dot_general
  lora_rank: int = 4
  lora_scale: float = 0.

  def setup(self):
    # abbreviations:
    #   va: vision-audio common space
    #   vat: vision-audio-text common space

    # vis-to-va is Dense + LN + gelu + Dense + LN
    vis_to_va = [
        linear.Dense(features=self.d_va,
                     dtype=self.dtype,
                     kernel_shardings=self.kernel_shardings,
                     dot_general=self.dense_dot_general,
                     lora_rank=self.lora_rank,
                     lora_scale=self.lora_scale,
                     name='vis_to_va_dense1'),
        normalization.LayerNorm(dtype=self.dtype,
                                name='vis_to_va_ln1'),
        nn.gelu,
        linear.Dense(features=self.d_va,
                     dtype=self.dtype,
                     dot_general=self.dense_dot_general,
                     lora_rank=self.lora_rank,
                     lora_scale=self.lora_scale,
                     name='vis_to_va_dense2'),
        normalization.LayerNorm(dtype=self.dtype,
                                name='vis_to_va_ln2'),
    ]

    # aud-to-va is Dense
    aud_to_va = [
        linear.Dense(features=self.d_va,
                     dtype=self.dtype,
                     kernel_shardings=self.kernel_shardings,
                     dot_general=self.dense_dot_general,
                     lora_rank=self.lora_rank,
                     lora_scale=self.lora_scale,
                     name='aud_to_va_dense1'),
    ]
    # va-to-vat is gelu + Dense + LN
    va_to_vat = [
        nn.gelu,
        linear.Dense(features=self.d_vat,
                     dtype=self.dtype,
                     dot_general=self.dense_dot_general,
                     lora_rank=self.lora_rank,
                     lora_scale=self.lora_scale,
                     name='va_to_vat_dense1'),
        normalization.LayerNorm(dtype=self.dtype,
                                name='va_to_vat_ln1'),
    ]
    # txt-to-vt is Dense
    txt_to_vat = [
        linear.Dense(features=self.d_vat,
                     dtype=self.dtype,
                     kernel_shardings=self.kernel_shardings,
                     dot_general=self.dense_dot_general,
                     lora_rank=self.lora_rank,
                     lora_scale=self.lora_scale,
                     name='txt_to_vat_dense1'),
    ]

    # define the cross-modal projections
    self.projections = {
        Modality.VISION: {
            Modality.AUDIO: vis_to_va,
            Modality.TEXT: vis_to_va + va_to_vat,
        },
        Modality.WAVEFORM: {
            Modality.VISION: aud_to_va,
            Modality.TEXT: aud_to_va + va_to_vat,
        },
        Modality.SPECTROGRAM: {
            Modality.VISION: aud_to_va,
            Modality.TEXT: aud_to_va + va_to_vat,
        },
        Modality.TEXT: {
            Modality.AUDIO: txt_to_vat,
            Modality.VISION: txt_to_vat,
        },
    }

  def __call__(self,
               inputs,
               modality,
               target_modalities,
               deterministic = None):
    outputs = {}
    for target_modality in target_modalities:
      layers = self.projections[modality][target_modality]
      layer_inputs = inputs
      for layer in layers:
        layer_inputs = layer(layer_inputs)
      outputs[target_modality] = layer_inputs

    return outputs


# TODO(hassanak): Remove all per-modality or per-modality-per-feature layers
# and directly use utils.construct_...
class PerModalityJointCommonSpace(nn.Module):
  """Joint common space mapping."""

  d_common: int
  d_hidden: int | None = None
  dtype: jax.typing.DTypeLike = jnp.float32
  dense_dot_general: typing.DotGeneral = lax.dot_general
  inner_kernel_shardings: typing.ShardingAxes = ()
  outer_kernel_shardings: typing.ShardingAxes = ()
  layernorm_shardings: typing.ShardingAxes = ()
  intermediate_shardings: typing.ShardingAxes = ()

  def setup(self):
    self.projections = {
        Modality.VISION: heads.MLP(
            d_hidden=self.d_hidden,
            d_common=self.d_common,
            dtype=self.dtype,
            dot_general=self.dense_dot_general,
            inner_kernel_shardings=self.inner_kernel_shardings,
            outer_kernel_shardings=self.outer_kernel_shardings,
            layernorm_shardings=self.layernorm_shardings,
            intermediate_shardings=self.intermediate_shardings,
            name='vision_to_common',
        ),
        Modality.WAVEFORM: heads.MLP(
            d_hidden=self.d_hidden,
            d_common=self.d_common,
            dtype=self.dtype,
            dot_general=self.dense_dot_general,
            inner_kernel_shardings=self.inner_kernel_shardings,
            outer_kernel_shardings=self.outer_kernel_shardings,
            layernorm_shardings=self.layernorm_shardings,
            intermediate_shardings=self.intermediate_shardings,
            name='waveform_to_common',
        ),
        Modality.SPECTROGRAM: heads.MLP(
            d_hidden=self.d_hidden,
            d_common=self.d_common,
            dtype=self.dtype,
            dot_general=self.dense_dot_general,
            inner_kernel_shardings=self.inner_kernel_shardings,
            outer_kernel_shardings=self.outer_kernel_shardings,
            layernorm_shardings=self.layernorm_shardings,
            intermediate_shardings=self.intermediate_shardings,
            name='spectrogram_to_common',
        ),
        Modality.TEXT: heads.MLP(
            d_hidden=self.d_hidden,
            d_common=self.d_common,
            dtype=self.dtype,
            dot_general=self.dense_dot_general,
            inner_kernel_shardings=self.inner_kernel_shardings,
            outer_kernel_shardings=self.outer_kernel_shardings,
            layernorm_shardings=self.layernorm_shardings,
            intermediate_shardings=self.intermediate_shardings,
            name='text_to_common',
        ),
    }

  def __call__(self,
               inputs,
               modality,
               target_modalities,
               deterministic = None):

    common_outputs = self.projections[modality](inputs=inputs)
    outputs = {}
    for target_modality in target_modalities:
      outputs[target_modality] = common_outputs

    return outputs


class PerModalityDisjointCommonSpace(nn.Module):
  """Pair-wise cross-modal common space mapping."""

  d_common: int
  d_hidden: int
  dtype: jax.typing.DTypeLike = jnp.float32
  dense_dot_general: typing.DotGeneral = lax.dot_general
  inner_kernel_shardings: typing.ShardingAxes = ()
  outer_kernel_shardings: typing.ShardingAxes = ()
  layernorm_shardings: typing.ShardingAxes = ()
  intermediate_shardings: typing.ShardingAxes = ()

  def setup(self):
    self.projections = {
        Modality.VISION: {
            Modality.WAVEFORM: heads.MLP(
                d_hidden=self.d_hidden,
                d_common=self.d_common,
                dtype=self.dtype,
                dot_general=self.dense_dot_general,
                inner_kernel_shardings=self.inner_kernel_shardings,
                outer_kernel_shardings=self.outer_kernel_shardings,
                layernorm_shardings=self.layernorm_shardings,
                intermediate_shardings=self.intermediate_shardings,
                name='vision_to_waveform',
            ),
            Modality.SPECTROGRAM: heads.MLP(
                d_hidden=self.d_hidden,
                d_common=self.d_common,
                dtype=self.dtype,
                dot_general=self.dense_dot_general,
                inner_kernel_shardings=self.inner_kernel_shardings,
                outer_kernel_shardings=self.outer_kernel_shardings,
                layernorm_shardings=self.layernorm_shardings,
                intermediate_shardings=self.intermediate_shardings,
                name='vision_to_spectrogram',
            ),
            Modality.TEXT: heads.MLP(
                d_hidden=self.d_hidden,
                d_common=self.d_common,
                dtype=self.dtype,
                dot_general=self.dense_dot_general,
                inner_kernel_shardings=self.inner_kernel_shardings,
                outer_kernel_shardings=self.outer_kernel_shardings,
                layernorm_shardings=self.layernorm_shardings,
                intermediate_shardings=self.intermediate_shardings,
                name='vision_to_text',
            ),
        },
        Modality.WAVEFORM: {
            Modality.VISION: heads.MLP(
                d_hidden=self.d_hidden,
                d_common=self.d_common,
                dtype=self.dtype,
                dot_general=self.dense_dot_general,
                inner_kernel_shardings=self.inner_kernel_shardings,
                outer_kernel_shardings=self.outer_kernel_shardings,
                layernorm_shardings=self.layernorm_shardings,
                intermediate_shardings=self.intermediate_shardings,
                name='waveform_to_vision',
            ),
            Modality.SPECTROGRAM: heads.MLP(
                d_hidden=self.d_hidden,
                d_common=self.d_common,
                dtype=self.dtype,
                dot_general=self.dense_dot_general,
                inner_kernel_shardings=self.inner_kernel_shardings,
                outer_kernel_shardings=self.outer_kernel_shardings,
                layernorm_shardings=self.layernorm_shardings,
                intermediate_shardings=self.intermediate_shardings,
                name='waveform_to_spectrogram',
            ),
            Modality.TEXT: heads.MLP(
                d_hidden=self.d_hidden,
                d_common=self.d_common,
                dtype=self.dtype,
                dot_general=self.dense_dot_general,
                inner_kernel_shardings=self.inner_kernel_shardings,
                outer_kernel_shardings=self.outer_kernel_shardings,
                layernorm_shardings=self.layernorm_shardings,
                intermediate_shardings=self.intermediate_shardings,
                name='waveform_to_text',
            ),
        },
        Modality.SPECTROGRAM: {
            Modality.VISION: heads.MLP(
                d_hidden=self.d_hidden,
                d_common=self.d_common,
                dtype=self.dtype,
                dot_general=self.dense_dot_general,
                inner_kernel_shardings=self.inner_kernel_shardings,
                outer_kernel_shardings=self.outer_kernel_shardings,
                layernorm_shardings=self.layernorm_shardings,
                intermediate_shardings=self.intermediate_shardings,
                name='spectrogram_to_vision',
            ),
            Modality.WAVEFORM: heads.MLP(
                d_hidden=self.d_hidden,
                d_common=self.d_common,
                dtype=self.dtype,
                dot_general=self.dense_dot_general,
                inner_kernel_shardings=self.inner_kernel_shardings,
                outer_kernel_shardings=self.outer_kernel_shardings,
                layernorm_shardings=self.layernorm_shardings,
                intermediate_shardings=self.intermediate_shardings,
                name='spectrogram_to_waveform',
            ),
            Modality.TEXT: heads.MLP(
                d_hidden=self.d_hidden,
                d_common=self.d_common,
                dtype=self.dtype,
                dot_general=self.dense_dot_general,
                inner_kernel_shardings=self.inner_kernel_shardings,
                outer_kernel_shardings=self.outer_kernel_shardings,
                layernorm_shardings=self.layernorm_shardings,
                intermediate_shardings=self.intermediate_shardings,
                name='spectrogram_to_text',
            ),
        },
        Modality.TEXT: {
            Modality.VISION: heads.MLP(
                d_hidden=self.d_hidden,
                d_common=self.d_common,
                dtype=self.dtype,
                dot_general=self.dense_dot_general,
                inner_kernel_shardings=self.inner_kernel_shardings,
                outer_kernel_shardings=self.outer_kernel_shardings,
                layernorm_shardings=self.layernorm_shardings,
                intermediate_shardings=self.intermediate_shardings,
                name='text_to_vision',
            ),
            Modality.WAVEFORM: heads.MLP(
                d_hidden=self.d_hidden,
                d_common=self.d_common,
                dtype=self.dtype,
                dot_general=self.dense_dot_general,
                inner_kernel_shardings=self.inner_kernel_shardings,
                outer_kernel_shardings=self.outer_kernel_shardings,
                layernorm_shardings=self.layernorm_shardings,
                intermediate_shardings=self.intermediate_shardings,
                name='text_to_waveform',
            ),
            Modality.SPECTROGRAM: heads.MLP(
                d_hidden=self.d_hidden,
                d_common=self.d_common,
                dtype=self.dtype,
                dot_general=self.dense_dot_general,
                inner_kernel_shardings=self.inner_kernel_shardings,
                outer_kernel_shardings=self.outer_kernel_shardings,
                layernorm_shardings=self.layernorm_shardings,
                intermediate_shardings=self.intermediate_shardings,
                name='text_to_spectrogram',
            ),
        },
        }

  def __call__(self,
               inputs,
               modality,
               target_modalities,
               deterministic = None):
    outputs = {}
    for target_modality in target_modalities:
      outputs[target_modality] = (
          self.projections[modality][target_modality](inputs=inputs)
      )

    return outputs


class BaseEmbedProjection(nn.Module):
  """Base x-to-embedding projection layer."""

  d_model: int
  modality: str
  pos_buckets: int | tuple[int, Ellipsis] | None = None
  dropout_rate: float = 0.1
  freeze_embeddings: bool = False
  dtype: jax.typing.DTypeLike = jnp.float32
  precision: typing.Precision = None
  pos_encode_take_dot_general: typing.DotGeneral = lax.dot_general
  pos_encode_lookup_dot_general: typing.DotGeneral = lax.dot_general
  pos_encode_embed_shardings: typing.ShardingAxes = ()
  pos_encode_layernorm_shardings: typing.ShardingAxes = ()

  def setup(self):
    # check if modality is supported
    supported_modalities = {Modality.VISION,
                            Modality.WAVEFORM,
                            Modality.SPECTROGRAM,
                            Modality.TEXT}
    if self.modality not in supported_modalities:
      raise ValueError(
          f'Invalid modality {self.modality!r}! Please specify a valid '
          f'modality from {supported_modalities}.')

    # add stop-gradient if the embeddings need to be frozen
    if self.freeze_embeddings:
      self.pre_pos = jax.lax.stop_gradient
    else:
      self.pre_pos = lambda x: x

    if self.modality == Modality.VISION:
      self.setup_vision_embedding_projection()
      if self.pos_buckets is not None:
        self.setup_vision_pos_encoding()

    elif self.modality == Modality.WAVEFORM:
      self.setup_waveform_embedding_projection()
      if self.pos_buckets is not None:
        self.setup_waveform_pos_encoding()

    elif self.modality == Modality.SPECTROGRAM:
      self.setup_spectrogram_embedding_projection()
      if self.pos_buckets is not None:
        self.setup_spectrogram_pos_encoding()

    elif self.modality == Modality.TEXT:
      self.setup_text_embedding_projection()
      if self.pos_buckets is not None:
        self.setup_text_pos_encoding()

  def setup_vision_pos_encoding(self):
    if isinstance(self.pos_buckets, tuple):
      pos_encoding_module = embeds.SpatioTemporalPosEncode
    elif (isinstance(self.pos_buckets, int)
          or utils.is_sub_np_dtype(self.pos_buckets, int)):
      pos_encoding_module = functools.partial(
          embeds.TemporalPosEncode,
          embedding_name='flattened_position_embeddings',
      )
    else:
      raise ValueError(
          f'Please provide a valid `pos_buckets`. Received {self.pos_buckets}.')

    self.add_pos_embeddings = pos_encoding_module(
        hidden_size=self.d_model,
        pos_buckets=self.pos_buckets,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        embedding_shardings=self.pos_encode_embed_shardings,
        layernorm_shardings=self.pos_encode_layernorm_shardings,
        lookup_dot_general=self.pos_encode_lookup_dot_general,
        name='rgb_pos_encoding')

  def setup_waveform_pos_encoding(self):
    self.add_pos_embeddings = embeds.TemporalPosEncode(
        hidden_size=self.d_model,
        pos_buckets=self.pos_buckets,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        embedding_shardings=self.pos_encode_embed_shardings,
        layernorm_shardings=self.pos_encode_layernorm_shardings,
        lookup_dot_general=self.pos_encode_lookup_dot_general,
        name='wav_pos_encoding')

  def setup_spectrogram_pos_encoding(self):
    if isinstance(self.pos_buckets, tuple):
      pos_encoding_module = embeds.SpectroTemporalPosEncode
    elif (isinstance(self.pos_buckets, int)
          or utils.is_sub_np_dtype(self.pos_buckets, int)):
      pos_encoding_module = functools.partial(
          embeds.TemporalPosEncode,
          embedding_name='flattened_position_embeddings',
      )
    else:
      raise ValueError(
          f'Please provide a valid `pos_buckets`. Received {self.pos_buckets}.')

    self.add_pos_embeddings = pos_encoding_module(
        hidden_size=self.d_model,
        pos_buckets=self.pos_buckets,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        lookup_dot_general=self.pos_encode_lookup_dot_general,
        embedding_shardings=self.pos_encode_embed_shardings,
        layernorm_shardings=self.pos_encode_layernorm_shardings,
        name='spc_pos_encoding')

  def setup_text_pos_encoding(self):
    self.add_pos_embeddings = embeds.TemporalPosEncode(
        hidden_size=self.d_model,
        pos_buckets=self.pos_buckets,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        lookup_dot_general=self.pos_encode_lookup_dot_general,
        embedding_shardings=self.pos_encode_embed_shardings,
        layernorm_shardings=self.pos_encode_layernorm_shardings,
        name='txt_pos_encoding')

  def setup_vision_embedding_projection(self):
    raise NotImplementedError

  def setup_waveform_embedding_projection(self):
    raise NotImplementedError

  def setup_spectrogram_embedding_projection(self):
    raise NotImplementedError

  def setup_text_embedding_projection(self):
    raise NotImplementedError


class RawToEmbed(BaseEmbedProjection):
  """Modality-specific Raw-to-Embeddings module."""

  patch_size: int | tuple[int, Ellipsis] | None = None
  droptoken_rate: float = 0.
  raw_to_embed_kernel_shardings: typing.ShardingAxes = ()
  raw_to_embed_bias_shardings: typing.ShardingAxes = ()

  def setup_vision_embedding_projection(self):
    self.input_rank = constants.DataFeatureRank.Vision.RAW
    self.raw_to_embeddings = linear.Conv(
        features=self.d_model,
        kernel_size=self.patch_size,
        strides=self.patch_size,
        padding='SAME',
        dtype=self.dtype,
        precision=self.precision,
        kernel_shardings=self.raw_to_embed_kernel_shardings,
        bias_shardings=self.raw_to_embed_bias_shardings,
        name='rgb_to_embedding')

  def setup_waveform_embedding_projection(self):
    self.input_rank = constants.DataFeatureRank.Waveform.RAW
    self.raw_to_embeddings = linear.Conv(
        features=self.d_model,
        kernel_size=(self.patch_size,),
        strides=(self.patch_size,),
        padding='SAME',
        dtype=self.dtype,
        precision=self.precision,
        kernel_shardings=self.raw_to_embed_kernel_shardings,
        bias_shardings=self.raw_to_embed_bias_shardings,
        name='wav_to_embedding')

  def setup_spectrogram_embedding_projection(self):
    self.input_rank = constants.DataFeatureRank.Spectrogram.RAW
    self.raw_to_embeddings = linear.Conv(
        features=self.d_model,
        kernel_size=self.patch_size,
        strides=self.patch_size,
        padding='SAME',
        dtype=self.dtype,
        precision=self.precision,
        kernel_shardings=self.raw_to_embed_kernel_shardings,
        bias_shardings=self.raw_to_embed_bias_shardings,
        name='spc_to_embedding')

  def setup(self):
    super().setup()

    # construct DropToken module if a non-zero rate is specified
    if self.droptoken_rate > 0.:
      if self.modality == Modality.TEXT:
        raise ValueError('DropToken is not supported with Text modality.')
      self.droptoken = stochastic.DropToken(self.droptoken_rate)
    else:
      self.droptoken = lambda x, y: x

  def _flatten_embeddings(
      self, inputs):
    # the projected (patches/tokens) embeddings have a shape of (B, N, ..., D)
    # hence, we need to accumulate all patches/tokens along the token dimension
    input_shape = inputs.shape
    flattened_shape = [input_shape[0], input_shape[1], -1, input_shape[-1]]

    return jnp.reshape(inputs, flattened_shape), input_shape

  def __call__(
      self,
      inputs,
      deterministic = True
  ):

    # assert input shape is correct
    input_shape = inputs.shape
    if len(input_shape) != self.input_rank:
      raise ValueError(
          f'Expected input rank {self.input_rank} for the {self.modality} '
          f'modality but {len(input_shape)} was given.'
          )

    # project raw inputs to embeddings space
    # the projection is vectorized along the 'instance' axis
    embeddings = jax.vmap(self.raw_to_embeddings, in_axes=1, out_axes=1)(inputs)

    # accumulate all tokens to one axis
    embeddings, embeddings_shape = self._flatten_embeddings(embeddings)

    # if any operation is needed before adding positional embeddings
    embeddings = self.pre_pos(embeddings)

    # add positional embeddings
    if hasattr(self, 'add_pos_embeddings'):
      embeddings = self.add_pos_embeddings(embeddings, deterministic)
    else:
      logging.info('No positional encoding has been annotated in %s', self)

    # apply droptoken (if any)
    embeddings = self.droptoken(embeddings, deterministic)

    return embeddings, embeddings_shape


# TODO(b/309951446): Add proper attributes
class TokenRawToEmbed(BaseEmbedProjection):
  """Modality-specific TokenRaw-to-Embeddings module."""

  raw_to_embed_kernel_shardings: typing.ShardingAxes = ()
  raw_to_embed_dense_dot_general: typing.DotGeneral = lax.dot_general
  seg_buckets: int | None = None
  seg_encode_lookup_dot_general: typing.DotGeneral = lax.dot_general
  seg_encode_embed_shardings: typing.ShardingAxes = ()

  def setup(self):
    super().setup()
    if self.seg_buckets is not None and self.seg_buckets > 0:
      naming = {Modality.VISION: 'rgb_seg_encoding',
                Modality.TEXT: 'txt_seg_encoding',
                Modality.SPECTROGRAM: 'spc_seg_encoding',
                Modality.WAVEFORM: 'wav_seg_encoding'}
      self.segment_embeddings = embeds.Embed(
          num_embeddings=self.seg_buckets,
          features=self.d_model,
          dtype=self.dtype,
          shardings=self.seg_encode_embed_shardings,
          lookup_dot_general=self.seg_encode_lookup_dot_general,
          name=naming[self.modality],
      )

  def setup_vision_embedding_projection(self):
    self.input_rank = constants.DataFeatureRank.Vision.TOKEN_RAW
    self.raw_to_embeddings = linear.DenseGeneral(
        features=self.d_model,
        axis=-1,
        use_bias=True,
        dtype=self.dtype,
        precision=self.precision,
        dot_general=self.raw_to_embed_dense_dot_general,
        kernel_shardings=self.raw_to_embed_kernel_shardings,
        name='rgb_to_embedding')

  def setup_waveform_embedding_projection(self):
    self.input_rank = constants.DataFeatureRank.Waveform.TOKEN_RAW
    self.raw_to_embeddings = linear.DenseGeneral(
        features=self.d_model,
        axis=-1,
        use_bias=True,
        dtype=self.dtype,
        precision=self.precision,
        dot_general=self.raw_to_embed_dense_dot_general,
        kernel_shardings=self.raw_to_embed_kernel_shardings,
        name='wav_to_embedding')

  def setup_spectrogram_embedding_projection(self):
    self.input_rank = constants.DataFeatureRank.Spectrogram.TOKEN_RAW
    self.raw_to_embeddings = linear.DenseGeneral(
        features=self.d_model,
        axis=-1,
        use_bias=True,
        dtype=self.dtype,
        precision=self.precision,
        dot_general=self.raw_to_embed_dense_dot_general,
        kernel_shardings=self.raw_to_embed_kernel_shardings,
        name='spc_to_embedding')

  def setup_text_embedding_projection(self):
    self.input_rank = constants.DataFeatureRank.Text.TOKEN_RAW
    self.raw_to_embeddings = linear.DenseGeneral(
        features=self.d_model,
        axis=-1,
        use_bias=True,
        dtype=self.dtype,
        precision=self.precision,
        dot_general=self.raw_to_embed_dense_dot_general,
        kernel_shardings=self.raw_to_embed_kernel_shardings,
        name='txt_to_embedding')

  def __call__(
      self,
      inputs,
      deterministic = True,
      token_coordinate = None,
      coordinate_scale = None,
      token_segment_id = None,
      ):

    # assert input shape is correct
    input_shape = inputs.shape
    if len(input_shape) != self.input_rank:
      raise ValueError(
          f'Expected input rank {self.input_rank} for the {self.modality} '
          f'modality but {len(input_shape)} was given.'
          )

    # project tokens to embeddings space
    embeddings = self.raw_to_embeddings(inputs)

    # if any operation is needed before adding positional embeddings
    embeddings = self.pre_pos(embeddings)

    # add segment embeddings
    if hasattr(self, 'segment_embeddings'):
      if token_segment_id is None:
        raise ValueError(
            '`token_segment_id` is not provided while `seg_buckets` has been '
            f'configured. Please either instantiate {self} with '
            '`seg_buckets=None` or provide a valid `token_segment_id`.'
        )
      embeddings += self.segment_embeddings(token_segment_id)

    # add positional embeddings
    if hasattr(self, 'add_pos_embeddings'):
      embeddings = self.add_pos_embeddings(
          embeddings, deterministic, token_coordinate, coordinate_scale)
    else:
      logging.info('No positional encoding has been annotated in %s', self)

    return embeddings


class TokenIdToEmbed(TokenRawToEmbed):
  """Modality-specific TokenID-to-Embeddings module."""

  vocab_size: int | None = None
  id_to_embed_kernel_shardings: typing.ShardingAxes = ()
  id_to_embed_lookup_dot_general: typing.DotGeneral = lax.dot_general

  def setup_vision_embedding_projection(self):
    self.input_rank = constants.DataFeatureRank.Vision.TOKEN_ID
    self.raw_to_embeddings = embeds.Embed(
        num_embeddings=self.vocab_size,
        features=self.d_model,
        dtype=self.dtype,
        lookup_dot_general=self.id_to_embed_lookup_dot_general,
        shardings=self.id_to_embed_kernel_shardings,
        name='vis_to_embedding')

  def setup_waveform_embedding_projection(self):
    self.input_rank = constants.DataFeatureRank.Waveform.TOKEN_ID
    self.raw_to_embeddings = embeds.Embed(
        num_embeddings=self.vocab_size,
        features=self.d_model,
        dtype=self.dtype,
        lookup_dot_general=self.id_to_embed_lookup_dot_general,
        shardings=self.id_to_embed_kernel_shardings,
        name='wav_to_embedding')

  def setup_spectrogram_embedding_projection(self):
    self.input_rank = constants.DataFeatureRank.Spectrogram.TOKEN_ID
    self.raw_to_embeddings = embeds.Embed(
        num_embeddings=self.vocab_size,
        features=self.d_model,
        dtype=self.dtype,
        lookup_dot_general=self.id_to_embed_lookup_dot_general,
        shardings=self.id_to_embed_kernel_shardings,
        name='spc_to_embedding')

  def setup_text_embedding_projection(self):
    self.input_rank = constants.DataFeatureRank.Text.TOKEN_ID
    self.raw_to_embeddings = embeds.Embed(
        num_embeddings=self.vocab_size,
        features=self.d_model,
        dtype=self.dtype,
        lookup_dot_general=self.id_to_embed_lookup_dot_general,
        shardings=self.id_to_embed_kernel_shardings,
        name='txt_to_embedding')


class PerModalitySpecialToken(embeds.SpecialToken):
  """Appends special token to a multimodal input sequence."""

  def setup(self):
    if self.extension not in [constants.Extension.APPEND,
                              constants.Extension.PREPEND]:
      raise ValueError('Wrong extension position!')

    embedding_init = sharding.modulate_param_init(
        self.embedding_init, self.embedding_shardings
    )
    self.spc_token = {
        Modality.VISION: self.param(
            name='vision_embedding',
            init_fn=embedding_init,
            shape=(1, self.features),
            dtype=self.param_dtype,
            unbox=True,
        ),
        Modality.WAVEFORM: self.param(
            name='waveform_embedding',
            init_fn=embedding_init,
            shape=(1, self.features),
            dtype=self.param_dtype,
            unbox=True,
        ),
        Modality.SPECTROGRAM: self.param(
            name='spectrogram_embedding',
            init_fn=embedding_init,
            shape=(1, self.features),
            dtype=self.param_dtype,
            unbox=True,
        ),
        Modality.TEXT: self.param(
            name='text_embedding',
            init_fn=embedding_init,
            shape=(1, self.features),
            dtype=self.param_dtype,
            unbox=True,
        ),
    }

  def __call__(
      self,
      inputs,
      modality = None,
      token_mask = None,
      attention_bias = None
  ):
    """Appends special token to a multimodal input sequence.

    Args:
      inputs: `[batch, n_instance, q_length, num_heads * d_head]`.
      modality: string indicating the modality of inputs. Currently supported
        are `vision`, `waveform`, `spectrogram`, and `text`.
      token_mask: A 0/1 token mask with shape `[batch, n_instance, q_length]`.
      attention_bias: bias for the attention scores. `[num_heads, q_length,
        kv_length]`.

    Returns:
      Extended inputs, extended token_mask and extended attention_bias.

    Raises:
      ValueError: If token_mask or attention_bias shape don't meet
        requirements.
    """
    if modality is None:
      raise ValueError('Modality argument should be specified.')

    utils.verify_attention_shapes(attention_bias, token_mask, inputs.shape)

    # append modalities special tokens: [vis, wav, spc, txt]
    spc_token = jnp.asarray(self.spc_token[modality], dtype=self.dtype)
    special_embd = spc_token[jnp.newaxis, jnp.newaxis, :, :]
    inputs = self._append_special_token(inputs, special_embd)

    # extend token_mask and attention_bias accordingly
    if token_mask is not None:
      token_mask = utils.extend_token_mask(token_mask, self.extension)

    if attention_bias is not None:
      attention_bias = utils.extend_attention_bias(attention_bias,
                                                   self.extension)

    return inputs, token_mask, attention_bias


class PerModalityTemperature(nn.Module):
  """Learnable temperature parameters for contrastive estimation.

  Attributes:
    init_value: initial value for all temperature parameters.
    modalities: all modalities to be used. The module will set up
      temperature parameters for all possible pairs of modalities.
    dtype: the dtype to use for the temperature parameters.
  """

  init_value: float
  modalities: Sequence[str] = (Modality.VISION,
                               Modality.WAVEFORM,
                               Modality.SPECTROGRAM,
                               Modality.TEXT)
  dtype: jax.typing.DTypeLike = jnp.float32
  param_dtype: jax.typing.DTypeLike = jnp.float32

  def setup(self):
    """Sets up the learnable temperature parameters for each modality pair."""
    temperature = {}

    # Generate standardized names for each modality pair.
    # TODO(b/236159331): pass in a dict via a config for greater flexibility.
    pair_modalities = set()
    for modality in self.modalities:
      for target_modality in self.modalities:
        pair_name = '_'.join(sorted([modality, target_modality]))
        pair_modalities.add(pair_name)
    pair_modalities = sorted(pair_modalities)

    temperature_param_init = sharding.modulate_param_init(
        jax.nn.initializers.constant(self.init_value), ())
    for modality in pair_modalities:
      temperature[modality] = self.param(
          name=f'temperature_{modality}',
          init_fn=temperature_param_init,
          shape=(),
          dtype=self.param_dtype,
          unbox=True,
      )
    self.temperature = temperature

  def __call__(self):
    """Returns the temperature parameters."""
    # Ensure temperature is never negative.
    return {
        modality: jax.nn.relu(jnp.asarray(self.temperature[modality],
                                          dtype=self.dtype))
        for modality in self.temperature
    }


class PerModalityMaskFiller(nn.Module):
  """Learnable multimodal mask embeddings to fill in certain given positions.

  Attributes:
    dim: dimension of the embeddings.
    dtype: the dtype to use for the masked inputs.
  """

  dim: int
  embedding_init: nn.initializers.Initializer = _default_mask_token_init
  embedding_shardings: typing.ShardingAxes = ()
  dtype: jax.typing.DTypeLike = jnp.float32
  param_dtype: jax.typing.DTypeLike = jnp.float32
  precision: typing.Precision = None
  scatter_dot_general: typing.DotGeneral = lax.dot_general

  def setup(self):
    mask_embedding_init = sharding.modulate_param_init(
        self.embedding_init, self.embedding_shardings
    )
    self.mask_embedding = {
        Modality.VISION: self.param(
            name='vision_mask_embedding',
            init_fn=mask_embedding_init,
            shape=(self.dim,),
            dtype=self.param_dtype,
            unbox=True,
        ),
        Modality.WAVEFORM: self.param(
            name='waveform_mask_embedding',
            init_fn=mask_embedding_init,
            shape=(self.dim,),
            dtype=self.param_dtype,
            unbox=True,
        ),
        Modality.SPECTROGRAM: self.param(
            name='spectrogram_mask_embedding',
            init_fn=mask_embedding_init,
            shape=(self.dim,),
            dtype=self.param_dtype,
            unbox=True,
        ),
        Modality.TEXT: self.param(
            name='text_mask_embedding',
            init_fn=mask_embedding_init,
            shape=(self.dim,),
            dtype=self.param_dtype,
            unbox=True,
        ),
    }

  def __call__(self,
               inputs,
               mask_position_ids,
               keep_position_ids,
               modality,
               axis):
    rank = inputs.ndim
    if not 2 <= rank <= 4:
      raise ValueError(
          'Input must have 2 <= rank <= 4. Instead, received a tensor with '
          f'shape {inputs.shape}')
    inputs = inputs.astype(self.dtype)

    # Fetch the modality-specific embedding
    mask_embedding = self.mask_embedding[modality]
    mask_embedding = jnp.asarray(mask_embedding, dtype=self.dtype)

    # Tile mask embeddings to reflect the non-dim axes (e.g. batch/instance/pos)
    num_mask_positions = mask_position_ids.shape[-1]
    tile_dims = inputs.shape[:-2] + (num_mask_positions, 1)
    mask_embedding_updates = jnp.tile(mask_embedding, tile_dims)

    if keep_position_ids is None:
      inputs_masked = utils.scatter_along_axis(
          inputs=inputs,
          updates=mask_embedding_updates,
          indices=mask_position_ids,
          axis=-2,
          batch_dims=tuple(range(mask_position_ids.ndim - 1)),
          precision=self.precision,
          dot_general=self.scatter_dot_general)

    else:
      inputs_masked = utils.fill_by_scatter(
          inputs=inputs,
          updates=mask_embedding_updates,
          keep_indices=keep_position_ids,
          fill_indices=mask_position_ids,
          axis=-2,
          length=keep_position_ids.shape[-1] + mask_position_ids.shape[-1],
          keep_batch_dims=tuple(range(keep_position_ids.ndim - 1)),
          fill_batch_dims=tuple(range(mask_position_ids.ndim - 1)),
          precision=self.precision,
          dot_general=self.scatter_dot_general)

    return inputs_masked

