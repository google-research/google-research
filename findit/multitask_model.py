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

"""Multi-task model for FindIt.
"""


import functools

from flax import linen as nn
import jax.numpy as jnp

from findit import base
from findit import resnet
from findit.task_utils import RefExpTask


# Padding token ID.
PAD_ID = 0


def safe_divide(x,
                y,
                rtol = 1e-5,
                atol = 1e-8):
  """Computes a safe divide which returns 0 if the denominator is zero.

  Args:
    x: A float of numerator.
    y: A float of denominator.
    rtol: The relative tolerance parameter. See numpy.isclose for more info.
    atol: The absolute tolerance parameter. See numpy.isclose for more info.

  Returns:
    z: output x / y or 0.
  """
  is_zero = jnp.isclose(y, 0.0, rtol=rtol, atol=atol)
  safe_y = jnp.where(is_zero, jnp.ones_like(y), y)
  return jnp.where(is_zero, jnp.zeros_like(x), x / safe_y)


class BaseFusionModel(base.BaseModel):
  """Module interface of the fusion models."""

  def _check_text_shape(self, text_features, text_paddings):
    """Check the shapes of text features and text paddings.

    Args:
      text_features: A text feature array of shape: [batch, sentence_length,
        feat_dim].
      text_paddings: A text token array of shape [batch, sentence_length].
    """
    if len(text_features.shape) != 3:
      raise ValueError(
          f'Text feature dim must be 3, got {text_features.shape}!')
    if text_features.shape[:2] != text_paddings.shape:
      raise ValueError(
          'Text features and text paddings first two dimensions must'
          f' match: got {text_features.shape} and {text_paddings.shape}!')

  def _pool_text_features(self, text_features,
                          text_paddings):
    """Average the text features over only the valid tokens.

    Args:
      text_features: A text feature array of shape: [batch, sentence_length,
        feat_dim].
      text_paddings: A text token array of shape [batch, sentence_length].

    Returns:
      text_features: A pooled text features of shape [batch, feat_dim].
    """
    masked_text_features = text_features * text_paddings[Ellipsis, None]
    # Average the features over only the valid tokens.
    return safe_divide(
        jnp.sum(masked_text_features, axis=-2),
        jnp.sum(text_paddings, axis=-1, keepdims=True))

  def _normalize_features(self, features, axis = -1):
    """Normalize the features along a given axis.

    Args:
      features: An array of vision or text features to normalize.
      axis: Axis to normalize from. Default to the last dimension.

    Returns:
      Normalized features in the same shape as input features.
    """
    return safe_divide(features,
                       jnp.linalg.norm(features, axis=axis, keepdims=True))

  def _fuse_multilevel_features(
      self,
      vision_features,
      text_features,
      text_paddings,
      fuse_vis_text_features_fn,
  ):
    """Fuse multilevel features with text.

    Args:
      vision_features: A dictionary of features array or just an array.
      text_features: An array of text feature of shape [batch, dim] or [batch,
        seq_len, dim].
      text_paddings: An array of text feature padding of shape [batch, seq_len].
      fuse_vis_text_features_fn: A function to fuse a vision feature array with
        text feature array.

    Returns:
      fused_features_and_paddings: This could be the following shapes/types:
        1. An array of fused features
        2. A tuple of fused features array and paddings array.
        3. A dictionary of the above.
    """
    if isinstance(vision_features, dict):
      fused_features_and_paddings = {}
      for level in vision_features:
        fused_features_and_paddings[level] = fuse_vis_text_features_fn(
            vision_features[level], text_features, text_paddings, level)
    else:
      fused_features_and_paddings = fuse_vis_text_features_fn(
          vision_features, text_features, text_paddings, -1)

    return fused_features_and_paddings

  def _flatten_and_concat_features(self, vision_features,
                                   text_features,
                                   text_paddings):
    """Flatten and concatenate vision features with text features.

    Args:
      vision_features: Vision feature in shape [batch, height, width, dim].
      text_features: Text feature of shape [batch, seq_len, dim].
      text_paddings: Text feature padding of shape [batch, seq_len].

    Returns:
      fused_features: Fused feature in shape [batch, vision_length + seq_len,
        dim].
      paddings: Fused padding in shape [batch, vision_length + seq_len].
    """
    text_dim = text_features.shape[-1]
    batch, height, width, vis_dim = vision_features.shape
    if text_dim != vis_dim:
      raise ValueError('Text and vision feature must have equal dimension!')
    vision_length = height * width
    vision_features = vision_features.reshape([batch, vision_length, -1])
    fused_features = jnp.concatenate([vision_features, text_features], axis=1)
    # Concatenates 1s to the paddings to mark the vision feature as
    # non-paddings.
    paddings = jnp.concatenate([
        jnp.ones(
            (batch, vision_length), dtype=text_paddings.dtype), text_paddings
    ],
                               axis=1)
    return fused_features, paddings


class ProductFusionModel(BaseFusionModel):
  """Fuse the vision and language features with linear layer and product.

  Attributes:
    dtype: Type of data to use.
    num_fused_feature: Number of fused feature.
    num_vision_layers: Number of vision linear layers.
    num_language_layers: Number of language linear layers.
  """
  dtype: jnp.dtype = jnp.float32
  num_fused_feature: int = 32
  num_vision_layers: int = 1
  num_language_layers: int = 1

  @nn.compact
  def __call__(self, vision_features, text_features, text_paddings):
    """Call function.

    Args:
      vision_features: Vision features derived from an image encoder model.
        The features are in the shape of: [batch, height, width, feat_dim]. It
          can also be a dictionary of arrays. The fusion logic itself is not
          limited to 2D spatial features, but the limitation comes from the
          conv_fn which only operates on 2D array.
      text_features: Text features in shape [batch, seq_len, feature_dim],
        usually derived from a text encoder model.
      text_paddings:  The paddings in shape [batch, seq_len] and are are marked
        as VALID (1s) or INVALID (PAD_ID = 0s). It should have the same batch
        and seq_len dimension with the text_features.

    Returns:
      fused_features: Fused features with shape [batch, ..., num_fused_features]
        , where ... is the same as input vision features. And paddings for the
      fused features.
    """
    self._check_text_shape(text_features, text_paddings)
    norm_fn = functools.partial(
        nn.BatchNorm,
        use_running_average=self.mode != base.ExecutionMode.TRAIN,
        momentum=0.997,
        epsilon=1e-4,
        dtype=self.dtype)
    conv_fn = functools.partial(
        nn.Conv, self.num_fused_feature, kernel_size=(1, 1), dtype=self.dtype)

    def fuse_vis_text_feature_fn(vision_feature_array,
                                 text_features,
                                 text_paddings,
                                 level = -1):
      """Fuse vision and text features by elementwise product.

      Args:
        vision_feature_array: An array of vision features to fuse with text.
          This function applies self.num_vision_layers convolution 1x1 kernels
          on the features along with batch normalization and relu.
        text_features: An array of text features to fuse with vision features.
          The text features here are assumed to have gone through necessary
          transformation to match the feature dimension with vision features.
        text_paddings: An array of text paddings to use for fusion.
        level: An integer of vision feature level to name the conv layer.

      Returns:
        fused_features: An array of fused vision/text features of the same
          shape as the vision features except for the last dimension.
      """
      # Average the valid text features.
      text_features = self._pool_text_features(text_features, text_paddings)
      # Normalize the vision and text features.
      vision_feature_array = self._normalize_features(vision_feature_array)
      text_features = self._normalize_features(text_features)
      level_name = f'l{level}_' if level > -1 else ''
      for layer_idx in range(self.num_vision_layers):
        conv_layer_name = 'fuse_vision_' + level_name + f'conv_{layer_idx}'
        vision_feature_array = conv_fn(name=conv_layer_name)(
            vision_feature_array)
        vision_feature_array = norm_fn(name=conv_layer_name + '_bn')(
            vision_feature_array)
        vision_feature_array = nn.relu(vision_feature_array)
      num_extra_dims = len(vision_feature_array.shape) - 2
      text_shape = ([text_features.shape[0]] + num_extra_dims * [1] +
                    [text_features.shape[1]])
      return vision_feature_array * jnp.reshape(text_features, text_shape)

    # Transform language features to the fixed dimension of fused features.
    for language_layer_idx in range(self.num_language_layers):
      text_features = nn.Dense(
          self.num_fused_feature,
          dtype=self.dtype,
          name=f'fuse_language_dense_{language_layer_idx}')(
              text_features)
      text_features = norm_fn(
          name=f'fuse_language_dense_{language_layer_idx}_bn')(
              text_features)
      text_features = nn.relu(text_features)

    fused_features = self._fuse_multilevel_features(vision_features,
                                                    text_features,
                                                    text_paddings,
                                                    fuse_vis_text_feature_fn)
    return fused_features, None


class PaddingTextEncoder(base.BaseModel):
  """Dummy Text Encoder.

   Attributes:
      embed_dim: The output text feature embedding dimension.
  """
  embed_dim: int = 64

  def __call__(self, texts):
    """Encodes texts data to text features.

    Args:
      texts: ndarray of the tokenized texts data in shape [batch, seq_len].

    Returns:
      ndarray of text features after applying the T5 encoder model in shape
      [batch, seq_length, embed_dim].
    """
    return jnp.tile(texts[:, :, None], (1, 1, self.embed_dim))


class MultitaskModel(base.BaseModel):
  """Multi-task model function.

  Attributes:
    tasks: A sequence of tasks to instantiate. Each task contains its own head.
    vision_model_fn: A function returning a nn.Module specifying which vision
      encoder to use.
    language_model_fn: A function returning nn.Module specifying which language
      encoder to use.
    fusion_model_fn: A function returning nn.Module specifying which fusion
      model to use.
    dtype: A jax data type.
  """
  tasks = (RefExpTask(),)
  vision_model_fn = resnet.ResNet9
  language_model_fn = PaddingTextEncoder
  fusion_model_fn = ProductFusionModel
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    """Initializes a Module lazily (similar to a lazy ``__init__``).
    """
    if not self.tasks:
      raise ValueError('Tasks must not be empty!!')

    module_attrs = {
        'train': (self.mode == base.ExecutionMode.TRAIN),
        'mode': self.mode,
        'dtype': self.dtype,
    }

    self.vision_model = self.vision_model_fn(
        **base.filter_attrs(self.vision_model_fn, module_attrs))
    self.language_model = self.language_model_fn(
        **base.filter_attrs(self.language_model_fn, module_attrs))
    self.fusion_model = self.fusion_model_fn(
        **base.filter_attrs(self.fusion_model_fn, module_attrs))

    # Set up task heads.
    self.task_heads = [task.head(
        **base.filter_attrs(task.head, module_attrs))for task in self.tasks]

  @nn.compact
  def __call__(self,
               image,
               text,
               labels):
    """Call function for the multi-task model.

    Args:
      image: An array of shape [batch_size, height, width, channels].
      text: A numeric array of the input text with shape
        [batch_size, ..., seq_len].
      labels: A dictionary with task-specific labels.

    Returns:
      model_outputs: A dictionary with task-specific outputs.
    """
    vision_features = self.vision_model(image)
    text_paddings = text > PAD_ID

    # Reshape to be [-1, seq_len] so the LM can encode it, then reshape back.
    seq_length = text.shape[-1]
    reshaped_text = text.reshape(-1, seq_length)
    text_features = self.language_model(reshaped_text)
    # When language model encodes each token (e.g. T5).
    if text_features.shape[:-1] == reshaped_text.shape:
      text_features = text_features.reshape(text.shape + (-1,))
    # When language model encodes the whole sentence (e.g. CLIP).
    elif text_features.shape[:-1] == reshaped_text.shape[:-1]:
      text_features = text_features.reshape(text.shape[:-1] + (-1,))
    else:
      raise ValueError(
          'LM features shape inconsistent with input text shape:'
          '{} vs {}'.format(text_features.shape, text.shape))

    image_text_features, paddings = self.fusion_model(vision_features,
                                                      text_features,
                                                      text_paddings)
    model_outputs = {}
    for task, task_head in zip(self.tasks, self.task_heads):
      task_labels = task.filter_by_task(labels)
      task_outputs = task_head(vision_features, text_features,
                               image_text_features, paddings, task_labels)
      # Add task specific scope name.
      model_outputs.update(task.unfilter_by_task(task_outputs))

    return model_outputs

