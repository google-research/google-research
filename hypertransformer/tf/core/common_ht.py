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

"""Configuration datastructures."""
import dataclasses
import enum
import functools

from typing import List, Optional

import tensorflow.compat.v1 as tf


@dataclasses.dataclass
class DatasetInfo:
  num_labels: int
  num_samples_per_label: Optional[int]
  transpose_images: bool = False


class Model:
  """Model abstract class."""

  def __init__(self):
    self.layer_outputs = {}

  def train(self, inputs, labels):
    raise NotImplementedError

  def evaluate(self, inputs, training = False):
    raise NotImplementedError


@dataclasses.dataclass
class DatasetConfig:
  """Specification of the train or test dataset."""

  # Dataset information
  dataset_name: str
  use_label_subset: Optional[List[int]] = None
  tfds_split: str = 'train'
  meta_dataset_split: str = 'train'
  data_dir: Optional[str] = None
  ds: Optional[tf.data.Dataset] = None
  dataset_info: Optional[DatasetInfo] = None
  per_label_augmentation: bool = False
  cache_path: str = ''
  balanced_batches: bool = False
  shuffle_labels_seed: int = 0
  # Number of unlabeled samples in the training batch (per label)
  num_unlabeled_per_class: int = 0

  # Data augmentation
  rotation_probability: float = 0.0
  smooth_probability: float = 0.5
  contrast_probability: float = 0.5
  resize_probability: float = 0.0
  negate_probability: float = 0.0
  roll_probability: float = 0.0
  angle_range: float = 180.0
  rotate_by_90: bool = False
  apply_image_augmentations: bool = False
  augment_individually: bool = False


@dataclasses.dataclass
class ModelConfig:
  """Common model configuration."""

  # Label parameters
  num_labels: int = 4
  embedding_dim: int = 8

  # Samples and images
  image_size: Optional[int] = 28
  num_transformer_samples: int = 32
  num_cnn_samples: int = 128

  # CNN parameters
  cnn_dropout_rate: float = 0.0
  cnn_model_name: str = '3-layer'
  use_decoder: bool = True
  add_trainable_weights: bool = False
  var_reg_weight: float = 0.0
  cnn_activation: str = 'relu'
  default_num_channels: int = 16

  # Common transformer parameters
  transformer_activation: str = 'softmax'
  shared_fe_dropout: float = 0.0
  fe_dropout: float = 0.0
  transformer_nonlinearity: str = 'relu'


class WeightAllocation(enum.Enum):
  """Type of the weight allocation by the Transformer."""
  SPATIAL = 1
  OUTPUT_CHANNEL = 2


@dataclasses.dataclass
class LayerwiseModelConfig(ModelConfig):
  """Laywerwise model configuration."""
  # -- Feature extractor parameters
  feature_layers: int = 2
  num_features: Optional[int] = None
  nonlinear_feature: bool = False
  shared_feature_extractor: str = ''
  shared_features_dim: int = 32
  shared_feature_extractor_padding: str = 'valid'
  number_of_trained_cnn_layers: int = 0
  # By default, the logits feature extractor is identical to the convolutional
  # layer feature extractor (typically, based on one or two convolutions applied
  # to the previous activations).
  logits_feature_extractor: str = ''
  number_of_trained_resnet_blocks: int = 0
  # -- Transformer parameters
  query_key_dim_frac: float = 1.0
  value_dim_frac: float = 1.0
  internal_dim_frac: float = 1.0
  num_layers: int = 3
  heads: int = 1
  kernel_size: int = 3
  stride: int = 2
  dropout_rate: float = 0.0
  weight_allocation: WeightAllocation = WeightAllocation.SPATIAL
  generate_bn: bool = False
  generate_bias: bool = False
  generator: str = 'joint'
  skip_last_nonlinearity: bool = False
  l2_reg_weight: Optional[float] = None
  shared_head_weight: float = 0.0

  # -- Additional weight generation parameters

  # If set, free variables (like BatchNorm beta and gamma) are not shared
  # between the CNN generating the weights and CNN computing the final
  # predictions.
  separate_bn_vars: bool = False
  train_heads: bool = False
  max_prob_remove_unlabeled: float = 0
  max_prob_remove_labeled: float = 0


@dataclasses.dataclass
class DatasetSamples:
  """Dataset samples."""
  transformer_images: tf.Tensor
  transformer_labels: tf.Tensor
  transformer_real_classes: Optional[tf.Tensor]
  cnn_images: tf.Tensor
  cnn_labels: tf.Tensor
  cnn_real_classes: Optional[tf.Tensor]
  randomize_op: tf.Operation
  transformer_masks: Optional[tf.Tensor] = None
  real_class_min: Optional[int] = None
  real_class_max: Optional[int] = None


def get_cnn_activation(config):
  if config.cnn_activation == 'relu':
    return tf.nn.relu
  elif config.cnn_activation == 'lrelu':
    # This alpha value is used in the HowToTrainYourMAML code
    return functools.partial(tf.nn.leaky_relu, alpha=0.01)
  else:
    raise ValueError(f'Unknown CNN nonlinearity {config.cnn_activation}.')


def get_transformer_activation(config):
  if config.transformer_activation == 'softmax':
    return tf.nn.softmax
  elif config.transformer_activation == 'sigmoid':
    return tf.nn.sigmoid
  else:
    raise ValueError(
        f'Unknown Transformer nonlinearity {config.transformer_activation}.')
