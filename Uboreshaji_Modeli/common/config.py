# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Unified configuration system for Poly-Sense2 experiments."""

import enum

import ml_collections

from Uboreshaji_Modeli.common import config_utils


class ModelFlavor(str, enum.Enum):
  OWL_V2_TORCH = "OWL_V2_TORCH"


class Precision(str, enum.Enum):
  BF16 = "bf16"
  FP32 = "fp32"


class TaskModality(str, enum.Enum):
  VISION = "VISION"
  TEXT = "TEXT"
  AUDIO = "AUDIO"


class TaskType(str, enum.Enum):
  DETECTION = "DETECTION"


class MatcherType(str, enum.Enum):
  GREEDY = "GREEDY"
  HUNGARIAN = "HUNGARIAN"


def get_base_config():
  """Returns the base experiment configuration for Poly-Sense2."""
  config = ml_collections.ConfigDict()

  config.experiment_name = "owl_finetune"
  config.model_flavor = ModelFlavor.OWL_V2_TORCH
  config.task_modality = TaskModality.VISION
  config.task_type = TaskType.DETECTION
  config.model_id = "/path/to/models/owlv2"
  config.output_dir = "/path/to/models/owl_hf"

  # Matcher specialized settings
  config.matcher = ml_collections.ConfigDict()
  config.matcher.matcher_type = MatcherType.HUNGARIAN
  config.matcher.cost_class = 2.459
  config.matcher.cost_bbox = 2.406
  config.matcher.cost_giou = 1.848

  # Detection specialized settings
  config.detection = ml_collections.ConfigDict()
  config.detection.weight_sigmoid_focal = 1.233
  config.detection.weight_bbox = 2.671
  config.detection.weight_giou = 1.045
  config.detection.eos_coef = 0.3153
  config.detection.losses = ["labels", "boxes", "cardinality"]
  config.detection.focal_loss_alpha = 0.25
  config.detection.focal_loss_gamma = 5.0

  # Training hyperparameters
  config.training = ml_collections.ConfigDict()
  config.training.learning_rate = 1e-5
  config.training.batch_size = 1
  config.training.num_train_epochs = 150
  config.training.gradient_accumulation_steps = 16
  config.training.seed = 42
  config.training.data_seed = 42
  config.training.gradient_checkpointing = True
  config.training.weight_decay = 1e-4
  config.training.precision = Precision.BF16
  config.training.logging_steps = 100
  config.training.eval_steps = 200
  config.training.save_steps = 200
  config.training.save_total_limit = 2
  config.training.save_strategy = "steps"
  config.training.lr_scheduler_type = "linear"
  config.training.warmup_ratio = 0.0
  config.training.max_grad_norm = 1.0

  # Data settings
  config.dataset = ml_collections.ConfigDict()
  config.dataset.dataset_uri = "/path/to/huggingface_dataset"
  config.dataset.dataset_version = "dataset_version"
  config.dataset.dataset_base = ""
  config.dataset.train_split = "train"
  config.dataset.eval_split = "valid"
  config.dataset.image_size = 840
  config.dataset.exclude_classes = []

  config.augmentation = ml_collections.ConfigDict()
  config.augmentation.enabled = True
  config.augmentation.horizontal_flip_p = 0.5
  config.augmentation.vertical_flip_p = 0.5
  config.augmentation.rotate90_p = 0.5
  config.augmentation.color_jitter_p = 0.8
  config.augmentation.color_jitter_brightness = 0.4
  config.augmentation.color_jitter_contrast = 0.4
  config.augmentation.color_jitter_saturation = 0.3
  config.augmentation.color_jitter_hue = 0.1
  config.augmentation.gaussian_blur_p = 0.3
  config.augmentation.random_crop_enabled = True
  config.augmentation.random_crop_p = 0.5
  config.augmentation.random_crop_scale = (0.6, 1.0)

  config.eval = ml_collections.ConfigDict()
  config.eval.run_eval_only = False
  config.eval.split = "test"
  config.eval.eval_batch_size = 4
  config.eval.confidence_threshold = 0.3
  config.eval.eval_json = ""

  # Platform and Hardware settings
  config.platform = ml_collections.ConfigDict()

  config.sweeps = ml_collections.ConfigDict()
  return config
