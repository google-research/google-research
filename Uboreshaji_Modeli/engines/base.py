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

"""Base classes and protocols for model engines."""

from collections.abc import Mapping, Sequence
import typing
from typing import Any, Callable, Protocol

import ml_collections
import torch
from torch import nn


@typing.runtime_checkable
class DataPreprocessor(Protocol):
  """Protocol for data preprocessing components."""

  def get_transform_fn(
      self,
      processor,
      cfg = None,
      *,
      is_train = False,
      **kwargs,
  ):
    """Returns the transformation function for the dataset."""
    Ellipsis

  def get_collate_fn(
      self,
      cfg = None,
      **kwargs,
  ):
    """Returns the collation function for the data loader."""
    Ellipsis


@typing.runtime_checkable
class LossHandler(Protocol):
  """Protocol for loss computation and post-processing."""

  def get_criterion(
      self,
      num_classes,
      cfg,
      device,
  ):
    """Returns the criterion (loss function) and loss weights."""
    Ellipsis

  def post_process(
      self,
      processor,
      outputs,
      target_sizes,
      score_threshold,
  ):
    """Post-processes model outputs (e.g. converting logits to boxes)."""
    Ellipsis


@typing.runtime_checkable
class PredictionDecoder(Protocol):
  """Protocol for prediction decoding components."""

  def decode(
      self,
      processor,
      outputs,
      **kwargs,
  ):
    """Decodes model outputs into readable formats."""
    Ellipsis


class ModelEngine:
  """Coordinator container for composed model engines.

  Can be instantiated directly with composed components (Composition Mode)
  or subclassed by legacy engines (Subclass Mode).
  """

  def __init__(
      self,
      backbone = None,
      preprocessor = None,
      loss_handler = None,
      decoder = None,
  ):
    self.backbone = backbone
    self.preprocessor = preprocessor
    self.loss_handler = loss_handler
    self.decoder = decoder

  # Core Composed & Legacy Adapter Interface

  def load_model_and_processor(
      self,
      model_id,
      device,
      **kwargs,
  ):
    """Loads model and processor. Delegates to composed backbone if present.

    Args:
      model_id: Model repository name or local path.
      device: Targeted hardware execution mapping device.
      **kwargs: Additional model-specific loading arguments.

    Returns:
      A tuple (model, processor), where model is the initialized neural network
      module on the device, and processor is the input data processing component
      instance.
    """
    if self.backbone is not None:
      if hasattr(self.backbone, "load_model_and_processor"):
        return self.backbone.load_model_and_processor(
            model_id, device, **kwargs
        )
    raise NotImplementedError(
        "load_model_and_processor must be overridden by subclass or delegated"
        " to a composed backbone."
    )

  def get_transform_fn(
      self,
      processor,
      text_inputs,
      dataset_id2label,
      model_label2id,
      cfg = None,
      is_train = False,
      **kwargs,
  ):
    """Delegates to composed preprocessor or handles legacy signature."""
    if self.preprocessor is not None:
      # Composition mode: expects cfg to be passed as kwarg or positional.
      resolved_cfg = cfg if cfg is not None else kwargs.get("cfg")
      return self.preprocessor.get_transform_fn(
          processor,
          resolved_cfg,
          is_train=is_train,
          text_inputs=text_inputs,
          dataset_id2label=dataset_id2label,
          model_label2id=model_label2id,
          **kwargs
      )

    raise NotImplementedError(
        "get_transform_fn must be overridden by subclass or delegated to a"
        " composed preprocessor."
    )

  def get_collate_fn(
      self,
      cfg = None,
      **kwargs,
  ):
    """Delegates to composed preprocessor."""
    if self.preprocessor is not None:
      return self.preprocessor.get_collate_fn(cfg, **kwargs)
    raise NotImplementedError(
        "get_collate_fn must be overridden by subclass or delegated to a"
        " composed preprocessor."
    )

  def get_criterion(
      self,
      num_classes,
      cfg,
      device,
      **kwargs,
  ):
    """Delegates to composed loss handler."""
    del kwargs  # Unused in this method.
    if self.loss_handler is not None:
      return self.loss_handler.get_criterion(num_classes, cfg, device)
    raise NotImplementedError(
        "get_criterion must be overridden by subclass or delegated to a"
        " composed loss handler."
    )

  def post_process(
      self,
      processor,
      outputs,
      target_sizes,
      score_threshold,
      **kwargs,
  ):
    """Delegates to composed loss handler."""
    del kwargs  # Unused in this method.
    if self.loss_handler is not None:
      return self.loss_handler.post_process(
          processor, outputs, target_sizes, score_threshold
      )
    raise NotImplementedError(
        "post_process must be overridden by subclass or delegated to a"
        " composed loss handler."
    )

  def decode_predictions(
      self,
      processor,
      outputs,
      **kwargs,
  ):
    """Delegates to composed prediction decoder."""
    if self.decoder is not None:
      return self.decoder.decode(processor, outputs, **kwargs)
    raise NotImplementedError(
        "decode_predictions must be overridden by subclass or delegated to a"
        " composed prediction decoder."
    )

  def freeze_vision_tower(self, model):
    """Freezes the vision tower of the model. Default is no-op."""
    if self.backbone is not None and hasattr(
        self.backbone, "freeze_vision_tower"
    ):
      self.backbone.freeze_vision_tower(model)

  def get_sft_config_overrides(
      self, cfg
  ):
    """Returns model-specific SFT config overrides. Default is empty dict."""
    if self.preprocessor is not None and hasattr(
        self.preprocessor, "get_sft_config_overrides"
    ):
      return self.preprocessor.get_sft_config_overrides(cfg)
    return {}

