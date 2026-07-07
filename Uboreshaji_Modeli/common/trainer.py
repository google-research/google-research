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

"""Trainer and loss logic adapted from DETR."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
import transformers

from Uboreshaji_Modeli.common import losses


class CustomTrainer(transformers.Trainer):
  """A custom HuggingFace Trainer for fine tuning using a SetCriterion loss.

  This trainer adapts the standard `transformers.Trainer` to incorporate
  the specialized loss computation required for object detection tasks,
  as defined by the `SetCriterion`.

  Attributes:
    criterion: An instance of `SetCriterion` that computes the various losses
      (e.g., classification, bounding box, GIoU).
    weight_dict: A dictionary containing the weights for each loss component
      computed by the `criterion`.
  """

  def __init__(
      self,
      *args,
      criterion = None,
      weight_dict = None,
      **kwargs,
  ):
    """Initializes the instance."""
    super().__init__(*args, **kwargs)
    self.criterion = criterion
    self.weight_dict = weight_dict
    self._loss_components_buffer = []
    self._eval_loss_components_buffer = []

  def _run_forward(self, model, inputs):
    """Unified forward pass; returns raw logits for processing."""
    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        interpolate_pos_encoding=True,
        return_dict=True,
    )

    return {"logits": outputs.logits, "pred_boxes": outputs.pred_boxes}

  def compute_loss(  # pyrefly: ignore[bad-override]
      self,
      model,
      inputs,
      return_outputs = False,
      **kwargs,
  ):
    """Computes the loss for the model inputs."""
    del kwargs
    labels = inputs.pop("labels")

    try:
      # The inputs are passed directly to the model as prepared
      # by the data collator.
      outputs = model(**inputs, interpolate_pos_encoding=True, return_dict=True)

      loss_dict = self.criterion(outputs, labels)  # pyrefly: ignore[not-callable]
      self._loss_components_buffer.append(
          {k: v.detach().cpu() for k, v in loss_dict.items()}
      )

      loss = sum(
          loss_dict[k] * self.weight_dict[k]  # pyrefly: ignore[unsupported-operation]
          for k in loss_dict.keys()
          if k in self.weight_dict
      )
    finally:
      # Restore original inputs (e.g., for metrics)
      inputs["labels"] = labels

    return (loss, outputs) if return_outputs else loss

  def prediction_step(
      self,
      model,
      inputs,
      prediction_loss_only,
      ignore_keys = None,
  ):
    """Performs an evaluation step on `model` using `inputs`."""
    labels = inputs.pop("labels")

    with torch.no_grad():
      with self.autocast_smart_context_manager():
        outputs = self._run_forward(model, inputs)
      loss_dict = self.criterion(outputs, labels)  # pyrefly: ignore[not-callable]
      self._eval_loss_components_buffer.append(
          {k: v.detach().cpu() for k, v in loss_dict.items()}
      )
      loss = sum(
          loss_dict[k] * self.weight_dict[k]  # pyrefly: ignore[unsupported-operation]
          for k in loss_dict.keys()
          if k in self.weight_dict
      )

    # Restore labels
    inputs["labels"] = labels

    if prediction_loss_only:
      return (loss, None, None)

    b = len(labels)
    max_n = max((l["boxes"].shape[0] for l in labels), default=0)
    device = outputs["logits"].device

    padded_boxes = torch.zeros((b, max_n, 4), device=device)
    padded_classes = torch.full(
        (b, max_n), -100, device=device, dtype=torch.long
    )
    num_boxes = torch.zeros((b, 1), device=device, dtype=torch.long)

    for i, label in enumerate(labels):
      n = label["boxes"].shape[0]
      if n > 0:
        padded_boxes[i, :n] = label["boxes"]
        padded_classes[i, :n] = label["class_labels"]
      num_boxes[i] = n

    return (
        loss,
        (outputs["logits"], outputs["pred_boxes"]),
        {
            "boxes": padded_boxes,
            "class_labels": padded_classes,
            "num_boxes": num_boxes,
        },
    )

  def log(
      self, logs, start_time = None
  ):
    """Logs the training metrics, including detailed loss components."""
    if self._loss_components_buffer:
      # Average the buffered loss components
      num_steps = len(self._loss_components_buffer)
      avg_losses = {}
      for components in self._loss_components_buffer:
        for k, v in components.items():
          avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

      for k in avg_losses:
        logs[f"train_{k}"] = avg_losses[k] / num_steps

      self._loss_components_buffer.clear()

    if "eval_loss" in logs and self._eval_loss_components_buffer:
      num_steps = len(self._eval_loss_components_buffer)
      avg_losses = {}
      for components in self._eval_loss_components_buffer:
        for k, v in components.items():
          avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

      for k in avg_losses:
        logs[f"eval_{k}"] = avg_losses[k] / num_steps

      self._eval_loss_components_buffer.clear()

    super().log(logs, start_time)
