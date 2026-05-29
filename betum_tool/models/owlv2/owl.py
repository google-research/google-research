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

"""OWL-v2 implementation of ModelEngine."""

from collections.abc import Mapping
from typing import Any, Callable

from absl import logging
from common import augmentations
import ml_collections
from models.owlv2 import box_utils
from models.owlv2 import config as base_config
from models.owlv2 import losses
from models.owlv2 import matcher
import numpy as np
import torch
from torch import nn
import transformers


def normalize_annotation_for_owlv2(
    boxes, original_size
):
  """Normalizes bounding boxes for the OWL-v2 model.

  Converts boxes from [xmin, ymin, width, height] to the normalized
  [center_x, center_y, width, height] format expected by OWL-v2. Coordinates
  are normalized to [0, 1] relative to the longest edge of the image to match
  the model's square canvas padding.

  Args:
    boxes: A torch.Tensor of shape (N, 4) or (4,) with bounding boxes in [xmin,
      ymin, width, height] pixel format.
    original_size: A tuple (height, width) of the original image.

  Returns:
    A torch.Tensor of the same shape as `boxes` with normalized bounding
    boxes in [center_x, center_y, width, height] format.

  Example:
    A 100x200 box at (10, 20) in a 1000x800 image:
      - box: [10, 20, 100, 200]
      - original_size: (1000, 800)
      - max_side: 1000
      - cx = (10 + 100 / 2) / 1000 = 0.06
      - cy = (20 + 200 / 2) / 1000 = 0.12
      - w = 100 / 1000 = 0.1
      - h = 200 / 1000 = 0.2
    Returns: [0.06, 0.12, 0.1, 0.2]
  """
  orig_h, orig_w = original_size
  max_side = float(max(orig_h, orig_w))

  xmin, ymin, w_orig, h_orig = boxes.unbind(-1)

  return torch.stack(
      [
          (xmin + w_orig / 2) / max_side,
          (ymin + h_orig / 2) / max_side,
          w_orig / max_side,
          h_orig / max_side,
      ],
      dim=-1,
  )


class Owlv2Engine:
  """Model engine for OWL-v2.

  This class implements the base.ModelEngine interface for OWL-v2. It provides
  methods for loading the model and processor, transforming the data, and
  computing the loss.
  """

  def load_model_and_processor(
      self, model_id, device
  ):
    processor = transformers.AutoProcessor.from_pretrained(model_id)
    model = transformers.Owlv2ForObjectDetection.from_pretrained(model_id).to(
        device
    )
    return model, processor

  def get_transform_fn(
      self,
      processor,
      text_inputs,
      dataset_id2label,
      model_label2id,
      cfg = None,
      is_train = False,
  ):
    """Returns the transformation function for the dataset."""

    aug = None
    if is_train and cfg:
      aug = augmentations.get_train_augmentation(cfg)
      logging.info("Using data augmentation: %s", aug is not None)

    def transform_fn(
        examples,
    ):
      input_ids = []
      attention_masks = []
      pixel_values = []
      labels = []
      for image_id, image, objects in zip(
          examples["image_id"], examples["image"], examples["objects"]
      ):
        image = np.array(image.convert("RGB"))

        new_labels = []
        new_bboxes = []
        for category_id, bbox in zip(objects["category"], objects["bbox"]):
          category_name = dataset_id2label[category_id]
          if category_name in model_label2id:
            if box_utils.is_valid_box(bbox, box_format="xywh"):
              new_labels.append(model_label2id[category_name])
              new_bboxes.append(bbox)

        if aug:
          image, new_bboxes, new_labels_aug = augmentations.apply_augmentation(
              aug, image, new_bboxes, new_labels
          )
          # Ensure the image is contiguous and has positive strides after
          # augmentations
          image = np.ascontiguousarray(image)
          # filter out cases where all bboxes were dropped
          if not new_bboxes:
            continue
          new_labels = new_labels_aug

        h, w = image.shape[:2]  # height/width might change with crop
        labels_dict = {}
        owl_dict = processor(
            text=text_inputs,
            images=image,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if not new_labels:
          # If no foreground objects are matched, create empty tensors.
          labels_dict["class_labels"] = torch.zeros(0, dtype=torch.long)
          labels_dict["boxes"] = torch.zeros((0, 4))
        else:
          labels_dict["class_labels"] = torch.tensor(new_labels)
          labels_dict["boxes"] = normalize_annotation_for_owlv2(
              torch.tensor(new_bboxes), (h, w)
          )
        labels_dict["image_id"] = torch.tensor([image_id])

        owl_dict["labels"] = labels_dict

        input_ids.append(owl_dict["input_ids"].squeeze(0))
        attention_masks.append(owl_dict["attention_mask"].squeeze(0))
        labels.append(owl_dict["labels"])
        pixel_values.append(owl_dict["pixel_values"].squeeze(0))

      return {
          "input_ids": input_ids,
          "labels": labels,
          "pixel_values": pixel_values,
          "attention_mask": attention_masks,
      }

    return transform_fn

  def get_collate_fn(
      self,
      cfg = None,
  ):
    """Returns the PyTorch DataLoader collate function."""

    def collate_fn(batch):
      input_ids = torch.stack(
          [torch.as_tensor(item["input_ids"]) for item in batch], dim=0
      )
      attention_mask = torch.stack(
          [torch.as_tensor(item["attention_mask"]) for item in batch], dim=0
      )
      pixel_values = torch.stack(
          [torch.as_tensor(item["pixel_values"]) for item in batch], dim=0
      )
      if cfg and cfg.training.precision == base_config.Precision.BF16:
        pixel_values = pixel_values.bfloat16()

      labels = []
      for item in batch:
        processed_labels = {}
        for key, value in item["labels"].items():
          processed_labels[key] = torch.as_tensor(value)
        labels.append(processed_labels)

      _, _, seq_len = input_ids.shape
      final_input_ids = input_ids.view(-1, seq_len)
      final_attention_mask = attention_mask.view(-1, seq_len)

      return {
          "input_ids": final_input_ids.int(),
          "attention_mask": final_attention_mask.int(),
          "pixel_values": pixel_values,
          "labels": labels,
      }

    return collate_fn

  def get_criterion(
      self,
      num_classes,
      cfg,
      device,
  ):
    """Returns the criterion and weight dictionary for the loss."""
    if cfg.matcher.matcher_type == base_config.MatcherType.HUNGARIAN:
      matcher_instance = matcher.HungarianMatcher(
          cost_class=cfg.matcher.cost_class,
          cost_bbox=cfg.matcher.cost_bbox,
          cost_giou=cfg.matcher.cost_giou,
      )
    elif cfg.matcher.matcher_type == base_config.MatcherType.GREEDY:
      matcher_instance = matcher.GreedyMatcher(
          cost_class=cfg.matcher.cost_class,
          cost_bbox=cfg.matcher.cost_bbox,
          cost_giou=cfg.matcher.cost_giou,
      )
    else:
      raise ValueError(f"Unsupported matcher type: {cfg.matcher.matcher_type}")

    logging.info("Using matcher: %s", matcher_instance.__class__.__name__)

    weight_dict = {
        "loss_sigmoid_focal": cfg.detection.weight_sigmoid_focal,
        "loss_bbox": cfg.detection.weight_bbox,
        "loss_giou": cfg.detection.weight_giou,
    }

    criterion = losses.SetCriterion(
        num_classes=num_classes,
        matcher=matcher_instance,
        weight_dict=weight_dict,
        eos_coef=cfg.detection.eos_coef,
        losses=cfg.detection.losses,
        focal_alpha=cfg.detection.focal_loss_alpha,
        focal_gamma=cfg.detection.focal_loss_gamma,
    ).to(device)

    return criterion, weight_dict

  @property
  def inference_kwargs(self):
    """Returns model-specific inference keyword arguments."""
    return {"interpolate_pos_encoding": True}

  def post_process(
      self,
      processor,
      outputs,
      target_sizes,
      score_threshold,
  ):
    """Post-processes raw model outputs into standardised detections."""
    batch_logits = outputs.logits
    batch_boxes = outputs.pred_boxes
    probs = batch_logits.sigmoid()
    batch_scores, batch_labels = torch.max(probs, dim=-1)

    batch_boxes = box_utils.box_cxcywh_to_xyxy(batch_boxes)

    results = []
    for i, (scores, labels, boxes) in enumerate(
        zip(batch_scores, batch_labels, batch_boxes)
    ):
      if target_sizes is not None:
        h, w = target_sizes[i].tolist()
        max_size = float(max(h, w))
        scale_fct = torch.tensor(
            [max_size, max_size, max_size, max_size], device=boxes.device
        )
        boxes = boxes * scale_fct
        # Clamp to remove white square padding and keep boxes inside image.
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

      mask = scores > score_threshold
      results.append(
          {"scores": scores[mask], "labels": labels[mask], "boxes": boxes[mask]}
      )

    return results
