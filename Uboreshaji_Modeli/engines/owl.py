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

"""Owlv2 engine for object detection in composed architecture."""

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from absl import logging
import ml_collections
import numpy as np
import torch
from torch import nn
import transformers
from transformers import Owlv2ForObjectDetection
from transformers import Owlv2Processor


from Uboreshaji_Modeli.common import box_utils
from Uboreshaji_Modeli.common import config as base_config
from Uboreshaji_Modeli.common import losses
from Uboreshaji_Modeli.common import matcher
from Uboreshaji_Modeli.engines import base

Owlv2ForObjectDetection = transformers.Owlv2ForObjectDetection
Owlv2Processor = transformers.Owlv2Processor


def normalize_annotation_for_owlv2(
    boxes, original_size
):
  """Normalizes bounding boxes to [0, 1] relative to image size."""
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


class Owlv2Preprocessor(base.DataPreprocessor):
  """Preprocessor for OWL-v2 object detection."""

  def get_transform_fn(
      self,
      processor,
      cfg = None,
      *,
      is_train = False,
      **kwargs,
  ):
    from Uboreshaji_Modeli.common import augmentations  # pylint: disable=g-import-not-at-top

    # Unpack legacy arguments passed from ModelEngine coordinator
    text_inputs = kwargs.get("text_inputs")
    dataset_id2label = kwargs.get("dataset_id2label")
    model_label2id = kwargs.get("model_label2id")

    if (
        text_inputs is None
        or dataset_id2label is None
        or model_label2id is None
    ):
      raise ValueError(
          "text_inputs, dataset_id2label, and model_label2id are required in"
          " kwargs for Owlv2Preprocessor."
      )

    aug = None
    if is_train and cfg:
      aug = augmentations.get_train_augmentation(cfg)
      logging.info("Using data augmentation: %s", aug is not None)

    # Tokenize text inputs once outside the image loop for efficiency.
    text_encoding = processor(
        text=text_inputs,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    shared_input_ids = text_encoding["input_ids"].squeeze(0)
    shared_attention_mask = text_encoding["attention_mask"].squeeze(0)

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
        image = np.array(image.convert("RGB")).copy()

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

          new_labels = new_labels_aug if new_bboxes else []

        h, w = image.shape[:2]
        labels_dict = {}
        # Process only the image. Text features are already precomputed.
        pixel_values_encoded = processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        if not new_labels:
          labels_dict["class_labels"] = torch.zeros(0, dtype=torch.long)
          labels_dict["boxes"] = torch.zeros((0, 4))
        else:
          labels_dict["class_labels"] = torch.tensor(new_labels)
          labels_dict["boxes"] = normalize_annotation_for_owlv2(
              torch.tensor(new_bboxes), (h, w)
          )
        labels_dict["image_id"] = torch.tensor([image_id])

        input_ids.append(shared_input_ids)
        attention_masks.append(shared_attention_mask)
        labels.append(labels_dict)
        pixel_values.append(pixel_values_encoded)

      return {
          "input_ids": input_ids,
          "labels": labels,
          "pixel_values": pixel_values,
          "attention_mask": attention_masks,
      }

    return transform_fn  # pyrefly: ignore[bad-return]

  def get_collate_fn(
      self,
      cfg = None,
      **kwargs,
  ):
    """Returns a collate function for batching processed examples.

    Args:
      cfg: Optional configuration dictionary.
      **kwargs: Additional keyword arguments.
    Returns:
      A callable that takes a list of preprocessed examples and
      returns a collated batch.
    """

    del self  # Unused in this method.

    def collate_fn(batch):
      input_ids = torch.stack(
          [torch.as_tensor(item["input_ids"]) for item in batch], dim=0
      )
      attention_mask = torch.stack(
          [torch.as_tensor(item["attention_mask"]) for item in batch], dim=0
      )
      input_ids = input_ids.view(-1, input_ids.shape[-1])
      attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
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

      return {
          "input_ids": input_ids.long(),
          "attention_mask": attention_mask.long(),
          "pixel_values": pixel_values,
          "labels": labels,
      }

    return collate_fn


class Owlv2LossHandler(base.LossHandler):
  """Loss handler for OWL-v2 object detection (SetCriterion)."""

  def get_criterion(
      self,
      num_classes,
      cfg,
      device,
  ):
    del self  # Unused in this method.
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

  def post_process(
      self,
      processor,
      outputs,
      target_sizes,
      score_threshold,
  ):
    """Post-processes raw model outputs into standardised detections."""
    del self  # Unused in this method.
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
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

      mask = scores > score_threshold
      results.append(
          {"scores": scores[mask], "labels": labels[mask], "boxes": boxes[mask]}
      )

    return results


class Owlv2Engine(base.ModelEngine):
  """Composed Owlv2 Engine for object detection."""

  def __init__(self):
    super().__init__(
        preprocessor=Owlv2Preprocessor(),
        loss_handler=Owlv2LossHandler(),
        decoder=None,  # post_process is bundled in LossHandler for Owlv2
    )

  def load_model_and_processor(
      self,
      model_id,
      device,
      **kwargs,
  ):
    """Loads the pretrained OWL-v2 object detector model and its processor.

    Args:
      model_id: Pretrained repository ID or local path.
      device: PyTorch target mapping device.
      **kwargs: Additional load options.

    Returns:
      A tuple (model, processor), where model is the initialized
      Owlv2ForObjectDetection model on the device, and processor is the
      Owlv2Processor instance.
    """
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id)
    model.to(device)
    return model, processor
