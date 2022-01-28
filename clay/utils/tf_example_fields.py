# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Extension of standard_fields.TfExampleFields with more fields."""
from typing import Optional

from tensorflow_models.object_detection.core import standard_fields

METADATA_PREFIX = 'image/object/metadata/'
VIEW_HIERARCHY_PREFIX = 'image/view_hierarchy/'


class TfExampleFields(standard_fields.TfExampleFields):
  """Extension of TfExampleFields.

  Drop in replacement for standard_fields.TfExampleFields.

  Attributes:
    object_score: score for ground truth label expresing uncertainty.
    detection_class_text: detection class text same as object_class_text.
    predicted_class_label: class label for prediction/classification output.
    predicted_class_text: class text for prediction/classification output.
    predicted_class_score: score for prediction/classification output label.
    predicted_all_scores: scores for all classes.
    predicted_gt_score: score for ground truth class.
    ocr_prefix: common prefix for all OCR fields.
    ocr_text: Detected OCR text.
    ocr_confidence: Confidence score returned by OCR engine. Range [0,1].
    ocr_ocr_type: OCR prediction type. Enum defined by OcrType:
      WORD, PARAGRAPH, BLOCK.
    ocr_bbox_xmin: OCR bounding box xmin coordinate. Pixel int value.
    ocr_bbox_xmax: OCR bounding box xmax coordinate. Pixel int value.
    ocr_bbox_ymin: OCR bounding box ymin coordinate. Pixel int value.
    ocr_bbox_ymax: OCR bounding box ymax coordinate. Pixel int value.
  """
  object_score = 'image/object/score'
  detection_class_text = 'image/detection/text'
  predicted_class_label = 'predicted/image/class/label'
  predicted_class_text = 'predicted/image/class/text'
  predicted_class_score = 'predicted/image/class/score'
  predicted_all_scores = 'predicted/image/scores'
  predicted_gt_score = 'predicted/image/gt_score'

  ocr_prefix = 'image/ocr'
  ocr_text = 'image/ocr/text'
  ocr_confidence = 'image/ocr/confidence'
  ocr_ocr_type = 'image/ocr/type'
  ocr_bbox_xmin = 'image/ocr/bbox/xmin'
  ocr_bbox_xmax = 'image/ocr/bbox/xmax'
  ocr_bbox_ymin = 'image/ocr/bbox/ymin'
  ocr_bbox_ymax = 'image/ocr/bbox/ymax'


def prefix_model(field, model_name = None):
  """Prefixes field name with model name.

  This is intended to be used with 'predicted/image/...' fields.

  Args:
    field: original field to prefixed.
    model_name: Optional model name that should be prefixed. Can't contain '/'
      as it's tf.Example schema separator.

  Returns:
    Prefixed field name.
  """
  if not model_name:
    return field
  if '/' in model_name:
    raise ValueError('model_name should not contain "/" symbol.')
  return f'{model_name}/{field}'
