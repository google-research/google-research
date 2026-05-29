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

# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper utilities to convert SAM 3 masks to COCO-format bounding boxes."""

import numpy as np


def mask_to_bbox(mask):
  """Convert a binary mask to COCO-format [x_min, y_min, width, height].

  Args:
    mask: A binary numpy array of shape (height, width).

  Returns:
    A list [x_min, y_min, width, height] or None if the mask is empty.
  """
  rows = np.any(mask, axis=1)
  cols = np.any(mask, axis=0)
  if not rows.any():
    return None  # Empty mask
  y_min, y_max = np.where(rows)[0][[0, -1]]
  x_min, x_max = np.where(cols)[0][[0, -1]]
  return [
      int(x_min),
      int(y_min),
      int(x_max - x_min + 1),
      int(y_max - y_min + 1),
  ]


def pcs_output_to_coco_predictions(masks, scores, image_id, category_id):
  """Convert PCS output for one concept to COCO prediction entries.

  Args:
    masks: A sequence/array of binary masks, each of shape (height, width).
    scores: A sequence of confidence scores matching the masks.
    image_id: The ID of the image containing the objects.
    category_id: The category ID of the predicted concept.

  Returns:
    A list of COCO prediction dictionaries.
  """
  predictions = []
  for mask, score in zip(masks, scores):
    bbox = mask_to_bbox(mask)
    if bbox is None:
      continue
    predictions.append({
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "score": float(score),
    })
  return predictions
