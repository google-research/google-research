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

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Util functions related to pycocotools and COCO eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from absl import logging

import numpy as np
from PIL import Image
from pycocotools import coco
from pycocotools import mask as mask_api
import six
from six.moves import range
from six.moves import zip

from utils import box_utils
from utils import mask_utils


class COCOWrapper(coco.COCO):
  """COCO wrapper class.

  This class wraps COCO API object, which provides the following additional
  functionalities:
    1. Support string type image id.
    2. Support loading the groundtruth dataset using the external annotation
       dictionary.
    3. Support loading the prediction results using the external annotation
       dictionary.
  """

  def __init__(self, eval_type='box', annotation_file=None, gt_dataset=None):
    """Instantiates a COCO-style API object.

    Args:
      eval_type: either 'box' or 'mask'.
      annotation_file: a JSON file that stores annotations of the eval dataset.
        This is required if `gt_dataset` is not provided.
      gt_dataset: the groundtruth eval datatset in COCO API format.
    """
    if ((annotation_file and gt_dataset) or
        ((not annotation_file) and (not gt_dataset))):
      raise ValueError('One and only one of `annotation_file` and `gt_dataset` '
                       'needs to be specified.')

    if eval_type not in ['box', 'mask']:
      raise ValueError('The `eval_type` can only be either `box` or `mask`.')

    coco.COCO.__init__(self, annotation_file=annotation_file)
    self._eval_type = eval_type
    if gt_dataset:
      self.dataset = gt_dataset
      self.createIndex()

  def loadRes(self, predictions):
    """Loads result file and return a result api object.

    Args:
      predictions: a list of dictionary each representing an annotation in COCO
        format. The required fields are `image_id`, `category_id`, `score`,
        `bbox`, `segmentation`.

    Returns:
      res: result COCO api object.

    Raises:
      ValueError: if the set of image id from predctions is not the subset of
        the set of image id of the groundtruth dataset.
    """
    res = coco.COCO()
    res.dataset['images'] = copy.deepcopy(self.dataset['images'])
    res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])

    image_ids = [ann['image_id'] for ann in predictions]
    if set(image_ids) != (set(image_ids) & set(self.getImgIds())):
      raise ValueError('Results do not correspond to the current dataset!')
    for ann in predictions:
      x1, x2, y1, y2 = [ann['bbox'][0], ann['bbox'][0] + ann['bbox'][2],
                        ann['bbox'][1], ann['bbox'][1] + ann['bbox'][3]]
      if self._eval_type == 'box':
        ann['area'] = ann['bbox'][2] * ann['bbox'][3]
        ann['segmentation'] = [
            [x1, y1, x1, y2, x2, y2, x2, y1]]
      elif self._eval_type == 'mask':
        ann['area'] = mask_api.area(ann['segmentation'])

    res.dataset['annotations'] = copy.deepcopy(predictions)
    res.createIndex()
    return res


def convert_predictions_to_coco_annotations(
    predictions, remove_invalid_boxes=False
):
  """Converts a batch of predictions to annotations in COCO format.

  "remove_invalid_boxes" is an indicator whether invalid boxes should be removed
  when evaluating on coco. Keeping invalid boxes may cause the groundtruth boxes
  matched to an invalid boxes, making the evaluation inaccurate. However, to
  make this function backward compatible, we set its default value to be false.
  Another way to avoid using invalid boxes when evaluating on COCO is to set the
  coordinates of invalid boxes to be "0" so they won't be matched to any
  groundtruth boxes.

  Args:
   predictions: a dictionary of lists of numpy arrays including the following
      fields. K below denotes the maximum number of instances per image.
      Required fields:
        - source_id: a list of numpy arrays of int or string of shape
            [batch_size].
        - num_detections: a list of numpy arrays of int of shape [batch_size].
        - detection_boxes: a list of numpy arrays of float of shape
            [batch_size, K, 4], where coordinates are in the original image
            space (not the scaled image space).
        - detection_classes: a list of numpy arrays of int of shape
            [batch_size, K].
        - detection_scores: a list of numpy arrays of float of shape
            [batch_size, K].
      Optional fields:
        - detection_masks: a list of numpy arrays of float of shape
            [batch_size, K, mask_height, mask_width].
    remove_invalid_boxes: A boolean indicating whether to remove invalid box
      during evaluation.

  Returns:
    coco_predictions: prediction in COCO annotation format.
  """
  coco_predictions = []
  num_batches = len(predictions['source_id'])
  max_num_detections = predictions['detection_classes'][0].shape[1]
  use_outer_box = 'detection_outer_boxes' in predictions
  for i in range(num_batches):
    predictions['detection_boxes'][i] = box_utils.yxyx_to_xywh(
        predictions['detection_boxes'][i])
    if use_outer_box:
      predictions['detection_outer_boxes'][i] = box_utils.yxyx_to_xywh(
          predictions['detection_outer_boxes'][i])
      mask_boxes = predictions['detection_outer_boxes']
    else:
      mask_boxes = predictions['detection_boxes']

    # NOTE: Batch size may differ between chunks.
    batch_size = predictions['source_id'][i].shape[0]
    for j in range(batch_size):
      if 'detection_masks' in predictions:
        image_masks = mask_utils.paste_instance_masks(
            predictions['detection_masks'][i][j],
            mask_boxes[i][j],
            int(predictions['image_info'][i][j, 0, 0]),
            int(predictions['image_info'][i][j, 0, 1]))
        binary_masks = (image_masks > 0.0).astype(np.uint8)
        encoded_masks = [
            mask_api.encode(np.asfortranarray(binary_mask))
            for binary_mask in list(binary_masks)]
      if remove_invalid_boxes:
        num_detections = min(
            predictions['num_detections'][i][j], max_num_detections
        )
      else:
        num_detections = max_num_detections
      for k in range(num_detections):
        ann = {}
        ann['image_id'] = predictions['source_id'][i][j]
        ann['category_id'] = predictions['detection_classes'][i][j, k]
        ann['bbox'] = predictions['detection_boxes'][i][j, k]
        ann['score'] = predictions['detection_scores'][i][j, k]
        if 'detection_masks' in predictions:
          ann['segmentation'] = encoded_masks[k]
        coco_predictions.append(ann)

  for i, ann in enumerate(coco_predictions):
    ann['id'] = i + 1

  return coco_predictions


def convert_groundtruths_to_coco_dataset(groundtruths, label_map=None):
  """Converts groundtruths to the dataset in COCO format.

  Args:
    groundtruths: a dictionary of numpy arrays including the fields below.
      Note that each element in the list represent the number for a single
      example without batch dimension. K below denotes the actual number of
      instances for each image.
      Required fields:
        - source_id: a list of numpy arrays of int or string of shape
          [batch_size].
        - height: a list of numpy arrays of int of shape [batch_size].
        - width: a list of numpy arrays of int of shape [batch_size].
        - num_detections: a list of numpy arrays of int of shape [batch_size].
        - boxes: a list of numpy arrays of float of shape [batch_size, K, 4],
            where coordinates are in the original image space (not the
            normalized coordinates).
        - classes: a list of numpy arrays of int of shape [batch_size, K].
      Optional fields:
        - is_crowds: a list of numpy arrays of int of shape [batch_size, K]. If
            th field is absent, it is assumed that this instance is not crowd.
        - areas: a list of numy arrays of float of shape [batch_size, K]. If the
            field is absent, the area is calculated using either boxes or
            masks depending on which one is available.
        - masks: a list of numpy arrays of string of shape [batch_size, K],
    label_map: (optional) a dictionary that defines items from the category id
      to the category name. If `None`, collect the category mappping from the
      `groundtruths`.

  Returns:
    coco_groundtruths: the groundtruth dataset in COCO format.
  """
  source_ids = np.concatenate(groundtruths['source_id'], axis=0)
  heights = np.concatenate(groundtruths['height'], axis=0)
  widths = np.concatenate(groundtruths['width'], axis=0)
  gt_images = [{'id': int(i), 'height': int(h), 'width': int(w)} for i, h, w
               in zip(source_ids, heights, widths)]

  gt_annotations = []
  num_batches = len(groundtruths['source_id'])
  for i in range(num_batches):
    # NOTE: Batch size may differ between chunks.
    batch_size = groundtruths['source_id'][i].shape[0]
    max_num_instances = groundtruths['classes'][i].shape[1]
    for j in range(batch_size):
      num_instances = int(groundtruths['num_detections'][i][j])
      if num_instances > max_num_instances:
        logging.warning(
            'num_groundtruths is larger than max_num_instances, %d v.s. %d',
            num_instances, max_num_instances)
        num_instances = max_num_instances
      for k in range(num_instances):
        ann = {}
        ann['image_id'] = int(groundtruths['source_id'][i][j])
        if 'is_crowds' in groundtruths:
          ann['iscrowd'] = int(groundtruths['is_crowds'][i][j, k])
        else:
          ann['iscrowd'] = 0
        ann['category_id'] = int(groundtruths['classes'][i][j, k])
        boxes = groundtruths['boxes'][i]
        ann['bbox'] = [
            float(boxes[j, k, 1]),
            float(boxes[j, k, 0]),
            float(boxes[j, k, 3] - boxes[j, k, 1]),
            float(boxes[j, k, 2] - boxes[j, k, 0])]
        if 'areas' in groundtruths:
          ann['area'] = float(groundtruths['areas'][i][j, k])
        else:
          ann['area'] = float(
              (boxes[j, k, 3] - boxes[j, k, 1]) *
              (boxes[j, k, 2] - boxes[j, k, 0]))
        if 'masks' in groundtruths:
          mask = Image.open(six.BytesIO(groundtruths['masks'][i][j, k]))
          np_mask = np.array(mask, dtype=np.uint8)
          np_mask[np_mask > 0] = 255
          encoded_mask = mask_api.encode(np.asfortranarray(np_mask))
          ann['segmentation'] = encoded_mask
          if 'areas' not in groundtruths:
            ann['area'] = mask_api.area(encoded_mask)
        gt_annotations.append(ann)

  for i, ann in enumerate(gt_annotations):
    ann['id'] = i + 1

  if label_map:
    gt_categories = [{'id': i, 'name': label_map[i]} for i in label_map]
  else:
    category_ids = [gt['category_id'] for gt in gt_annotations]
    gt_categories = [{'id': i} for i in set(category_ids)]

  gt_dataset = {
      'images': gt_images,
      'categories': gt_categories,
      'annotations': copy.deepcopy(gt_annotations),
  }
  return gt_dataset
