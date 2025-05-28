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

"""Evaluation metrics for MSCOCO dataset."""
import dataclasses
import enum
import io
from typing import Any, Dict, List, Optional, Text, Tuple, Union

from absl import logging
from clu import metrics
import flax
import gin
import jax
import jax.numpy as jnp
from metrics import coco_evaluator
import numpy as np
from PIL import Image
from utils import gin_utils

Array = jnp.ndarray


@gin.constants_from_enum
class EvaluatorName(enum.Enum):
  COCO = 'coco'
  LVIS = 'lvis'
  ZSDCOCO = 'zsdcoco'


@gin.configurable
def get_host_evaluator(
    evaluator_name,
    annotation_file,
    include_mask = False):
  """Get the host evaluator for detection metrics.

  Args:
    evaluator_name: the name of the evaluator.
    annotation_file: A string of annotation file path.
    include_mask: Whether to evaluate masks.

  Returns:
    The evaluator class for the specified evaluator.

  """
  if evaluator_name == EvaluatorName.COCO:
    evaluator_cls = coco_evaluator.COCOEvaluator
  else:
    raise ValueError('{} is not a supported evaluator'.format(evaluator_name))

  class TaskEvaluator(evaluator_cls):
    """Task evaluator API."""

    def task_evaluate(self,):
      """Evaluate the task and log/save the results."""
      output_metrics = self.evaluate()
      # Log out the metrics.
      logging.info('Detection metrics:')
      logging.info('=======================')
      for k, v in output_metrics.items():
        logging.info('%s: %f', k, v)


      return output_metrics

    def filter_by_source_ids(self, predictions):
      """Filter the predictions by existing source ids to handle paddings."""
      output_source_ids = predictions['source_id'].reshape(-1)
      overlap_indices = np.in1d(output_source_ids,
                                np.array(list(self._existing_source_ids)))
      new_source_ids = set(output_source_ids[~overlap_indices].tolist())
      logging.info('New source ids: %s', str(new_source_ids))
      self._existing_source_ids = self._existing_source_ids.union(
          new_source_ids)
      # Remove the predictions of padded examples.
      if np.any(overlap_indices):
        logging.info('Overlap source ids: %s',
                     str(output_source_ids[overlap_indices]))
        for k, v in predictions.items():
          if isinstance(v, np.ndarray):
            predictions[k] = v[~overlap_indices]
          else:
            logging.warning('Key %s has non-array value %s!!', k, str(type(v)))

      return predictions

    def task_update(self, predictions):
      """Process the predictions before update."""
      if 'detection' in predictions:
        predictions = predictions['detection']  # Get task scope.
      predictions = jax.tree.map(np.array, predictions)
      predictions = jax.tree.map(flatten_leading_dims, predictions)
      predictions = self.filter_by_source_ids(predictions)
      self.update(predictions)

    def reset(self):
      """Resets internal states for a fresh run."""
      self._existing_source_ids = set()
      self._predictions = {}
      if not self._annotation_file:
        self._groundtruths = {}

  return TaskEvaluator(
      annotation_file=annotation_file,
      include_mask=include_mask,
      need_rescale_bboxes=True,
      per_category_metrics=False)


@gin.configurable
def get_annotation_file(
    annotation_file = None):
  """Get annotation file for evaluation.

  Args:
    annotation_file: A string of annotation file path or None.

  Returns:
    annotation_file: A string of annotation file path or None.
  """
  return annotation_file


@gin.configurable
def get_evaluator(
    evaluator_name = EvaluatorName.COCO):
  """Get the coco evaluator needed to compute metrics.

  Args:
    evaluator_name: the name of the evaluator

  Returns:
    The evaluator class for the specified evaluator.

  """

  if evaluator_name == EvaluatorName.COCO:
    return coco_evaluator.COCOEvaluator
  else:
    raise ValueError('{} is not a supported evaluator'.format(evaluator_name))


class ValueType(enum.IntEnum):
  """Class for specific values that relies on int for JAX pmap compatibility."""
  NONE = -1


def decode_xla_masks(bytemask_array,
                     gt_mask_size,
                     max_gt_mask_per_image = 100,
                     pad_value = -1.0):
  """Decode the byte string groundtruth masks to fixed-shape jax numpy arrays.

  Args:
    bytemask_array: A list of byte strings of groundtruth masks.
    gt_mask_size: A 2-tuple of groundtruth mask size to resize to. This is
      necessary for XLA compatibility.
    max_gt_mask_per_image: An integer to pad the number of groundtruth masks
      of each image to this number.
    pad_value: The value to pad.

  Returns:
    A jnp.array of shape (1, max_gt_mask_per_image) + gt_mask_size
  """
  masks = []
  for byte_mask in bytemask_array:
    full_mask = decode_and_resize_mask(byte_mask, gt_mask_size)
    masks.append(full_mask[None, :, :])

  num_masks = len(masks)
  if num_masks > max_gt_mask_per_image:
    raise ValueError(
        'Number of GT masks exceed maximum: '
        f'{num_masks} vs {max_gt_mask_per_image}')
  output_masks = np.ones((max_gt_mask_per_image,) + gt_mask_size) * pad_value
  output_masks[:num_masks] = np.concatenate(masks, axis=0)

  return jnp.array(output_masks[None, Ellipsis])


def encode_xla_masks(output_masks,
                     heights,
                     widths
                     ):
  """Encode the jnp.array of output masks to an array of bytestrings.

  This is necessary for compatibility with COCO API.

  Args:
    output_masks: An array of shape [num_images, max_gt_mask_per_image] +
      gt_mask_size, where the parameters are defined in the docstring of
      'decode_xla_masks' function.
    heights: A 1-D array of heights [num_images].
    widths: A 1-D array of widths [num_images].

  Returns:
    mask_batch_byte_array: A numpy array of shape
      [num_images, max_gt_mask_per_image], where each element is the encoded
      byte string of mask.
  """
  if isinstance(heights, ValueType) or isinstance(widths, ValueType):
    return ValueType.NONE

  if heights.size != widths.size:
    raise ValueError('Heights and widths must have equal length')

  if heights.ndim != 1 or widths.ndim != 1:
    raise ValueError('Heights and widths must be 1-dimensional')

  mask_batch_byte_array = []
  for image_masks, height, width in zip(output_masks, heights, widths):
    mask_image_byte_array = []
    for np_mask in image_masks:
      byte_array = resize_and_encode_mask(np_mask.astype(np.uint8),
                                          width, height, mask_format='PNG')
      mask_image_byte_array.append(byte_array)

    mask_batch_byte_array.append(mask_image_byte_array)

  return np.array(mask_batch_byte_array)


def decode_and_resize_mask(byte_mask,
                           target_size):
  """Decode a byte mask and resize to target size.

  Args:
    byte_mask: A byte string of mask.
    target_size: A 2-tuple of mask target reshape size.

  Returns:
    np_mask: An array of resized decoded mask.
  """
  width, height = target_size
  mask = Image.open(io.BytesIO(byte_mask))
  mask = mask.resize((width, height), resample=Image.Resampling.BILINEAR)
  np_mask = np.array(mask, np.uint8)
  np_mask[np_mask > 0] = 255
  return np_mask


def resize_and_encode_mask(np_mask,
                           width,
                           height,
                           mask_format = 'PNG'):
  """Resize the decoded mask back to original size and encode back to bytes.

  Args:
    np_mask: Array of decoded mask.
    width: original width.
    height: original height.
    mask_format: A string of the format of original mask to encode back into.

  Returns:
    A string of encoded bytes.
  """
  pil_im = Image.fromarray(np.asarray(np_mask))
  pil_im.resize((width, height))
  byte_array = io.BytesIO()
  pil_im.save(byte_array, format=mask_format)
  return byte_array.getvalue()


def rescale_detection_boxes(detection_boxes, image_info):
  """Rescale detection boxes to original image space.

  Args:
    detection_boxes: Detection boxes of shape [batch, num_detections, 4].
    image_info: An array that encodes the information of the image and the
      applied preprocessing. The shape is [batch_size, 4, 2]. It is in the
      format of:
      [[original_height, original_width], [desired_height, desired_width],
       [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
      desired_width] is the actual scaled image size, and [y_scale, x_scale]
      is the scaling factor, which is the ratio of scaled dimension /
      original dimension.

  Returns:
    rescaled_boxes: An array of shape [batch, num_detections, 4].
  """
  image_scale = jnp.tile(image_info[:, 2:3, :], (1, 1, 2))
  detection_boxes = detection_boxes.astype(jnp.float32)
  return detection_boxes / image_scale


def flatten_leading_dims(array, num_dims_to_flatten = 2):
  """Flatten the leading N dimensions of array.

  If input array dimension is less than num_dims, flatten the whole array.

  Args:
    array: Input array.
    num_dims_to_flatten: Number of leading dimension to flatten.

  Returns:
    Leading N-dimension flattened array.

  """
  if array.ndim < num_dims_to_flatten:
    return array.reshape(-1)

  dim0 = np.prod(array.shape[:num_dims_to_flatten])
  remaining_shape = list(array.shape[num_dims_to_flatten:])
  target_shape = [dim0] + remaining_shape
  return array.reshape(target_shape)


def merge_fields(field_one,
                 field_two,
                 ):
  """Merges the fields of metrics by concatenating along first dimension.

  The two fields to be merged must have identical types, and identical value if
  the contents are strings (annotation file path) or ValueType.NONE.

  Args:
    field_one: Field of the first metric.
    field_two: Field of the second metric.

  Returns:
    Merged field of the metric.
  """
  if not isinstance(field_one, type(field_two)):
    raise ValueError('Field types unequal: '
                     f'{type(field_one)} vs {type(field_two)}')

  if isinstance(field_one, ValueType) and field_one == ValueType.NONE:
    if field_one != field_two:
      raise ValueError(f'Fields must both be NONE: {field_one} vs {field_two}')
    return field_one

  return jnp.concatenate([field_one, field_two], axis=0)


@gin.configurable
@flax.struct.dataclass
class COCODetectionMetric(metrics.Metric):
  """Computes the precision from model outputs `logits` and `labels`.

  In the following docstring, K denotes the number of detections per image.
  Attributes:
    source_id: A numpy array of int or string of shape [batch_size].
    image_info : A numpy array of float of shape [batch_size, 4, 2].
    num_detections: A numpy array of int of shape [batch_size].
    detection_boxes: A numpy array of float of shape [batch_size, K, 4].
    detection_classes: A numpy array of int of shape [batch_size, K].
    detection_scores: A numpy array of float of shape [batch_size, K].
    detection_masks (optional for segmentation): A numpy array of float of shape
              [batch_size, K, mask_height, mask_width].

    Optional fields:
    gt_source_id: A numpy array of int or string of shape [batch_size].
    gt_height: A numpy array of int of shape [batch_size].
    gt_width: A numpy array of int of shape [batch_size].
    gt_num_detections: A numpy array of int of shape [batch_size].
    gt_boxes: A numpy array of float of shape [batch_size, K, 4].
    gt_classes: A numpy array of int of shape [batch_size, K].
    gt_is_crowds: A numpy array of int of shape [batch_size, K].
              If the field is absent, assumed that this instance is not crowd.
    gt_areas: A numpy array of float of shape [batch_size, K]. If the
              field is absent, the area is calculated using either boxes or
              masks depending on which one is available.
    gt_masks: A numpy array of float of shape
              [batch_size, K, mask_height, mask_width].
  """
  source_id: Array
  image_info: Array
  num_detections: Array
  detection_boxes: Array
  detection_classes: Array
  detection_scores: Array
  detection_masks: Union[Array, ValueType] = ValueType.NONE
  gt_source_id: Union[Array, ValueType] = ValueType.NONE
  gt_height: Union[Array, ValueType] = ValueType.NONE
  gt_width: Union[Array, ValueType] = ValueType.NONE
  gt_num_detections: Union[Array, ValueType] = ValueType.NONE
  gt_boxes: Union[Array, ValueType] = ValueType.NONE
  gt_classes: Union[Array, ValueType] = ValueType.NONE
  gt_is_crowds: Union[Array, ValueType] = ValueType.NONE
  gt_areas: Union[Array, ValueType] = ValueType.NONE
  gt_masks: Union[Array, ValueType] = ValueType.NONE

  @classmethod
  @gin_utils.allow_remapping
  def from_model_output(cls, outputs, labels, **_):
    """Accumulates model outputs for evaluation.

    Args:
      outputs: A dictionary with a single entry with 'detection' as a key, and
        the value of should be a dictionary of model outputs containing the
        following required keys - 'source_id', 'image_info', 'num_detections',
        'detection_boxes', 'detection_classes', 'detection_scores'.
      labels: A dictionary with a single entry with 'detection' as a key, and
        the value of should be a dictionary of labels. It needs to have the
        following keys: 'source_id', 'height', 'width', 'num_detections',
        'boxes', 'classes', 'is_crowds', 'areas'.

    Returns:
      A metric object initialized from outputs and labels.

    Raises:
      KeyError: Missing keys in model outputs.
    """
    outputs = outputs['detection']
    labels = labels['detection']
    if 'image_info' not in outputs:
      outputs['image_info'] = labels['image_info']

    output_keys = [
        'source_id', 'image_info', 'num_detections', 'detection_boxes',
        'detection_classes', 'detection_scores', 'detection_masks'
    ]
    label_keys = [
        'source_id', 'height', 'width', 'num_detections', 'boxes', 'classes',
        'is_crowds', 'areas', 'masks'
    ]
    update_dict = {}
    for key in output_keys:
      if key == 'detection_masks' and key not in outputs:
        continue
      if key == 'source_id' and key not in outputs:
        update_dict[key] = labels['groundtruths'][key]
        continue
      if key not in outputs:
        raise KeyError(f'Model outputs must contain {key}')

      update_dict[key] = outputs[key]

    update_dict['detection_boxes'] = rescale_detection_boxes(
        update_dict['detection_boxes'], update_dict['image_info'])
    for key in label_keys:
      # Handle empty groundtruths and missing masks fields.
      if key not in labels['groundtruths']:
        continue
      update_dict['gt_' + key] = labels['groundtruths'][key]

    return cls(**update_dict)

  def merge(self, other):
    fields = dataclasses.fields(self)
    merged = {
        field.name:
        merge_fields(getattr(self, field.name), getattr(other, field.name))
        for field in fields
    }
    return type(self)(**merged)

  def reduce(self):
    reduced = {}
    for field in dataclasses.fields(self):
      attr = getattr(self, field.name)
      if isinstance(attr, Array):
        reduced[field.name] = flatten_leading_dims(attr, num_dims_to_flatten=2)
      else:
        reduced[field.name] = attr

    return type(self)(**reduced)

  def compute(self, annotation_file = None):
    # Allow the gin config to pass in annotation file here.
    if annotation_file is None:
      annotation_file = get_annotation_file()
    # Make sure detection mask is valid and gt mask will be provided.
    include_mask = ~np.all(self.detection_masks == ValueType.NONE) and (
        ~np.all(self.gt_masks == ValueType.NONE) or annotation_file is not None)

    evaluator_cls = get_evaluator()

    evaluator = evaluator_cls(
        annotation_file=annotation_file,
        include_mask=include_mask,
        need_rescale_bboxes=False,
        per_category_metrics=False)

    # Filter out the examples with invalid image source_id.
    def _filter_predictions():
      indices = np.where(self.source_id > 0)
      source_id = np.take_along_axis(self.source_id, indices[0], axis=0)
      image_info = np.take(self.image_info, indices[0], axis=0, mode='wrap')
      num_detections = np.take(
          self.num_detections, indices[0], axis=0, mode='wrap')
      detection_boxes = np.take(
          self.detection_boxes, indices[0], axis=0, mode='wrap')
      detection_classes = np.take(
          self.detection_classes, indices[0], axis=0, mode='wrap')
      detection_scores = np.take(
          self.detection_scores, indices[0], axis=0, mode='wrap')

      predictions = {
          'source_id': source_id,
          'image_info': image_info,
          'num_detections': num_detections,
          'detection_boxes': detection_boxes,
          'detection_classes': detection_classes,
          'detection_scores': detection_scores,
      }
      return predictions

    predictions = _filter_predictions()
    if include_mask:
      predictions['detection_masks'] = self.detection_masks
    predictions = {k: np.asarray(v) for k, v in predictions.items()}

    if annotation_file is None:
      ground_truth = {
          'source_id': self.gt_source_id,
          'height': self.gt_height,
          'width': self.gt_width,
          'num_detections': self.gt_num_detections,
          'boxes': self.gt_boxes,
          'classes': self.gt_classes,
          'is_crowds': self.gt_is_crowds,
          'areas': self.gt_areas,
      }
      if include_mask:
        ground_truth['masks'] = encode_xla_masks(
            self.gt_masks, self.gt_height, self.gt_width)
      ground_truth = {k: np.asarray(v) for k, v in ground_truth.items()}
    else:
      ground_truth = None

    evaluator.update(predictions, ground_truth)
    output_metrics = evaluator.evaluate()
    # Log out the metrics.
    logging.info('Final metrics:')
    logging.info('=======================')
    for k, v in output_metrics.items():
      logging.info('%s: %f', k, v)
    logging.info('=======================')
    return output_metrics
