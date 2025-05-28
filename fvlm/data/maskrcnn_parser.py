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

"""Data parser for Mask R-CNN model.

This is adapted from the parser here:
https://github.com/tensorflow/tpu/blob/master/models/official/detection/dataloader/maskrcnn_parser.py
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Text

from data.dataloader import anchor
from data.dataloader import box_utils
from data.dataloader import dataloader_utils
from data.dataloader import input_utils
from data.dataloader import mode_keys as ModeKeys
from data.dataloader import tf_example_decoder
import gin
import tensorflow as tf
from utils import dataset_utils
from utils import task_utils


@gin.configurable
def clip_identity_normalization_values():
  """Get CLIP image normalization values."""
  return {'offset': (0.0, 0.0, 0.0),
          'scale': (1.0, 1.0, 1.0)}


@gin.configurable
def clip_image_normalization_values():
  """Get CLIP image normalization values."""
  return {'offset': (0.48145466, 0.4578275, 0.40821073),
          'scale': (0.26862954, 0.26130258, 0.27577711)}


def _denormalize_image(image,
                       offset=(0.485, 0.456, 0.406),
                       scale=(0.229, 0.224, 0.225)):
  """Denormalizes the image back to original scale and mean."""
  with tf.name_scope('denormalize_image'):
    scale = tf.constant(scale)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    image *= scale

    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    image += offset
    return image


@gin.configurable('maskrcnn_map_fn')
class Parser(object):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(
      self,
      output_size,
      min_level,
      max_level,
      num_scales,
      aspect_ratios,
      anchor_size,
      rpn_match_threshold = 0.7,
      rpn_unmatched_threshold = 0.3,
      rpn_batch_size_per_im = 256,
      rpn_fg_fraction = 0.5,
      aug_rand_hflip = False,
      aug_scale_min = 1.0,
      aug_scale_max = 1.0,
      skip_crowd_during_training = True,
      max_num_instances = 100,
      include_mask = False,
      mask_crop_size = 112,
      use_dummy_mask = None,
      use_bfloat16 = True,
      regenerate_source_id = False,
      mode = None,
      normalize_image_values = None,
      example_decoder = None,
      denormalize_image = False,
      # centerness targets.
      use_centerness = False,
      rpn_center_match_threshold = 0.3,
      rpn_center_unmatched_threshold = 0.1,
      rpn_center_batch_size_per_im = 256,
      rpn_center_fg_fraction = 255/256.,
      use_lrtb_box_targets = False):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      min_level: `int` number of minimum level of the output feature pyramid.
      max_level: `int` number of maximum level of the output feature pyramid.
      num_scales: `int` number representing intermediate scales added
        on each level. For instances, num_scales=2 adds one additional
        intermediate anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: `list` of float numbers representing the aspect raito
        anchors added on each level. The number indicates the ratio of width to
        height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
        on each scale level.
      anchor_size: `float` number representing the scale of size of the base
        anchor to the feature stride 2^level.
      rpn_match_threshold:
      rpn_unmatched_threshold:
      rpn_batch_size_per_im:
      rpn_fg_fraction:
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      skip_crowd_during_training: `bool`, if True, skip annotations labeled with
        `is_crowd` equals to 1.
      max_num_instances: `int` number of maximum number of instances in an
        image. The groundtruth data will be padded to `max_num_instances`.
      include_mask: a bool to indicate whether parse mask groundtruth.
      mask_crop_size: the size which groundtruth mask is cropped to.
      use_dummy_mask: Use dummy masks or data masks. Default 'None' to use
        data mask and not return 'mask_valid_loss' key. Objects365 dataset has
        no instance masks which requires us to set use_dummy_mask to True. The
        datasets with instance masks should set this to False for joint training
        with Objects365. Default 'None' is good for single dataset training or
        detection only training (no instance segmentation).
      use_bfloat16: `bool`, if True, cast output image to tf.bfloat16.
      regenerate_source_id: `bool`, if True TFExampleParser will use hashed
        value of `image/encoded` for `image/source_id`.
      mode: a ModeKeys. Specifies if this is training, evaluation, prediction
        or prediction with groundtruths in the outputs.
      normalize_image_values: A dictionary of offset and scale values to
        normalize the image with.
      example_decoder: Custom example decoder.
      denormalize_image: `bool`, if True it will denormalize the normalized
        image back to the [0, 1] scale. This to match the pretraining setting
        that normalize images to [0, 1]. Defaults to False.
      use_centerness: `bool` indicating whether to use centerness sampling.
      rpn_center_match_threshold: `float`, match IoU threshold for positive
        samples in centerness sampling. Defaults to 0.3.
      rpn_center_unmatched_threshold: `float`, (un)match IoU threshold for
        negative samples in centerenss sampling. Defaults to 0.1.
      rpn_center_batch_size_per_im: `int`, a batch size per image.
      rpn_center_fg_fraction: `float`, a ratio of positive / (positive +
        negative) samples in centerness sampling.
      use_lrtb_box_targets: `bool` whether to use lrtb format in box targets.
    """
    self._mode = mode
    self._max_num_instances = max_num_instances
    self._skip_crowd_during_training = skip_crowd_during_training
    self._is_training = (mode == ModeKeys.TRAIN)
    self._use_centerness = use_centerness
    self._use_lrtb_box_targets = use_lrtb_box_targets

    if example_decoder is None:
      self._example_decoder = tf_example_decoder.TfExampleDecoder(
          include_mask=include_mask, regenerate_source_id=regenerate_source_id)
    else:
      self._example_decoder = example_decoder

    if isinstance(output_size, int):
      output_size = (output_size, output_size)
    self._output_size = output_size
    self._min_level = min_level
    self._max_level = max_level
    self._num_scales = num_scales
    self._aspect_ratios = aspect_ratios
    self._anchor_size = anchor_size
    if self._use_centerness and len(self._aspect_ratios) != 1:
      raise ValueError('A single aspect ratio must be used with centerness.')

    # Target assigning.
    self._rpn_match_threshold = rpn_match_threshold
    self._rpn_unmatched_threshold = rpn_unmatched_threshold
    self._rpn_batch_size_per_im = rpn_batch_size_per_im
    self._rpn_fg_fraction = rpn_fg_fraction

    # Centerness target assigning.
    if use_centerness:
      self._rpn_center_match_threshold = rpn_center_match_threshold
      self._rpn_center_unmatched_threshold = rpn_center_unmatched_threshold
      self._rpn_center_batch_size_per_im = rpn_center_batch_size_per_im
      self._rpn_center_fg_fraction = rpn_center_fg_fraction

    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max

    # Mask.
    self._include_mask = include_mask
    self._mask_crop_size = mask_crop_size
    self._use_dummy_mask = use_dummy_mask

    # Device.
    self._use_bfloat16 = use_bfloat16

    # Normalize values
    self._normalize_image_values = (
        normalize_image_values if normalize_image_values else {})
    self._denormalize_image = denormalize_image

    # Data is parsed depending on the model Modekey.
    if mode == ModeKeys.TRAIN:
      self._parse_fn = self._parse_train_data
    elif mode == ModeKeys.EVAL:
      self._parse_fn = self._parse_eval_data
    elif mode == ModeKeys.PREDICT or mode == ModeKeys.PREDICT_WITH_GT:
      self._parse_fn = self._parse_predict_data
    else:
      raise ValueError('mode is not defined.')

  def __call__(self, value):
    """Parses data to an image and associated training labels.

    Args:
      value: a string tensor holding a serialized tf.Example proto.

    Returns:
      image, labels: if mode == ModeKeys.TRAIN. see _parse_train_data.
      {'images': image, 'labels': labels}: if mode == ModeKeys.PREDICT
        or ModeKeys.PREDICT_WITH_GT.
    """
    with tf.name_scope('parser'):
      data = self._example_decoder.decode(value)
      return self._parse_fn(data)

  def _parse_train_data(self, data):
    """Parses data for training.

    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
      image: image tensor that is preproessed to have normalized value and
        dimension [output_size[0], output_size[1], 3]
      labels: a dictionary of tensors used for training. The following describes
        {key: value} pairs in the dictionary.
        image_info: a 2D `Tensor` that encodes the information of the image and
          the applied preprocessing. It is in the format of
          [[original_height, original_width], [scaled_height, scaled_width],
        anchor_boxes: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, 4] representing anchor boxes at each level.
        rpn_score_targets: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, anchors_per_location]. The height_l and
          width_l represent the dimension of class logits at l-th level.
        rpn_box_targets: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, anchors_per_location * 4]. The height_l and
          width_l represent the dimension of bounding box regression output at
          l-th level.
        gt_boxes: Groundtruth bounding box annotations. The box is represented
           in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled
           image that is fed to the network. The tennsor is padded with -1 to
           the fixed dimension [self._max_num_instances, 4].
        gt_classes: Groundtruth classes annotations. The tennsor is padded
          with -1 to the fixed dimension [self._max_num_instances].
        gt_masks: groundtrugh masks cropped by the bounding box and
          resized to a fixed size determined by mask_crop_size.
    """
    classes = data['groundtruth_classes']
    boxes = data['groundtruth_boxes']
    if self._include_mask:
      if self._use_dummy_mask:
        # Only for dataset without mask labels e.g. Objects365.
        num_instances = tf.shape(classes)[:1]
        image_shape = tf.shape(data['image'])[0:2]
        masks = tf.zeros(tf.concat([num_instances, image_shape], axis=0),
                         dtype=tf.float32)
      else:
        masks = data['groundtruth_instance_masks']

    is_crowds = data['groundtruth_is_crowd']
    # Skips annotations with `is_crowd` = True.
    if self._skip_crowd_during_training and self._is_training:
      num_groundtrtuhs = tf.shape(classes)[0]
      with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
        indices = tf.cond(
            tf.greater(tf.size(is_crowds), 0),
            lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
            lambda: tf.cast(tf.range(num_groundtrtuhs), tf.int64))
      classes = tf.gather(classes, indices)
      boxes = tf.gather(boxes, indices)
      if self._include_mask:
        masks = tf.gather(masks, indices)

    # Gets original image and its size.
    image = data['image']
    image_shape = tf.shape(image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = input_utils.normalize_image(image, **self._normalize_image_values)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      if self._include_mask:
        image, boxes, masks = input_utils.random_horizontal_flip(
            image, boxes, masks)
      else:
        image, boxes = input_utils.random_horizontal_flip(
            image, boxes)

    # Converts boxes from normalized coordinates to pixel coordinates.
    # Now the coordinates of boxes are w.r.t. the original image.
    boxes = box_utils.denormalize_boxes(boxes, image_shape)

    # Resizes and crops image.
    image, image_info = input_utils.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=input_utils.compute_padded_size(
            self._output_size, 2 ** self._max_level),
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)
    image_height, image_width, _ = image.get_shape().as_list()

    if self._denormalize_image:
      image = _denormalize_image(image, **self._normalize_image_values)

    # Resizes and crops boxes.
    # Now the coordinates of boxes are w.r.t the scaled image.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = input_utils.resize_and_crop_boxes(
        boxes, image_scale, image_info[1, :], offset)

    # Filters out ground truth boxes that are all zeros.
    indices = box_utils.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    if self._include_mask:
      masks = tf.gather(masks, indices)
      # Transfer boxes to the original image space and do normalization.
      cropped_boxes = boxes + tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
      cropped_boxes /= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
      cropped_boxes = box_utils.normalize_boxes(cropped_boxes, image_shape)
      num_masks = tf.shape(masks)[0]
      masks = tf.image.crop_and_resize(
          tf.expand_dims(masks, axis=-1),
          cropped_boxes,
          box_indices=tf.range(num_masks, dtype=tf.int32),
          crop_size=[self._mask_crop_size, self._mask_crop_size],
          method='bilinear')
      masks = tf.squeeze(masks, axis=-1)

    # Assigns anchor targets.
    # Note that after the target assignment, box targets are absolute pixel
    # offsets w.r.t. the scaled image.
    input_anchor = anchor.Anchor(
        self._min_level,
        self._max_level,
        self._num_scales,
        self._aspect_ratios,
        self._anchor_size,
        (image_height, image_width))
    if self._use_lrtb_box_targets:
      rpn_anchor_labeler = anchor.RpnAnchorCenternessLabeler
    else:
      rpn_anchor_labeler = anchor.RpnAnchorLabeler
    anchor_labeler = rpn_anchor_labeler(
        input_anchor,
        self._rpn_match_threshold,
        self._rpn_unmatched_threshold,
        self._rpn_batch_size_per_im,
        self._rpn_fg_fraction)
    rpn_score_targets, rpn_box_targets = anchor_labeler.label_anchors(
        boxes, tf.cast(tf.expand_dims(classes, axis=-1), dtype=tf.float32))

    # Assign centerness targets.
    if self._use_centerness:
      anchor_center_labeler = anchor.RpnAnchorCenternessLabeler(
          input_anchor,
          self._rpn_center_match_threshold,
          self._rpn_center_unmatched_threshold,
          self._rpn_center_batch_size_per_im,
          self._rpn_center_fg_fraction)
      rpn_score_targets, _ = anchor_center_labeler.label_anchors(
          boxes, tf.cast(tf.expand_dims(classes, axis=-1), dtype=tf.float32))

    # If bfloat16 is used, casts input image to tf.bfloat16.
    if self._use_bfloat16:
      image = tf.cast(image, dtype=tf.bfloat16)

    # Packs labels for model_fn outputs.
    labels = {
        'anchor_boxes': input_anchor.multilevel_boxes,
        'image_info': image_info,
        'rpn_score_targets': rpn_score_targets,
        'rpn_box_targets': rpn_box_targets,
    }
    labels['gt_boxes'] = input_utils.clip_or_pad_to_fixed_size(
        boxes, self._max_num_instances, -1)
    labels['gt_classes'] = input_utils.clip_or_pad_to_fixed_size(
        classes, self._max_num_instances, -1)
    if self._include_mask:
      labels['gt_masks'] = input_utils.clip_or_pad_to_fixed_size(
          masks, self._max_num_instances, -1)

    if self._use_dummy_mask is not None:
      labels['mask_loss_valid'] = tf.constant(
          [1.0 - float(self._use_dummy_mask)], dtype=tf.float32)

    return image, labels

  def _parse_eval_data(self, data):
    """Parses data for evaluation.

    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
      image: image tensor that is preproessed to have normalized value and
        dimension [output_size[0], output_size[1], 3]
      labels: a dictionary of tensors used for training. The following describes
        {key: value} pairs in the dictionary.
        image_info: a 2D `Tensor` that encodes the information of the image and
          the applied preprocessing. It is in the format of
          [[original_height, original_width], [scaled_height, scaled_width],
        anchor_boxes: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, 4] representing anchor boxes at each level.
        groundtruths:
          source_id: Groundtruth source id.
          height: Original image height.
          width: Original image width.
          boxes: Groundtruth bounding box annotations. The box is represented
             in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled
             image that is fed to the network. The tennsor is padded with -1 to
             the fixed dimension [self._max_num_instances, 4].
          classes: Groundtruth classes annotations. The tennsor is padded
            with -1 to the fixed dimension [self._max_num_instances].
          areas: Box area or mask area depend on whether mask is present.
          is_crowds: Whether the ground truth label is a crowd label.
          num_groundtruths: Number of ground truths in the image.
    """
    # Gets original image and its size.
    image = data['image']
    image_shape = tf.shape(image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = input_utils.normalize_image(image)

    # Resizes and crops image.
    image, image_info = input_utils.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=input_utils.compute_padded_size(
            self._output_size, 2 ** self._max_level),
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image_height, image_width, _ = image.get_shape().as_list()
    if self._denormalize_image:
      image = _denormalize_image(image, **self._normalize_image_values)

    # Assigns anchor targets.
    input_anchor = anchor.Anchor(
        self._min_level,
        self._max_level,
        self._num_scales,
        self._aspect_ratios,
        self._anchor_size,
        (image_height, image_width))

    # If bfloat16 is used, casts input image to tf.bfloat16.
    if self._use_bfloat16:
      image = tf.cast(image, dtype=tf.bfloat16)

    # Sets up groundtruth data for evaluation.
    groundtruths = {
        'source_id':
            data['source_id'],
        'height':
            data['height'],
        'width':
            data['width'],
        'num_groundtruths':
            tf.shape(data['groundtruth_classes']),
        'boxes':
            box_utils.denormalize_boxes(data['groundtruth_boxes'], image_shape),
        'classes':
            data['groundtruth_classes'],
        'areas':
            data['groundtruth_area'],
        'is_crowds':
            tf.cast(data['groundtruth_is_crowd'], tf.int32),
    }
    groundtruths['source_id'] = dataloader_utils.process_source_id(
        groundtruths['source_id'])
    groundtruths = dataloader_utils.pad_groundtruths_to_fixed_size(
        groundtruths, self._max_num_instances)

    # Packs labels for model_fn outputs.
    labels = {
        'anchor_boxes': input_anchor.multilevel_boxes,
        'image_info': image_info,
        'groundtruths': groundtruths,
    }

    return image, labels

  def _parse_predict_data(self, data):
    """Parses data for prediction.

    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
      A dictionary of {'images': image, 'labels': labels} where
        images: image tensor that is preproessed to have normalized value and
          dimension [output_size[0], output_size[1], 3]
        labels: a dictionary of tensors used for training. The following
          describes {key: value} pairs in the dictionary.
          source_ids: Source image id. Default value -1 if the source id is
            empty in the groundtruth annotation.
          image_info: a 2D `Tensor` that encodes the information of the image
            and the applied preprocessing. It is in the format of
            [[original_height, original_width], [scaled_height, scaled_width],
          anchor_boxes: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, 4] representing anchor boxes at each
            level.
    """
    # Gets original image and its size.
    image = data['image']
    image_shape = tf.shape(image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = input_utils.normalize_image(image, **self._normalize_image_values)

    # Resizes and crops image.
    image, image_info = input_utils.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=input_utils.compute_padded_size(
            self._output_size, 2 ** self._max_level),
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image_height, image_width, _ = image.get_shape().as_list()
    if self._denormalize_image:
      image = _denormalize_image(image, **self._normalize_image_values)

    # If bfloat16 is used, casts input image to tf.bfloat16.
    if self._use_bfloat16:
      image = tf.cast(image, dtype=tf.bfloat16)

    # Compute Anchor boxes.
    input_anchor = anchor.Anchor(
        self._min_level,
        self._max_level,
        self._num_scales,
        self._aspect_ratios,
        self._anchor_size,
        (image_height, image_width))

    labels = {
        'source_id': dataloader_utils.process_source_id(data['source_id']),
        'anchor_boxes': input_anchor.multilevel_boxes,
        'image_info': image_info,
    }

    if self._mode == ModeKeys.PREDICT_WITH_GT:
      # Converts boxes from normalized coordinates to pixel coordinates.
      boxes = box_utils.denormalize_boxes(
          data['groundtruth_boxes'], image_shape)
      groundtruths = {
          'source_id': data['source_id'],
          'height': data['height'],
          'width': data['width'],
          'num_detections': tf.shape(data['groundtruth_classes']),
          'boxes': boxes,
          'classes': data['groundtruth_classes'],
          'areas': data['groundtruth_area'],
          'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32),
      }
      groundtruths['source_id'] = dataloader_utils.process_source_id(
          groundtruths['source_id'])
      groundtruths = dataloader_utils.pad_groundtruths_to_fixed_size(
          groundtruths, self._max_num_instances)
      labels['groundtruths'] = groundtruths
      # Now the coordinates of boxes are w.r.t the scaled image.
      image_scale = image_info[2, :]
      offset = image_info[3, :]
      scaled_boxes = input_utils.resize_and_crop_boxes(
          boxes, image_scale, image_info[1, :], offset)
      labels['gt_boxes'] = input_utils.clip_or_pad_to_fixed_size(
          scaled_boxes, self._max_num_instances, 0.0)

    return {
        'images': image,
        'labels': labels,
    }


@gin.configurable
class TfdsMaskRCNNParser(Parser):
  """A subclass to parse without tf.ExampleDecoder."""

  def __call__(self, value):
    with tf.name_scope('parser'):
      return self._parse_fn(value)


@gin.configurable
class VALMaskRCNNParser(Parser):
  """A subclass to supply source id along with each parsed example."""

  def __call__(self, value):
    with tf.name_scope('parser'):
      data = self._example_decoder.decode(value)
      if 'source_id' not in data:
        raise KeyError('Source id must be in data because this parser is'
                       ' designed for source id filtering. Use another '
                       'parser when this filtering is not needed.')
      source_id = tf.strings.to_number(data['source_id'], tf.int64)
      return self._parse_fn(data), source_id


@gin.configurable(denylist=['value'])
def maskrcnn_parser_fn(value, parser_fn,
                       is_training):
  """Wrapper around mask rcnn parser to standardize its output to a dictionary.

  Args:
    value: a string tensor holding a serialized tf.Example proto.
    parser_fn: a function to parse data for training and testing.
    is_training: a bool to indicate whether it's in training or testing.

  Returns:
    A dictionary {'images': image, 'labels': labels} whether it's in training
    or prediction mode.
  """
  data = parser_fn(value)
  if is_training:
    images, labels = data
    data = {'images': images, 'labels': labels}

  return data


@gin.configurable(denylist=['value'])
def multitask_detection_parser_fn(value, parser_fn,
                                  is_training):
  """Wrapper around mask rcnn parser to standardize its output to a dictionary.

  Args:
    value: a string tensor holding a serialized tf.Example proto.
    parser_fn: a function to parse data for training and testing.
    is_training: a bool to indicate whether it's in training or testing.

  Returns:
    A dictionary {'images': image, 'labels': labels} whether it's in training
    or prediction mode.
  """
  data = parser_fn(value)
  if is_training:
    images, labels = data
    data = {'images': images, 'labels': labels}

  # Dummy texts unused by the model.
  data['texts'] = tf.zeros((1, 1), dtype=data['images'].dtype)
  data['labels'] = task_utils.DetectionTask.unfilter_by_task(data['labels'])
  return data


@gin.register
def get_multitask_text_detection_parser_fn(
    parser_fn,
    is_training,
    ):
  """Wrapper around mask rcnn parser to standardize its output to a dictionary.

  Args:
    parser_fn: a function to parse data for training and testing.
    is_training: a bool to indicate whether it's in training or testing.

  Returns:
    A dictionary {'images': image, 'labels': labels} whether it's in training
    or prediction mode.
  """
  class_embeddings = dataset_utils.load_dataset_vocab_embed()

  def detection_parser_fn(value):
    data = parser_fn(value)
    if is_training:
      images, labels = data
      data = {'images': images, 'labels': labels}

    data['texts'] = class_embeddings
    data['labels'] = task_utils.DetectionTask.unfilter_by_task(data['labels'])
    return data

  return detection_parser_fn
