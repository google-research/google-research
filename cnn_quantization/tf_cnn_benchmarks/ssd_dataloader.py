# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Data loader and processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as it
import math

import numpy as np
import tensorflow.compat.v1 as tf
from cnn_quantization.tf_cnn_benchmarks import mlperf
from cnn_quantization.tf_cnn_benchmarks import ssd_constants
from tensorflow_models.object_detection.box_coders import faster_rcnn_box_coder
from tensorflow_models.object_detection.core import box_list
from tensorflow_models.object_detection.core import region_similarity_calculator
from tensorflow_models.object_detection.core import target_assigner
from tensorflow_models.object_detection.matchers import argmax_matcher


class DefaultBoxes(object):
  """Default bounding boxes for 300x300 5 layer SSD.

  Default bounding boxes generation follows the order of (W, H, anchor_sizes).
  Therefore, the tensor converted from DefaultBoxes has a shape of
  [anchor_sizes, H, W, 4]. The last dimension is the box coordinates; 'ltrb'
  is [ymin, xmin, ymax, xmax] while 'xywh' is [cy, cx, h, w].
  """

  def __init__(self):
    fk = ssd_constants.IMAGE_SIZE / np.array(ssd_constants.STEPS)

    self.default_boxes = []
    # size of feature and number of feature
    for idx, feature_size in enumerate(ssd_constants.FEATURE_SIZES):
      sk1 = ssd_constants.SCALES[idx] / ssd_constants.IMAGE_SIZE
      sk2 = ssd_constants.SCALES[idx+1] / ssd_constants.IMAGE_SIZE
      sk3 = math.sqrt(sk1*sk2)
      all_sizes = [(sk1, sk1), (sk3, sk3)]

      for alpha in ssd_constants.ASPECT_RATIOS[idx]:
        w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
        all_sizes.append((w, h))
        all_sizes.append((h, w))

      assert len(all_sizes) == ssd_constants.NUM_DEFAULTS[idx]

      for w, h in all_sizes:
        for i, j in it.product(range(feature_size), repeat=2):
          cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
          box = tuple(np.clip(k, 0, 1) for k in (cy, cx, h, w))
          self.default_boxes.append(box)

    assert len(self.default_boxes) == ssd_constants.NUM_SSD_BOXES

    mlperf.logger.log(key=mlperf.tags.FEATURE_SIZES,
                      value=ssd_constants.FEATURE_SIZES)
    mlperf.logger.log(key=mlperf.tags.STEPS,
                      value=ssd_constants.STEPS)
    mlperf.logger.log(key=mlperf.tags.SCALES,
                      value=ssd_constants.SCALES)
    mlperf.logger.log(key=mlperf.tags.ASPECT_RATIOS,
                      value=ssd_constants.ASPECT_RATIOS)
    mlperf.logger.log(key=mlperf.tags.NUM_DEFAULTS,
                      value=ssd_constants.NUM_SSD_BOXES)

    def to_ltrb(cy, cx, h, w):
      return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

    # For IoU calculation
    self.default_boxes_ltrb = tuple(to_ltrb(*i) for i in self.default_boxes)

  def __call__(self, order='ltrb'):
    if order == 'ltrb': return self.default_boxes_ltrb
    if order == 'xywh': return self.default_boxes


def calc_iou_tensor(boxes1, boxes2):
  """Calculation of IoU based on two boxes tensor.

  Reference to https://github.com/kuangliu/pytorch-ssd

  Args:
    boxes1: shape (N, 4), four coordinates of N boxes
    boxes2: shape (M, 4), four coordinates of M boxes
  Returns:
    IoU: shape (N, M), IoU of the i-th box in `boxes1` and j-th box in `boxes2`
  """
  b1_left, b1_top, b1_right, b1_bottom = tf.split(boxes1, 4, axis=1)
  b2_left, b2_top, b2_right, b2_bottom = tf.split(boxes2, 4, axis=1)

  # Shape of intersect_* (N, M)
  intersect_left = tf.maximum(b1_left, tf.transpose(b2_left))
  intersect_top = tf.maximum(b1_top, tf.transpose(b2_top))
  intersect_right = tf.minimum(b1_right, tf.transpose(b2_right))
  intersect_bottom = tf.minimum(b1_bottom, tf.transpose(b2_bottom))

  boxes1_area = (b1_right - b1_left) * (b1_bottom - b1_top)
  boxes2_area = (b2_right - b2_left) * (b2_bottom - b2_top)

  intersect = tf.multiply(tf.maximum((intersect_right - intersect_left), 0),
                          tf.maximum((intersect_bottom - intersect_top), 0))
  union = boxes1_area + tf.transpose(boxes2_area) - intersect
  iou = intersect / union

  return iou


def ssd_parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  Each Example proto contains the following fields that we care about:

    image/encoded: <JPEG encoded string>
    image/source_id: tf.string
    image/height: tf.int64
    image/width: tf.int64
    image/object/bbox/xmin: tf.VarLenFeature(tf.float32)
    image/object/bbox/xmax: tf.VarLenFeature(tf.float32)
    image/object/bbox/ymin: tf.VarLenFeature(tf.float32
    image/object/bbox/ymax: tf.VarLenFeature(tf.float32)
    image/object/class/label: tf.VarLenFeature(tf.int64)
    image/object/class/text: tf.VarLenFeature(tf.string)

  Complete decoder can be found in:
  https://github.com/tensorflow/models/blob/master/research/object_detection/data_decoders/tf_example_decoder.py

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    A dictionary with the following key-values:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    groundtruth_boxes: Tensor tf.float32 of shape [num_boxes, 4], containing
      coordinates of object bounding boxes.
    groundtruth_classeS: Tensor tf.int64 of shape [num_boxes, 1], containing
      class labels of objects.
    source_id: unique image identifier.
    raw_shape: [height, width, 3].
  """
  feature_map = {
      'image/encoded': tf.FixedLenFeature(
          (), dtype=tf.string, default_value=''),
      'image/source_id': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/height': tf.FixedLenFeature((), tf.int64, default_value=1),
      'image/width': tf.FixedLenFeature((), tf.int64, default_value=1),
      'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
  }
  features = tf.parse_single_example(example_serialized, feature_map)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 1)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 1)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 1)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 1)

  image_buffer = features['image/encoded']
  # Bounding box coordinates should be in ltrb order
  boxes = tf.concat([ymin, xmin, ymax, xmax], 1)
  classes = tf.expand_dims(features['image/object/class/label'].values, 1)
  source_id = features['image/source_id']
  raw_shape = tf.stack([features['image/height'], features['image/width'], 3])

  return {'image_buffer': image_buffer,
          'groundtruth_boxes': boxes,
          'groundtruth_classes': classes,
          'source_id': source_id,
          'raw_shape': raw_shape}


def ssd_decode_and_crop(image_buffer, boxes, classes, raw_shape):
  """Crop image randomly and decode the cropped region.

  This function will crop an image to meet the following requirements:
  1. height to width ratio between 0.5 and 2;
  2. IoUs of some boxes exceed specified threshold;
  3. At least one box center is in the cropped region.
  We defer the jpeg decoding task until after the crop to avoid wasted work.

  Reference: https://github.com/chauhan-utk/ssd.DomainAdaptation

  Args:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    boxes: Tensor tf.float32 of shape [num_boxes, 4], containing coordinates of
      object bounding boxes.
    classes: Tensor tf.int64 of shape [num_boxes, 1], containing class labels
      of objects.
    raw_shape: [height, width, 3].

  Returns:
    resized_image: decoded, cropped, and resized image Tensor tf.float32 of
      shape [ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE, 3], value
      range 0--255.
    cropped_boxes: box coordinates for objects in the cropped region.
    cropped_classes: class labels for objects in the cropped region.
  """

  num_boxes = tf.shape(boxes)[0]

  def no_crop_check():
    return (tf.random_uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
            < ssd_constants.P_NO_CROP_PER_PASS)

  def no_crop_proposal():
    return (
        tf.ones((), tf.bool),
        tf.convert_to_tensor([0, 0, 1, 1], dtype=tf.float32),
        tf.ones((num_boxes,), tf.bool),
    )

  def crop_proposal():
    rand_vec = lambda minval, maxval: tf.random_uniform(
        shape=(ssd_constants.NUM_CROP_PASSES, 1), minval=minval, maxval=maxval,
        dtype=tf.float32)

    width, height = rand_vec(0.3, 1), rand_vec(0.3, 1)
    left, top = rand_vec(0, 1-width), rand_vec(0, 1-height)

    right = left + width
    bottom = top + height

    ltrb = tf.concat([left, top, right, bottom], axis=1)

    min_iou = tf.random_shuffle(ssd_constants.CROP_MIN_IOU_CHOICES)[0]
    ious = calc_iou_tensor(ltrb, boxes)

    # discard any bboxes whose center not in the cropped image
    xc, yc = [tf.tile(0.5 * (boxes[:, i + 0] + boxes[:, i + 2])[tf.newaxis, :],
                      (ssd_constants.NUM_CROP_PASSES, 1)) for i in range(2)]

    masks = tf.reduce_all(tf.stack([
        tf.greater(xc, tf.tile(left, (1, num_boxes))),
        tf.less(xc, tf.tile(right, (1, num_boxes))),
        tf.greater(yc, tf.tile(top, (1, num_boxes))),
        tf.less(yc, tf.tile(bottom, (1, num_boxes))),
    ], axis=2), axis=2)

    # Checks of whether a crop is valid.
    valid_aspect = tf.logical_and(tf.less(height/width, 2),
                                  tf.less(width/height, 2))
    valid_ious = tf.reduce_all(tf.greater(ious, min_iou), axis=1, keepdims=True)
    valid_masks = tf.reduce_any(masks, axis=1, keepdims=True)

    valid_all = tf.cast(tf.reduce_all(tf.concat(
        [valid_aspect, valid_ious, valid_masks], axis=1), axis=1), tf.int32)

    # One indexed, as zero is needed for the case of no matches.
    index = tf.range(1, 1 + ssd_constants.NUM_CROP_PASSES, dtype=tf.int32)

    # Either one-hot, or zeros if there is no valid crop.
    selection = tf.equal(tf.reduce_max(index * valid_all), index)

    use_crop = tf.reduce_any(selection)
    output_ltrb = tf.reduce_sum(tf.multiply(ltrb, tf.tile(tf.cast(
        selection, tf.float32)[:, tf.newaxis], (1, 4))), axis=0)
    output_masks = tf.reduce_any(tf.logical_and(masks, tf.tile(
        selection[:, tf.newaxis], (1, num_boxes))), axis=0)

    return use_crop, output_ltrb, output_masks

  def proposal(*args):
    return tf.cond(
        pred=no_crop_check(),
        true_fn=no_crop_proposal,
        false_fn=crop_proposal,
    )

  _, crop_bounds, box_masks = tf.while_loop(
      cond=lambda x, *_: tf.logical_not(x),
      body=proposal,
      loop_vars=[tf.zeros((), tf.bool), tf.zeros((4,), tf.float32), tf.zeros((num_boxes,), tf.bool)],
  )

  filtered_boxes = tf.boolean_mask(boxes, box_masks, axis=0)

  mlperf.logger.log(key=mlperf.tags.NUM_CROPPING_ITERATIONS,
                    value=ssd_constants.NUM_CROP_PASSES)

  # Clip boxes to the cropped region.
  filtered_boxes = tf.stack([
      tf.maximum(filtered_boxes[:, 0], crop_bounds[0]),
      tf.maximum(filtered_boxes[:, 1], crop_bounds[1]),
      tf.minimum(filtered_boxes[:, 2], crop_bounds[2]),
      tf.minimum(filtered_boxes[:, 3], crop_bounds[3]),
  ], axis=1)

  left = crop_bounds[0]
  top = crop_bounds[1]
  width = crop_bounds[2] - left
  height = crop_bounds[3] - top

  cropped_boxes = tf.stack([
      (filtered_boxes[:, 0] - left) / width,
      (filtered_boxes[:, 1] - top) / height,
      (filtered_boxes[:, 2] - left) / width,
      (filtered_boxes[:, 3] - top) / height,
  ], axis=1)

  # crop_window containing integer coordinates of cropped region. A normalized
  # coordinate value of y should be mapped to the image coordinate at
  # y * (height - 1).
  raw_shape = tf.cast(raw_shape, tf.float32)
  crop_window = tf.stack([left * (raw_shape[0] - 1),
                          top * (raw_shape[1] - 1),
                          width * raw_shape[0],
                          height * raw_shape[1]])
  crop_window = tf.cast(crop_window, tf.int32)

  # Fused op only decodes the cropped portion of an image
  cropped_image = tf.image.decode_and_crop_jpeg(
      image_buffer, crop_window, channels=3)

  # Resize converts image dtype from uint8 to float32, without rescaling values.
  resized_image = tf.image.resize_images(
      cropped_image, [ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE])
  mlperf.logger.log(key=mlperf.tags.INPUT_SIZE,
                    value=ssd_constants.IMAGE_SIZE)

  cropped_classes = tf.boolean_mask(classes, box_masks, axis=0)

  return resized_image, cropped_boxes, cropped_classes


def color_jitter(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distort the color of the image."""
  with tf.name_scope('distort_color'):
    if brightness > 0:
      image = tf.image.random_brightness(image, max_delta=brightness)
    if contrast > 0:
      image = tf.image.random_contrast(
          image, lower=1-contrast, upper=1+contrast)
    if saturation > 0:
      image = tf.image.random_saturation(
          image, lower=1-saturation, upper=1+saturation)
    if hue > 0:
      image = tf.image.random_hue(image, max_delta=hue)
    return image


def normalize_image(image):
  """Normalize the image to zero mean and unit variance.

  Args:
    image: 3D tensor of type float32, value in [0, 1]
  Returns:
    image normalized by mean and stdev.
  """
  image = tf.subtract(image, ssd_constants.NORMALIZATION_MEAN)
  image = tf.divide(image, ssd_constants.NORMALIZATION_STD)

  mlperf.logger.log(key=mlperf.tags.DATA_NORMALIZATION_MEAN,
                    value=ssd_constants.NORMALIZATION_MEAN)
  mlperf.logger.log(key=mlperf.tags.DATA_NORMALIZATION_STD,
                    value=ssd_constants.NORMALIZATION_STD)
  return image


class Encoder(object):
  """Encoder for SSD boxes and labels."""

  def __init__(self):
    similarity_calc = region_similarity_calculator.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(
        matched_threshold=ssd_constants.MATCH_THRESHOLD,
        unmatched_threshold=ssd_constants.MATCH_THRESHOLD,
        negatives_lower_than_unmatched=True,
        force_match_for_each_row=True)

    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=ssd_constants.BOX_CODER_SCALES)

    self.default_boxes = DefaultBoxes()('ltrb')
    self.default_boxes = box_list.BoxList(
        tf.convert_to_tensor(self.default_boxes))
    self.assigner = target_assigner.TargetAssigner(
        similarity_calc, matcher, box_coder)

  def encode_labels(self, gt_boxes, gt_labels):
    target_boxes = box_list.BoxList(gt_boxes)
    encoded_classes, _, encoded_boxes, _, matches = self.assigner.assign(
        self.default_boxes, target_boxes, gt_labels)
    num_matched_boxes = tf.reduce_sum(
        tf.cast(tf.not_equal(matches.match_results, -1), tf.float32))
    return encoded_classes, encoded_boxes, num_matched_boxes
