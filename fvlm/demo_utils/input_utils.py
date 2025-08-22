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

# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Input functions for F-VLM demo.

We use CLIP text encoder to compute the text embeddings.
CLIP paper: https://arxiv.org/abs/2103.00020

Adapted from Cloud TPU detection codebase:
https://github.com/tensorflow/tpu/tree/master/models/official/detection/dataloader
https://github.com/tensorflow/tpu/tree/master/models/official/detection/utils
https://github.com/tensorflow/tpu/tree/master/models/official/detection/utils/object_detection
"""

import collections
import math

import clip
import tensorflow as tf
import torch


category_dict = {
    'citrus.jpg': [
        'kiwi',
        'orange',
        'lemon',
        'blackberry',
        'pine cone',
        'red orange',
        'table',
        'spoon',
        'pine needles',
        'seed',
    ],
}


def get_category_index(id_mapping):
  """Get a dictionary of category index."""
  category_index = {k: {'id': k, 'name': id_mapping[k]} for k in id_mapping}
  return category_index


def get_clip_text_features(model_name, cls_prompts):
  """Load CLIP model and get text features.
  """
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model, _ = clip.load(model_name, device=device)
  cls_tokens = clip.tokenize(cls_prompts)
  text_features = model.encode_text(cls_tokens)
  text_features /= text_features.norm(dim=-1, keepdim=True)

  return text_features


def rovit_image_normalization_values():
  """Get RO-ViT image normalization values."""
  return {'offset': (0.0, 0.0, 0.0), 'scale': (1.0, 1.0, 1.0)}


def clip_image_normalization_values():
  """Get CLIP image normalization values."""
  return {
      'offset': (0.48145466, 0.4578275, 0.40821073),
      'scale': (0.26862954, 0.26130258, 0.27577711),
  }


def normalize_image(image,
                    offset=(0.485, 0.456, 0.406),
                    scale=(0.229, 0.224, 0.225)):
  """Normalizes the image to zero mean and unit variance."""
  with tf.name_scope('normalize_image'):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    image -= offset

    scale = tf.constant(scale)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    image /= scale
    return image


def resize_and_crop_image(image,
                          desired_size,
                          padded_size,
                          aug_scale_min=1.0,
                          aug_scale_max=1.0,
                          seed=1,
                          method=tf.image.ResizeMethod.BILINEAR):
  """Resizes the input image to output size.

  Resize and pad images given the desired output size of the image and
  stride size.

  Here are the preprocessing steps.
  1. For a given image, keep its aspect ratio and rescale the image to make it
     the largest rectangle to be bounded by the rectangle specified by the
     `desired_size`.
  2. Pad the rescaled image to the padded_size.

  Args:
    image: a `Tensor` of shape [height, width, 3] representing an image.
    desired_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the desired actual output image size.
    padded_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the padded output image size. Padding will be applied
      after scaling the image to the desired_size.
    aug_scale_min: a `float` with range between [0, 1.0] representing minimum
      random scale applied to desired_size for training scale jittering.
    aug_scale_max: a `float` with range between [1.0, inf] representing maximum
      random scale applied to desired_size for training scale jittering.
    seed: seed for random scale jittering.
    method: function to resize input image to scaled image.

  Returns:
    output_image: `Tensor` of shape [height, width, 3] where [height, width]
      equals to `output_size`.
    image_info: a 2D `Tensor` that encodes the information of the image and the
      applied preprocessing. It is in the format of
      [[original_height, original_width], [desired_height, desired_width],
       [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
      desired_width] is the actual scaled image size, and [y_scale, x_scale] is
      the scaling factor, which is the ratio of
      scaled dimension / original dimension.
  """
  with tf.name_scope('resize_and_crop_image'):
    image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

    random_jittering = (aug_scale_min != 1.0 or aug_scale_max != 1.0)

    if random_jittering:
      random_scale = tf.random_uniform(
          [], aug_scale_min, aug_scale_max, seed=seed)
      scaled_size = tf.round(random_scale * desired_size)
    else:
      scaled_size = desired_size

    scale = tf.minimum(
        scaled_size[0] / image_size[0], scaled_size[1] / image_size[1])
    scaled_size = tf.round(image_size * scale)

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size

    # Selects non-zero random offset (x, y) if scaled image is larger than
    # desired_size.
    if random_jittering:
      max_offset = scaled_size - desired_size
      max_offset = tf.where(tf.less(max_offset, 0),
                            tf.zeros_like(max_offset),
                            max_offset)
      offset = max_offset * tf.random_uniform([2,], 0, 1, seed=seed)
      offset = tf.cast(offset, tf.int32)
    else:
      offset = tf.zeros((2,), tf.int32)

    scaled_image = tf.image.resize(
        image, tf.cast(scaled_size, tf.int32), method=method)

    if random_jittering:
      scaled_image = scaled_image[
          offset[0]:offset[0] + desired_size[0],
          offset[1]:offset[1] + desired_size[1], :]

    output_image = tf.image.pad_to_bounding_box(
        scaled_image, 0, 0, padded_size[0], padded_size[1])

    image_info = tf.stack([
        image_size,
        tf.constant(desired_size, dtype=tf.float32),
        image_scale,
        tf.cast(offset, tf.float32)])
    return output_image, image_info


def compute_padded_size(desired_size, stride):
  """Compute the padded size given the desired size and the stride.

  The padded size will be the smallest rectangle, such that each dimension is
  the smallest multiple of the stride which is larger than the desired
  dimension. For example, if desired_size = (100, 200) and stride = 32,
  the output padded_size = (128, 224).

  Args:
    desired_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the target output image size.
    stride: an integer, the stride of the backbone network.

  Returns:
    padded_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the padded output image size.
  """
  if isinstance(desired_size, list) or isinstance(desired_size, tuple):
    padded_size = [int(math.ceil(d * 1.0 / stride) * stride)
                   for d in desired_size]
  else:
    padded_size = tf.cast(
        tf.math.ceil(
            tf.cast(desired_size, dtype=tf.float32) / stride) * stride,
        tf.int32)
  return padded_size


class Anchor(object):
  """Anchor class for anchor-based object detectors."""

  def __init__(self,
               min_level,
               max_level,
               num_scales,
               aspect_ratios,
               anchor_size,
               image_size):
    """Constructs multiscale anchors.

    Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds one additional
        intermediate anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of float numbers representing the aspect ratio anchors
        added on each level. The number indicates the ratio of width to height.
        For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors on each
        scale level.
      anchor_size: float number representing the scale of size of the base
        anchor to the feature stride 2^level.
      image_size: a list of integer numbers or Tensors representing
        [height, width] of the input image size.The image_size should be divided
        by the largest feature stride 2^max_level.
    """
    self.min_level = min_level
    self.max_level = max_level
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    self.anchor_size = anchor_size
    self.image_size = image_size
    self.boxes = self._generate_boxes()

  def _generate_boxes(self):
    """Generates multiscale anchor boxes.

    Returns:
      a Tensor of shape [N, 4], representing anchor boxes of all levels
      concatenated together.
    """
    boxes_all = []
    for level in range(self.min_level, self.max_level + 1):
      boxes_l = []
      for scale in range(self.num_scales):
        for aspect_ratio in self.aspect_ratios:
          stride = 2 ** level
          intermidate_scale = 2 ** (scale / float(self.num_scales))
          base_anchor_size = self.anchor_size * stride * intermidate_scale
          aspect_x = aspect_ratio ** 0.5
          aspect_y = aspect_ratio ** -0.5
          half_anchor_size_x = base_anchor_size * aspect_x / 2.0
          half_anchor_size_y = base_anchor_size * aspect_y / 2.0
          x = tf.range(stride / 2, self.image_size[1], stride)
          y = tf.range(stride / 2, self.image_size[0], stride)
          xv, yv = tf.meshgrid(x, y)
          xv = tf.cast(tf.reshape(xv, [-1]), dtype=tf.float32)
          yv = tf.cast(tf.reshape(yv, [-1]), dtype=tf.float32)
          # Tensor shape Nx4.
          boxes = tf.stack([yv - half_anchor_size_y, xv - half_anchor_size_x,
                            yv + half_anchor_size_y, xv + half_anchor_size_x],
                           axis=1)
          boxes_l.append(boxes)
      # Concat anchors on the same level to tensor shape NxAx4.
      boxes_l = tf.stack(boxes_l, axis=1)
      boxes_l = tf.reshape(boxes_l, [-1, 4])
      boxes_all.append(boxes_l)
    return tf.concat(boxes_all, axis=0)

  def unpack_labels(self, labels, is_box=False):
    """Unpacks an array of labels into multiscales labels.

    Args:
      labels: labels to unpack.
      is_box: to unpack anchor boxes or not. If it is true, will unpack to 2D,
        otherwise, will unpack to 3D.

    Returns:
      unpacked_labels: a dictionary contains unpack labels in different levels.
    """
    unpacked_labels = collections.OrderedDict()
    count = 0
    for level in range(self.min_level, self.max_level + 1):
      feat_size_y = tf.cast(self.image_size[0] / 2 ** level, tf.int32)
      feat_size_x = tf.cast(self.image_size[1] / 2 ** level, tf.int32)
      steps = feat_size_y * feat_size_x * self.anchors_per_location
      if is_box:
        unpacked_labels[level] = tf.reshape(labels[count:count + steps],
                                            [-1, 4])
      else:
        unpacked_labels[level] = tf.reshape(labels[count:count + steps],
                                            [feat_size_y, feat_size_x, -1])
      count += steps
    return unpacked_labels

  @property
  def anchors_per_location(self):
    return self.num_scales * len(self.aspect_ratios)

  @property
  def multilevel_boxes(self):
    return self.unpack_labels(self.boxes, is_box=True)


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
      max_num_instances,
      normalize_image_values=None):
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
      max_num_instances: `int` number of maximum number of instances in an
        image. The groundtruth data will be padded to `max_num_instances`.
      normalize_image_values: A dictionary of offset and scale values to
        normalize the image with.
    """
    self._max_num_instances = max_num_instances

    if isinstance(output_size, int):
      output_size = (output_size, output_size)

    self._output_size = output_size
    self._min_level = min_level
    self._max_level = max_level
    self._num_scales = num_scales
    self._aspect_ratios = aspect_ratios
    self._anchor_size = anchor_size

    # Normalize values
    self._normalize_image_values = (
        normalize_image_values if normalize_image_values else {})

  def parse_predict_data(self, data):
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

    # Normalizes image with mean and std pixel values.
    image = normalize_image(image, **self._normalize_image_values)

    # Resizes and crops image.
    image, image_info = resize_and_crop_image(
        image,
        self._output_size,
        padded_size=compute_padded_size(
            self._output_size, 2 ** self._max_level),
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image = tf.image.convert_image_dtype(image, dtype=tf.bfloat16)
    image_height, image_width, _ = image.get_shape().as_list()

    # Compute Anchor boxes.
    input_anchor = Anchor(
        self._min_level,
        self._max_level,
        self._num_scales,
        self._aspect_ratios,
        self._anchor_size,
        (image_height, image_width))

    labels = {
        'anchor_boxes': input_anchor.multilevel_boxes,
        'image_info': image_info,
    }
    return {
        'images': image,
        'labels': labels,
    }


def get_maskrcnn_parser():
  """Get F-VLM input parser."""
  return Parser(
      output_size=1024,
      min_level=2,
      max_level=6,
      num_scales=1,
      aspect_ratios=[1.0, 2.0, 0.5],
      anchor_size=8,
      max_num_instances=100,
      normalize_image_values=clip_image_normalization_values(),
  ).parse_predict_data


def get_rovit_parser():
  """Get RO-VIT input parser."""
  return Parser(
      output_size=1024,
      min_level=2,
      max_level=5,
      num_scales=1,
      aspect_ratios=[1.0,],
      anchor_size=8,
      max_num_instances=100,
      normalize_image_values={
          'offset': (0.0, 0.0, 0.0), 'scale': (1.0, 1.0, 1.0)},
  ).parse_predict_data
