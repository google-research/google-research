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

"""Video preprocessing ops."""

import abc
import dataclasses
import functools
import math
from typing import Optional, Sequence, Tuple, Union

from absl import logging
from clu import preprocess_spec
import tensorflow as tf

from invariant_slot_attention.lib import transforms

Features = preprocess_spec.Features
all_ops = lambda: preprocess_spec.get_all_ops(__name__)
SEED_KEY = preprocess_spec.SEED_KEY
NOTRACK_BOX = (0., 0., 0., 0.)  # No-track bounding box for padding.
NOTRACK_LABEL = -1

IMAGE = "image"
VIDEO = "video"
SEGMENTATIONS = "segmentations"
RAGGED_SEGMENTATIONS = "ragged_segmentations"
SPARSE_SEGMENTATIONS = "sparse_segmentations"
SHAPE = "shape"
PADDING_MASK = "padding_mask"
RAGGED_BOXES = "ragged_boxes"
BOXES = "boxes"
FRAMES = "frames"
FLOW = "flow"
DEPTH = "depth"
ORIGINAL_SIZE = "original_size"
INSTANCE_LABELS = "instance_labels"
INSTANCE_MULTI_LABELS = "instance_multi_labels"
BOXES_VIDEO = "boxes_video"
IMAGE_PADDING_MASK = "image_padding_mask"
VIDEO_PADDING_MASK = "video_padding_mask"


def convert_uint16_to_float(array, min_val, max_val):
  return tf.cast(array, tf.float32) / 65535. * (max_val - min_val) + min_val


def get_resize_small_shape(original_size,
                           small_size):
  h, w = original_size
  ratio = (
      tf.cast(small_size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32))
  h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
  w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
  return h, w


def adjust_small_size(original_size,
                      small_size, max_size):
  """Computes the adjusted small size to ensure large side < max_size."""
  h, w = original_size
  min_original_size = tf.cast(tf.minimum(w, h), tf.float32)
  max_original_size = tf.cast(tf.maximum(w, h), tf.float32)
  if max_original_size / min_original_size * small_size > max_size:
    small_size = tf.cast(tf.floor(
        max_size * min_original_size / max_original_size), tf.int32)
  return small_size


def crop_or_pad_boxes(boxes, top, left, height,
                      width, h_orig, w_orig):
  """Transforms the relative box coordinates according to the frame crop.

  Note that, if height/width are larger than h_orig/w_orig, this function
  implements the equivalent of padding.

  Args:
    boxes: Tensor of bounding boxes with shape (..., 4).
    top: Top of crop box in absolute pixel coordinates.
    left: Left of crop box in absolute pixel coordinates.
    height: Height of crop box in absolute pixel coordinates.
    width: Width of crop box in absolute pixel coordinates.
    h_orig: Original image height in absolute pixel coordinates.
    w_orig: Original image width in absolute pixel coordinates.
  Returns:
    Boxes tensor with same shape as input boxes but updated values.
  """
  # Video track bound boxes: [num_instances, num_tracks, 4]
  # Image bounding boxes: [num_instances, 4]
  assert boxes.shape[-1] == 4
  seq_len = tf.shape(boxes)[0]
  has_tracks = len(boxes.shape) == 3
  if has_tracks:
    num_tracks = boxes.shape[1]
  else:
    assert len(boxes.shape) == 2
    num_tracks = 1

  # Transform the box coordinates.
  a = tf.cast(tf.stack([h_orig, w_orig]), tf.float32)
  b = tf.cast(tf.stack([top, left]), tf.float32)
  c = tf.cast(tf.stack([height, width]), tf.float32)
  boxes = tf.reshape(
      (tf.reshape(boxes, (seq_len, num_tracks, 2, 2)) * a - b) / c,
      (seq_len, num_tracks, len(NOTRACK_BOX)))

  # Filter the valid boxes.
  boxes = tf.minimum(tf.maximum(boxes, 0.0), 1.0)
  if has_tracks:
    cond = tf.reduce_all((boxes[:, :, 2:] - boxes[:, :, :2]) > 0.0, axis=-1)
    boxes = tf.where(cond[:, :, tf.newaxis], boxes, NOTRACK_BOX)
  else:
    boxes = tf.reshape(boxes, (seq_len, 4))

  return boxes


def flow_tensor_to_rgb_tensor(motion_image, flow_scaling_factor=50.):
  """Visualizes flow motion image as an RGB image.

  Similar as the flow_to_rgb function, but with tensors.

  Args:
    motion_image: A tensor either of shape [batch_sz, height, width, 2] or of
      shape [height, width, 2]. motion_image[..., 0] is flow in x and
      motion_image[..., 1] is flow in y.
    flow_scaling_factor: How much to scale flow for visualization.

  Returns:
    A visualization tensor with same shape as motion_image, except with three
    channels. The dtype of the output is tf.uint8.
  """

  hypot = lambda a, b: (a ** 2.0 + b ** 2.0) ** 0.5  # sqrt(a^2 + b^2)

  height, width = motion_image.get_shape().as_list()[-3:-1]  # pytype: disable=attribute-error  # allow-recursive-types
  scaling = flow_scaling_factor / hypot(height, width)
  x, y = motion_image[Ellipsis, 0], motion_image[Ellipsis, 1]
  motion_angle = tf.atan2(y, x)
  motion_angle = (motion_angle / math.pi + 1.0) / 2.0
  motion_magnitude = hypot(y, x)
  motion_magnitude = tf.clip_by_value(motion_magnitude * scaling, 0.0, 1.0)
  value_channel = tf.ones_like(motion_angle)
  flow_hsv = tf.stack([motion_angle, motion_magnitude, value_channel], axis=-1)
  flow_rgb = tf.image.convert_image_dtype(
      tf.image.hsv_to_rgb(flow_hsv), tf.uint8)
  return flow_rgb


def get_paddings(image_shape,
                 size,
                 pre_spatial_dim = None,
                 allow_crop = True):
  """Returns paddings tensors for tf.pad operation.

  Args:
    image_shape: The shape of the Tensor to be padded. The shape can be
      [..., N, H, W, C] or [..., H, W, C]. The paddings are computed for H, W
      and optionally N dimensions.
    size: The total size for the H and W dimensions to pad to.
    pre_spatial_dim: Optional, additional padding dimension before the spatial
      dimensions. It is only used if given and if len(shape) > 3.
    allow_crop: If size is bigger than requested max size, padding will be
      negative. If allow_crop is true, negative padding values will be set to 0.

  Returns:
    Paddings the given tensor shape.
  """
  assert image_shape.shape.rank == 1
  if isinstance(size, int):
    size = (size, size)
  h, w = image_shape[-3], image_shape[-2]
  # Spatial padding.
  paddings = [
      tf.stack([0, size[0] - h]),
      tf.stack([0, size[1] - w]),
      tf.stack([0, 0])
  ]
  ndims = len(image_shape)  # pytype: disable=wrong-arg-types
  # Prepend padding for temporal dimension or number of instances.
  if pre_spatial_dim is not None and ndims > 3:
    paddings = [[0, pre_spatial_dim - image_shape[-4]]] + paddings
  # Prepend with non-padded dimensions if available.
  if ndims > len(paddings):
    paddings = [[0, 0]] * (ndims - len(paddings)) + paddings
  if allow_crop:
    paddings = tf.maximum(paddings, 0)
  return tf.stack(paddings)


@dataclasses.dataclass
class VideoFromTfds:
  """Standardize features coming from TFDS video datasets."""

  video_key: str = VIDEO
  segmentations_key: str = SEGMENTATIONS
  ragged_segmentations_key: str = RAGGED_SEGMENTATIONS
  shape_key: str = SHAPE
  padding_mask_key: str = PADDING_MASK
  ragged_boxes_key: str = RAGGED_BOXES
  boxes_key: str = BOXES
  frames_key: str = FRAMES
  instance_multi_labels_key: str = INSTANCE_MULTI_LABELS
  flow_key: str = FLOW
  depth_key: str = DEPTH

  def __call__(self, features):

    features_new = {}

    if "rng" in features:
      features_new[SEED_KEY] = features.pop("rng")

    if "instances" in features:
      features_new[self.ragged_boxes_key] = features["instances"]["bboxes"]
      features_new[self.frames_key] = features["instances"]["bbox_frames"]
      if "segmentations" in features["instances"]:
        features_new[self.ragged_segmentations_key] = tf.cast(
            features["instances"]["segmentations"][Ellipsis, 0], tf.int32)

      # Special handling of CLEVR (https://arxiv.org/abs/1612.06890) objects.
      if ("color" in features["instances"] and
          "shape" in features["instances"] and
          "material" in features["instances"]):
        color = tf.cast(features["instances"]["color"], tf.int32)
        shape = tf.cast(features["instances"]["shape"], tf.int32)
        material = tf.cast(features["instances"]["material"], tf.int32)
        features_new[self.instance_multi_labels_key] = tf.stack(
            (color, shape, material), axis=-1)

    if "segmentations" in features:
      features_new[self.segmentations_key] = tf.cast(
          features["segmentations"][Ellipsis, 0], tf.int32)

    if "depth" in features:
      # Undo float to uint16 scaling
      if "metadata" in features and "depth_range" in features["metadata"]:
        depth_range = features["metadata"]["depth_range"]
        features_new[self.depth_key] = convert_uint16_to_float(
            features["depth"], depth_range[0], depth_range[1])

    if "flows" in features:
      # Some datasets use "flows" instead of "flow" for optical flow.
      features["flow"] = features["flows"]
    if "backward_flow" in features:
      # By default, use "backward_flow" if available.
      features["flow"] = features["backward_flow"]
      features["metadata"]["flow_range"] = features["metadata"][
          "backward_flow_range"]
    if "flow" in features:
      # Undo float to uint16 scaling
      flow_range = features["metadata"].get("flow_range", (-255, 255))
      features_new[self.flow_key] = convert_uint16_to_float(
          features["flow"], flow_range[0], flow_range[1])

    # Convert video to float and normalize.
    video = features["video"]
    assert video.dtype == tf.uint8  # pytype: disable=attribute-error  # allow-recursive-types
    video = tf.image.convert_image_dtype(video, tf.float32)
    features_new[self.video_key] = video

    # Store original video shape (e.g. for correct evaluation metrics).
    features_new[self.shape_key] = tf.shape(video)

    # Store padding mask
    features_new[self.padding_mask_key] = tf.cast(
        tf.ones_like(video)[Ellipsis, 0], tf.uint8)

    return features_new


@dataclasses.dataclass
class AddTemporalAxis:
  """Lift images to videos by adding a temporal axis at the beginning.

  We need to distinguish two cases because `image_ops.py` uses
  ORIGINAL_SIZE = [H,W] and `video_ops.py` uses SHAPE = [T,H,W,C]:
  a) The features are fed from image ops: ORIGINAL_SIZE is converted
    to SHAPE ([H,W] -> [1,H,W,C]) and removed from the features.
    Typical use case: Evaluation of GV image tasks in a video setting. This op
    is added after the image preprocessing in order not to change the standard
    image preprocessing.
  b) The features are fed from video ops: The image SHAPE is lifted to a video
    SHAPE ([H,W,C] -> [1,H,W,C]).
    Typical use case: Training using images in a video setting. This op is added
    before the video preprocessing in order not to change the standard video
    preprocessing.
  """

  image_key: str = IMAGE
  video_key: str = VIDEO
  boxes_key: str = BOXES
  padding_mask_key: str = PADDING_MASK
  segmentations_key: str = SEGMENTATIONS
  sparse_segmentations_key: str = SPARSE_SEGMENTATIONS
  shape_key: str = SHAPE
  original_size_key: str = ORIGINAL_SIZE

  def __call__(self, features):
    assert self.image_key in features

    features_new = {}
    for k, v in features.items():
      if k == self.image_key:
        features_new[self.video_key] = v[tf.newaxis]
      elif k in (self.padding_mask_key, self.boxes_key, self.segmentations_key,
                 self.sparse_segmentations_key):
        features_new[k] = v[tf.newaxis]
      elif k == self.original_size_key:
        pass  # See comment in the docstring of the class.
      else:
        features_new[k] = v

    if self.original_size_key in features:
      # The features come from an image preprocessing pipeline.
      shape = tf.concat([[1], features[self.original_size_key],
                         [features[self.image_key].shape[-1]]],  # pytype: disable=attribute-error  # allow-recursive-types
                        axis=0)
    elif self.shape_key in features:
      # The features come from a video preprocessing pipeline.
      shape = tf.concat([[1], features[self.shape_key]], axis=0)
    else:
      shape = tf.shape(features_new[self.video_key])
    features_new[self.shape_key] = shape

    if self.padding_mask_key not in features_new:
      features_new[self.padding_mask_key] = tf.cast(
          tf.ones_like(features_new[self.video_key])[Ellipsis, 0], tf.uint8)

    return features_new


@dataclasses.dataclass
class SparseToDenseAnnotation:
  """Converts the sparse to a dense representation."""

  max_instances: int = 10
  segmentations_key: str = SEGMENTATIONS

  def __call__(self, features):

    features_new = {}

    for k, v in features.items():

      if k == self.segmentations_key:
        # Dense segmentations are available for this dataset. It may be that
        # max_instances < max(features_new[self.segmentations_key]).
        # We prune out extra objects here.
        segmentations = v
        segmentations = tf.where(
            tf.less_equal(segmentations, self.max_instances), segmentations, 0)
        features_new[self.segmentations_key] = segmentations
      else:
        features_new[k] = v

    return features_new


class VideoPreprocessOp(abc.ABC):
  """Base class for all video preprocess ops."""

  video_key: str = VIDEO
  segmentations_key: str = SEGMENTATIONS
  padding_mask_key: str = PADDING_MASK
  boxes_key: str = BOXES
  flow_key: str = FLOW
  depth_key: str = DEPTH
  sparse_segmentations_key: str = SPARSE_SEGMENTATIONS

  def __call__(self, features):
    # Get current video shape.
    video_shape = tf.shape(features[self.video_key])
    # Assemble all feature keys that the op should be applied on.
    all_keys = [
        self.video_key, self.segmentations_key, self.padding_mask_key,
        self.flow_key, self.depth_key, self.sparse_segmentations_key,
        self.boxes_key
    ]
    # Apply the op to all features.
    for key in all_keys:
      if key in features:
        features[key] = self.apply(features[key], key, video_shape)
    return features

  @abc.abstractmethod
  def apply(self, tensor, key,
            video_shape):
    """Returns the transformed tensor.

    Args:
      tensor: Any of a set of different video modalites, e.g video, flow,
        bounding boxes, etc.
      key: a string that indicates what feature the tensor represents so that
        the apply function can take that into account.
      video_shape: The shape of the video (which is necessary for some
        transformations).
    """


class RandomVideoPreprocessOp(VideoPreprocessOp):
  """Base class for all random video preprocess ops."""

  def __call__(self, features):
    if features.get(SEED_KEY) is None:
      logging.warning(
          "Using random operation without seed. To avoid this "
          "please provide a seed in feature %s.", SEED_KEY)
      op_seed = tf.random.uniform(shape=(2,), maxval=2**32, dtype=tf.int64)
    else:
      features[SEED_KEY], op_seed = tf.unstack(
          tf.random.experimental.stateless_split(features[SEED_KEY]))
    # Get current video shape.
    video_shape = tf.shape(features[self.video_key])
    # Assemble all feature keys that the op should be applied on.
    all_keys = [
        self.video_key, self.segmentations_key, self.padding_mask_key,
        self.flow_key, self.depth_key, self.sparse_segmentations_key,
        self.boxes_key
    ]
    # Apply the op to all features.
    for key in all_keys:
      if key in features:
        features[key] = self.apply(features[key], op_seed, key, video_shape)
    return features

  @abc.abstractmethod
  def apply(self, tensor, seed, key,
            video_shape):
    """Returns the transformed tensor.

    Args:
      tensor: Any of a set of different video modalites, e.g video, flow,
        bounding boxes, etc.
      seed: A random seed.
      key: a string that indicates what feature the tensor represents so that
        the apply function can take that into account.
      video_shape: The shape of the video (which is necessary for some
        transformations).
    """


@dataclasses.dataclass
class ResizeSmall(VideoPreprocessOp):
  """Resizes the smaller (spatial) side to `size` keeping aspect ratio.

  Attr:
    size: An integer representing the new size of the smaller side of the input.
    max_size: If set, an integer representing the maximum size in terms of the
      largest side of the input.
  """

  size: int
  max_size: Optional[int] = None

  def apply(self, tensor, key=None, video_shape=None):
    """See base class."""

    # Boxes are defined in normalized image coordinates and are not affected.
    if key == self.boxes_key:
      return tensor

    if key in (self.padding_mask_key, self.segmentations_key):
      tensor = tensor[Ellipsis, tf.newaxis]
    elif key == self.sparse_segmentations_key:
      tensor = tf.reshape(tensor,
                          (-1, tf.shape(tensor)[2], tf.shape(tensor)[3], 1))

    h, w = tf.shape(tensor)[1], tf.shape(tensor)[2]

    # Determine resize method based on dtype (e.g. segmentations are int).
    if tensor.dtype.is_integer:
      resize_method = "nearest"
    else:
      resize_method = "bilinear"

    # Clip size to max_size if needed.
    small_size = self.size
    if self.max_size is not None:
      small_size = adjust_small_size(
          original_size=(h, w), small_size=small_size, max_size=self.max_size)
    new_h, new_w = get_resize_small_shape(
        original_size=(h, w), small_size=small_size)
    tensor = tf.image.resize(tensor, [new_h, new_w], method=resize_method)

    # Flow needs to be rescaled according to the new size to stay valid.
    if key == self.flow_key:
      scale_h = tf.cast(new_h, tf.float32) / tf.cast(h, tf.float32)
      scale_w = tf.cast(new_w, tf.float32) / tf.cast(w, tf.float32)
      scale = tf.reshape(tf.stack([scale_h, scale_w], axis=0), (1, 2))
      # Optionally repeat scale in case both forward and backward flow are
      # stacked in the last dimension.
      scale = tf.repeat(scale, tf.shape(tensor)[-1] // 2, axis=0)
      scale = tf.reshape(scale, (1, 1, 1, tf.shape(tensor)[-1]))
      tensor *= scale

    if key in (self.padding_mask_key, self.segmentations_key):
      tensor = tensor[Ellipsis, 0]
    elif key == self.sparse_segmentations_key:
      tensor = tf.reshape(tensor, (video_shape[0], -1, new_h, new_w))

    return tensor


@dataclasses.dataclass
class CentralCrop(VideoPreprocessOp):
  """Makes central (spatial) crop of a given size.

  Attr:
    height: An integer representing the height of the crop.
    width: An (optional) integer representing the width of the crop. Make square
      crop if width is not provided.
  """

  height: int
  width: Optional[int] = None

  def apply(self, tensor, key=None, video_shape=None):
    """See base class."""
    if key == self.boxes_key:
      width = self.width or self.height
      h_orig, w_orig = video_shape[1], video_shape[2]
      top = (h_orig - self.height) // 2
      left = (w_orig - width) // 2
      tensor = crop_or_pad_boxes(tensor, top, left, self.height,
                                 width, h_orig, w_orig)
      return tensor
    else:
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[Ellipsis, tf.newaxis]
      seq_len, n_channels = tensor.get_shape()[0], tensor.get_shape()[3]
      h_orig, w_orig = tf.shape(tensor)[1], tf.shape(tensor)[2]
      width = self.width or self.height
      crop_size = (seq_len, self.height, width, n_channels)
      top = (h_orig - self.height) // 2
      left = (w_orig - width) // 2
      tensor = tf.image.crop_to_bounding_box(tensor, top, left, self.height,
                                             width)
      tensor = tf.ensure_shape(tensor, crop_size)
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[Ellipsis, 0]
      return tensor


@dataclasses.dataclass
class CropOrPad(VideoPreprocessOp):
  """Spatially crops or pads a video to a specified size.

  Attr:
    height: An integer representing the new height of the video.
    width: An integer representing the new width of the video.
    allow_crop: A boolean indicating if cropping is allowed.
  """

  height: int
  width: int
  allow_crop: bool = True

  def apply(self, tensor, key=None, video_shape=None):
    """See base class."""
    if key == self.boxes_key:
      # Pad and crop the spatial dimensions.
      h_orig, w_orig = video_shape[1], video_shape[2]
      if self.allow_crop:
        # After cropping, the frame shape is always [self.height, self.width].
        height, width = self.height, self.width
      else:
        # If only padding is performed, the frame size is at least
        # [self.height, self.width].
        height = tf.maximum(h_orig, self.height)
        width = tf.maximum(w_orig, self.width)
      tensor = crop_or_pad_boxes(
          tensor,
          top=0,
          left=0,
          height=height,
          width=width,
          h_orig=h_orig,
          w_orig=w_orig)
      return tensor
    elif key == self.sparse_segmentations_key:
      seq_len = tensor.get_shape()[0]
      paddings = get_paddings(
          tf.shape(tensor[Ellipsis, tf.newaxis]), (self.height, self.width),
          allow_crop=self.allow_crop)[:-1]
      tensor = tf.pad(tensor, paddings, constant_values=0)
      if self.allow_crop:
        tensor = tensor[Ellipsis, :self.height, :self.width]
      tensor = tf.ensure_shape(
          tensor, (seq_len, None, self.height, self.width))
      return tensor
    else:
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[Ellipsis, tf.newaxis]
      seq_len, n_channels = tensor.get_shape()[0], tensor.get_shape()[3]
      paddings = get_paddings(
          tf.shape(tensor), (self.height, self.width),
          allow_crop=self.allow_crop)
      tensor = tf.pad(tensor, paddings, constant_values=0)
      if self.allow_crop:
        tensor = tensor[:, :self.height, :self.width, :]
      tensor = tf.ensure_shape(tensor,
                               (seq_len, self.height, self.width, n_channels))
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[Ellipsis, 0]
      return tensor


@dataclasses.dataclass
class RandomCrop(RandomVideoPreprocessOp):
  """Gets a random (width, height) crop of input video.

  Assumption: Height and width are the same for all video-like modalities.

  Attr:
    height: An integer representing the height of the crop.
    width: An integer representing the width of the crop.
  """

  height: int
  width: int

  def apply(self, tensor, seed, key=None, video_shape=None):
    """See base class."""
    if key == self.boxes_key:
      # We copy the random generation part from tf.image.stateless_random_crop
      # to generate exactly the same offset as for the video.
      crop_size = (video_shape[0], self.height, self.width, video_shape[-1])
      size = tf.convert_to_tensor(crop_size, tf.int32)
      limit = video_shape - size + 1
      offset = tf.random.stateless_uniform(
          tf.shape(video_shape), dtype=tf.int32, maxval=tf.int32.max,
          seed=seed) % limit
      tensor = crop_or_pad_boxes(tensor, offset[1], offset[2], self.height,
                                 self.width, video_shape[1], video_shape[2])
      return tensor
    elif key == self.sparse_segmentations_key:
      raise NotImplementedError("Sparse segmentations aren't supported yet")
    else:
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[Ellipsis, tf.newaxis]
      seq_len, n_channels = tensor.get_shape()[0], tensor.get_shape()[3]
      crop_size = (seq_len, self.height, self.width, n_channels)
      tensor = tf.image.stateless_random_crop(tensor, size=crop_size, seed=seed)
      tensor = tf.ensure_shape(tensor, crop_size)
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[Ellipsis, 0]
      return tensor


@dataclasses.dataclass
class DropFrames(VideoPreprocessOp):
  """Subsamples a video by skipping frames.

  Attr:
    frame_skip: An integer representing the subsampling frequency of the video,
      where 1 means no frames are skipped, 2 means every other frame is skipped,
      and so forth.
  """

  frame_skip: int

  def apply(self, tensor, key=None, video_shape=None):
    """See base class."""
    del key
    del video_shape
    tensor = tensor[::self.frame_skip]
    new_length = tensor.get_shape()[0]
    tensor = tf.ensure_shape(tensor, [new_length] + tensor.get_shape()[1:])
    return tensor


@dataclasses.dataclass
class TemporalCropOrPad(VideoPreprocessOp):
  """Crops or pads a video in time to a specified length.

  Attr:
    length: An integer representing the new length of the video.
    allow_crop: A boolean, specifying whether temporal cropping is allowed. If
      False, will throw an error if length of the video is more than "length"
  """

  length: int
  allow_crop: bool = True

  def _apply(self, tensor, constant_values):
    frames_to_pad = self.length - tf.shape(tensor)[0]
    if self.allow_crop:
      frames_to_pad = tf.maximum(frames_to_pad, 0)
    tensor = tf.pad(
        tensor, ((0, frames_to_pad),) + ((0, 0),) * (len(tensor.shape) - 1),
        constant_values=constant_values)
    tensor = tensor[:self.length]
    tensor = tf.ensure_shape(tensor, [self.length] + tensor.get_shape()[1:])
    return tensor

  def apply(self, tensor, key=None, video_shape=None):
    """See base class."""
    del video_shape
    if key == self.boxes_key:
      constant_values = NOTRACK_BOX[0]
    else:
      constant_values = 0
    return self._apply(tensor, constant_values=constant_values)


@dataclasses.dataclass
class TemporalRandomWindow(RandomVideoPreprocessOp):
  """Gets a random slice (window) along 0-th axis of input tensor.

  Pads the video if the video length is shorter than the provided length.

  Assumption: The number of frames is the same for all video-like modalities.

  Attr:
    length: An integer representing the new length of the video.
  """

  length: int

  def _apply(self, tensor, seed, constant_values):
    length = tf.minimum(self.length, tf.shape(tensor)[0])
    frames_to_pad = tf.maximum(self.length - tf.shape(tensor)[0], 0)
    window_size = tf.concat(([length], tf.shape(tensor)[1:]), axis=0)
    tensor = tf.image.stateless_random_crop(tensor, size=window_size, seed=seed)
    tensor = tf.pad(
        tensor, ((0, frames_to_pad),) + ((0, 0),) * (len(tensor.shape) - 1),
        constant_values=constant_values)
    tensor = tf.ensure_shape(tensor, [self.length] + tensor.get_shape()[1:])
    return tensor

  def apply(self, tensor, seed, key=None, video_shape=None):
    """See base class."""
    del video_shape
    if key == self.boxes_key:
      constant_values = NOTRACK_BOX[0]
    else:
      constant_values = 0
    return self._apply(tensor, seed, constant_values=constant_values)


@dataclasses.dataclass
class TemporalRandomStridedWindow(RandomVideoPreprocessOp):
  """Gets a random strided slice (window) along 0-th axis of input tensor.

  This op is like TemporalRandomWindow but it samples from one of a set of
  strides of the video, whereas TemporalRandomWindow will densely sample from
  all possible slices of `length` frames from the video.

  For the following video and `length=3`: [1, 2, 3, 4, 5, 6, 7, 8, 9]

  This op will return one of [1, 2, 3], [4, 5, 6], or [7, 8, 9]

  This pads the video if the video length is shorter than the provided length.

  Assumption: The number of frames is the same for all video-like modalities.

  Attr:
    length: An integer representing the new length of the video and the sampling
      stride width.
  """

  length: int

  def _apply(self, tensor, seed,
             constant_values):
    """Applies the strided crop operation to the video tensor."""
    num_frames = tf.shape(tensor)[0]
    num_crop_points = tf.cast(tf.math.ceil(num_frames / self.length), tf.int32)
    crop_point = tf.random.stateless_uniform(
        shape=(), minval=0, maxval=num_crop_points, dtype=tf.int32, seed=seed)
    crop_point *= self.length
    frames_sample = tensor[crop_point:crop_point + self.length]
    frames_to_pad = tf.maximum(self.length - tf.shape(frames_sample)[0], 0)
    frames_sample = tf.pad(
        frames_sample,
        ((0, frames_to_pad),) + ((0, 0),) * (len(frames_sample.shape) - 1),
        constant_values=constant_values)
    frames_sample = tf.ensure_shape(frames_sample, [self.length] +
                                    frames_sample.get_shape()[1:])
    return frames_sample

  def apply(self, tensor, seed, key=None, video_shape=None):
    """See base class."""
    del video_shape
    if key == self.boxes_key:
      constant_values = NOTRACK_BOX[0]
    else:
      constant_values = 0
    return self._apply(tensor, seed, constant_values=constant_values)


@dataclasses.dataclass
class FlowToRgb:
  """Converts flow to an RGB image.

  NOTE: This operation requires a statically known shape for the input flow,
    i.e. it is best to place it as final operation into the preprocessing
    pipeline after all shapes are statically known (e.g. after cropping /
    padding).
  """
  flow_key: str = FLOW

  def __call__(self, features):
    if self.flow_key in features:
      flow_rgb = flow_tensor_to_rgb_tensor(features[self.flow_key])
      assert flow_rgb.dtype == tf.uint8
      features[self.flow_key] = tf.image.convert_image_dtype(
          flow_rgb, tf.float32)
    return features


@dataclasses.dataclass
class TransformDepth:
  """Applies one of several possible transformations to depth features."""
  transform: str
  depth_key: str = DEPTH

  def __call__(self, features):
    if self.depth_key in features:
      if self.transform == "log":
        depth_norm = tf.math.log(features[self.depth_key])
      elif self.transform == "log_plus":
        depth_norm = tf.math.log(1. + features[self.depth_key])
      elif self.transform == "invert_plus":
        depth_norm = 1. / (1. + features[self.depth_key])
      else:
        raise ValueError(f"Unknown depth transformation {self.transform}")

      features[self.depth_key] = depth_norm
    return features


@dataclasses.dataclass
class RandomResizedCrop(RandomVideoPreprocessOp):
  """Random-resized crop for each of the two views.

  Assumption: Height and width are the same for all video-like modalities.

  We randomly crop the input and record the transformation this crop corresponds
  to as a new feature. Croped images are resized to (height, width). Boxes are
  corrected adjusted and boxes outside the crop are discarded. Flow is rescaled
  so as to be pixel accurate after the operation. lidar_points_2d are
  transformed using the computed transformation. These points may lie outside
  the image after the operation.

  Attr:
    height: An integer representing the height to resize to.
    width: An integer representing the width to resize to.
    min_object_covered, aspect_ratio_range, area_range, max_attempts: See
      docstring of `stateless_sample_distorted_bounding_box`. Aspect ratio range
      has not been scaled by target aspect ratio. This differs from other
      implementations of this data augmentation.
    relative_box_area_threshold: If ratio of areas before and after cropping are
      lower than this threshold, then the box is discarded (set to NOTRACK_BOX).
  """
  # Target size.
  height: int
  width: int

  # Crop sampling attributes.
  min_object_covered: float = 0.1
  aspect_ratio_range: Tuple[float, float] = (3. / 4., 4. / 3.)
  area_range: Tuple[float, float] = (0.08, 1.0)
  max_attempts: int = 100

  # Box retention attributes
  relative_box_area_threshold: float = 0.0

  def apply(self, tensor, seed, key,
            video_shape):
    """Applies the crop operation on tensor."""
    param = self.sample_augmentation_params(video_shape, seed)
    si, sj = param[0], param[1]
    crop_h, crop_w = param[2], param[3]

    to_float32 = lambda x: tf.cast(x, tf.float32)

    if key == self.boxes_key:
      # First crop the boxes.
      cropped_boxes = crop_or_pad_boxes(
          tensor, si, sj,
          crop_h, crop_w,
          video_shape[1], video_shape[2])
      # We do not need to scale the boxes because they are in normalized coords.
      resized_boxes = cropped_boxes
      # Lastly detects NOTRACK_BOX boxes and avoid manipulating those.
      no_track_boxes = tf.convert_to_tensor(NOTRACK_BOX)
      no_track_boxes = tf.reshape(no_track_boxes, [1, 4])
      resized_boxes = tf.where(
          tf.reduce_all(tensor == no_track_boxes, axis=-1, keepdims=True),
          tensor, resized_boxes)

      if self.relative_box_area_threshold > 0:
        # Thresholds boxes that have been cropped too much, as in their area is
        # lower, in relative terms, than `relative_box_area_threshold`.
        area_before_crop = tf.reduce_prod(tensor[Ellipsis, 2:] - tensor[Ellipsis, :2],
                                          axis=-1)
        # Sets minimum area_before_crop to 1e-8 we avoid divisions by 0.
        area_before_crop = tf.maximum(area_before_crop,
                                      tf.zeros_like(area_before_crop) + 1e-8)
        area_after_crop = tf.reduce_prod(
            resized_boxes[Ellipsis, 2:] - resized_boxes[Ellipsis, :2], axis=-1)
        # As the boxes have normalized coordinates, they need to be rescaled to
        # be compared against the original uncropped boxes.
        scale_x = to_float32(crop_w) / to_float32(self.width)
        scale_y = to_float32(crop_h) / to_float32(self.height)
        area_after_crop *= scale_x * scale_y

        ratio = area_after_crop / area_before_crop
        return tf.where(
            tf.expand_dims(ratio > self.relative_box_area_threshold, -1),
            resized_boxes, no_track_boxes)

      else:
        return resized_boxes

    else:
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[Ellipsis, tf.newaxis]

      # Crop.
      seq_len, n_channels = tensor.get_shape()[0], tensor.get_shape()[3]
      crop_size = (seq_len, crop_h, crop_w, n_channels)
      tensor = tf.slice(tensor, tf.stack([0, si, sj, 0]), crop_size)

      # Resize.
      resize_method = tf.image.ResizeMethod.BILINEAR
      if (tensor.dtype == tf.int32 or tensor.dtype == tf.int64 or
          tensor.dtype == tf.uint8):
        resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
      tensor = tf.image.resize(tensor, [self.height, self.width],
                               method=resize_method)
      out_size = (seq_len, self.height, self.width, n_channels)
      tensor = tf.ensure_shape(tensor, out_size)

      if key == self.flow_key:
        # Rescales optical flow.
        scale_x = to_float32(self.width) / to_float32(crop_w)
        scale_y = to_float32(self.height) / to_float32(crop_h)
        tensor = tf.stack(
            [tensor[Ellipsis, 0] * scale_y, tensor[Ellipsis, 1] * scale_x], axis=-1)

      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[Ellipsis, 0]
      return tensor

  def sample_augmentation_params(self, video_shape, rng):
    """Sample a random bounding box for the crop."""
    sample_bbox = tf.image.stateless_sample_distorted_bounding_box(
        video_shape[1:],
        bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0],
                                   dtype=tf.float32, shape=[1, 1, 4]),
        seed=rng,
        min_object_covered=self.min_object_covered,
        aspect_ratio_range=self.aspect_ratio_range,
        area_range=self.area_range,
        max_attempts=self.max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_bbox

    # The specified bounding box provides crop coordinates.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)

    return tf.stack([offset_y, offset_x, target_height, target_width])

  def estimate_transformation(self, param, video_shape
                              ):
    """Computes the affine transformation for crop params.

    Args:
      param: Crop parameters in the [y, x, h, w] format of shape [4,].
      video_shape: Unused.

    Returns:
      Affine transformation of shape [3, 3] corresponding to cropping the image
      at [y, x] of size [h, w] and resizing it into [self.height, self.width].
    """
    del video_shape
    crop = tf.cast(param, tf.float32)
    si, sj = crop[0], crop[1]
    crop_h, crop_w = crop[2], crop[3]
    ei, ej = si + crop_h - 1.0, sj + crop_w - 1.0
    h, w = float(self.height), float(self.width)

    a1 = (ei - si + 1.)/h
    a2 = 0.
    a3 = si - 0.5 + a1 / 2.
    a4 = 0.
    a5 = (ej - sj + 1.)/w
    a6 = sj - 0.5 + a5 / 2.
    affine = tf.stack([a1, a2, a3, a4, a5, a6, 0., 0., 1.])
    return tf.reshape(affine, [3, 3])


@dataclasses.dataclass
class TfdsImageToTfdsVideo:
  """Lift TFDS image format to TFDS video format by adding a temporal axis.

  This op is intended to be called directly before VideoFromTfds.
  """

  TFDS_SEGMENTATIONS_KEY = "segmentations"
  TFDS_INSTANCES_KEY = "instances"
  TFDS_BOXES_KEY = "bboxes"
  TFDS_BOXES_FRAMES_KEY = "bbox_frames"

  image_key: str = IMAGE
  video_key: str = VIDEO
  boxes_image_key: str = BOXES
  boxes_key: str = BOXES_VIDEO
  image_padding_mask_key: str = IMAGE_PADDING_MASK
  video_padding_mask_key: str = VIDEO_PADDING_MASK
  depth_key: str = DEPTH
  depth_mask_key: str = "depth_mask"
  force_overwrite: bool = False

  def __call__(self, features):
    if self.video_key in features and not self.force_overwrite:
      return features

    features_new = {}
    for k, v in features.items():
      if k == self.image_key:
        features_new[self.video_key] = v[tf.newaxis]
      elif k == self.image_padding_mask_key:
        features_new[self.video_padding_mask_key] = v[tf.newaxis]
      elif k == self.boxes_image_key:
        features_new[self.boxes_key] = v[tf.newaxis]
      elif k == self.TFDS_SEGMENTATIONS_KEY:
        features_new[self.TFDS_SEGMENTATIONS_KEY] = v[tf.newaxis]
      elif k == self.TFDS_INSTANCES_KEY and self.TFDS_BOXES_KEY in v:
        # Add sequence dimension to boxes and create boxes frames for indexing.
        features_new[k] = v

        # Create dummy ragged tensor (1, None) and broadcast
        dummy = tf.ragged.constant([[0]], dtype=tf.int32)
        boxes_frames_value = tf.zeros_like(
            v[self.TFDS_BOXES_KEY][Ellipsis, 0], dtype=tf.int32)[Ellipsis, tf.newaxis]
        features_new[k][self.TFDS_BOXES_FRAMES_KEY] = boxes_frames_value + dummy
        # Create dummy ragged tensor (1, None, 1) and broadcast
        dummy = tf.ragged.constant([[0]], dtype=tf.float32)[Ellipsis, tf.newaxis]
        boxes_value = v[self.TFDS_BOXES_KEY][Ellipsis, tf.newaxis, :]
        features_new[k][self.TFDS_BOXES_KEY] = boxes_value + dummy
      elif k == self.depth_key:
        features_new[self.depth_key] = v[tf.newaxis]
      elif k == self.depth_mask_key:
        features_new[self.depth_mask_key] = v[tf.newaxis]
      else:
        features_new[k] = v

    if self.video_padding_mask_key not in features_new:
      logging.warning("Adding default video_padding_mask")
      features_new[self.video_padding_mask_key] = tf.cast(
          tf.ones_like(features_new[self.video_key])[Ellipsis, 0], tf.uint8)

    return features_new


@dataclasses.dataclass
class TopLeftCrop(VideoPreprocessOp):
  """Makes an arbitrary crop in all video frames.

  Attr:
    top: An integer representing the horizontal coordinate of the crop start.
    left: An integer representing the vertical coordinate of the crop start.
    height: An integer representing the height of the crop.
    width: An (optional) integer representing the width of the crop. Make square
      crop if width is not provided.
  """

  top: int
  left: int
  height: int
  width: Optional[int] = None

  def apply(self, tensor, key=None, video_shape=None):
    """See base class."""
    if key in (self.boxes_key,):
      width = self.width or self.height
      h_orig, w_orig = video_shape[1], video_shape[2]
      tensor = transforms.crop_or_pad_boxes(
          tensor, self.top, self.left, self.height, width, h_orig, w_orig)
      return tensor
    else:
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[Ellipsis, tf.newaxis]
      seq_len, n_channels = tensor.get_shape()[0], tensor.get_shape()[3]
      h_orig, w_orig = tf.shape(tensor)[1], tf.shape(tensor)[2]
      width = self.width or self.height
      crop_size = (seq_len, self.height, width, n_channels)
      tensor = tf.image.crop_to_bounding_box(
          tensor, self.top, self.left, self.height, width)
      tensor = tf.ensure_shape(tensor, crop_size)
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[Ellipsis, 0]
      return tensor


@dataclasses.dataclass
class DeleteSmallMasks:
  """Delete masks smaller than a selected fraction of pixels."""
  threshold: float = 0.05
  max_instances: int = 50
  max_instances_after: int = 11

  def __call__(self, features):

    features_new = {}

    for key in features.keys():

      if key == SEGMENTATIONS:
        seg = features[key]
        size = tf.shape(seg)

        assert_op = tf.Assert(
            tf.equal(size[0], 1), ["Implemented only for a single frame."])

        with tf.control_dependencies([assert_op]):
          # Delete time dimension.
          seg = seg[0]

          # Get the minimum number of pixels a masks needs to have.
          max_pixels = size[1] * size[2]
          threshold_pixels = tf.cast(
              tf.cast(max_pixels, tf.float32) * self.threshold, tf.int32)

          # Decompose the segmentation map as a single image for each instance.
          dec_seg = tf.stack(
              tf.map_fn(functools.partial(self._decompose, seg=seg),
                        tf.range(self.max_instances)), axis=0)

          # Count the pixels and find segmentation masks that are big enough.
          sums = tf.reduce_sum(dec_seg, axis=(1, 2))
          # We want the background to always be slot zero.
          # We can accomplish that be pretending it has the maximum
          # number of pixels.
          sums = tf.concat(
              [tf.ones_like(sums[0: 1]) * max_pixels, sums[1:]],
              axis=0)

          sort = tf.argsort(sums, axis=0, direction="DESCENDING")
          sums_s = tf.gather(sums, sort, axis=0)
          mask_s = tf.cast(tf.greater_equal(sums_s, threshold_pixels), tf.int32)

          dec_seg_plus = tf.stack(
              tf.map_fn(functools.partial(
                  self._compose_sort, seg=seg, sort=sort, mask_s=mask_s),
                        tf.range(self.max_instances_after)), axis=0)
          new_seg = tf.reduce_sum(dec_seg_plus, axis=0)

          features_new[key] = tf.cast(new_seg[None], tf.int32)

      else:
        # keep all other features
        features_new[key] = features[key]

    return features_new

  @classmethod
  def _decompose(cls, i, seg):
    return tf.cast(tf.equal(seg, i), tf.int32)

  @classmethod
  def _compose_sort(cls, i, seg, sort, mask_s):
    return tf.cast(tf.equal(seg, sort[i]), tf.int32) * i * mask_s[i]


@dataclasses.dataclass
class SundsToTfdsVideo:
  """Lift Sunds format to TFDS video format.

  Renames fields and adds a temporal axis.
  This op is intended to be called directly before VideoFromTfds.
  """

  SUNDS_IMAGE_KEY = "color_image"
  SUNDS_SEGMENTATIONS_KEY = "instance_image"
  SUNDS_DEPTH_KEY = "depth_image"

  image_key: str = SUNDS_IMAGE_KEY
  image_segmentations_key = SUNDS_SEGMENTATIONS_KEY
  video_key: str = VIDEO
  video_segmentations_key = SEGMENTATIONS
  image_depths_key: str = SUNDS_DEPTH_KEY
  depths_key = DEPTH
  video_padding_mask_key: str = VIDEO_PADDING_MASK
  force_overwrite: bool = False

  def __call__(self, features):
    if self.video_key in features and not self.force_overwrite:
      return features

    features_new = {}
    for k, v in features.items():
      if k == self.image_key:
        features_new[self.video_key] = v[tf.newaxis]
      elif k == self.image_segmentations_key:
        features_new[self.video_segmentations_key] = v[tf.newaxis]
      elif k == self.image_depths_key:
        features_new[self.depths_key] = v[tf.newaxis]
      else:
        features_new[k] = v

    if self.video_padding_mask_key not in features_new:
      logging.warning("Adding default video_padding_mask")
      features_new[self.video_padding_mask_key] = tf.cast(
          tf.ones_like(features_new[self.video_key])[Ellipsis, 0], tf.uint8)

    return features_new


@dataclasses.dataclass
class SubtractOneFromSegmentations:
  """Subtract one from segmentation masks. Used for MultiShapeNet-Easy."""

  segmentations_key: str = SEGMENTATIONS

  def __call__(self, features):
    features[self.segmentations_key] = features[self.segmentations_key] - 1
    return features
