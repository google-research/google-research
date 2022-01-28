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

"""ScreenAi specific utilities to deal with images and bounding boxes."""
import abc
import dataclasses as dc
import io
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TypeVar, Union

from immutabledict import immutabledict
import numpy as np
from PIL import Image
import skimage.color
import tensorflow as tf

from clay.proto import dimension_pb2

Array = TypeVar('Array', np.ndarray, tf.Tensor)
TBox = TypeVar('TBox', bound='Box')


@dc.dataclass(frozen=True)
class Box(abc.ABC):
  """Abstract base class to represent bounding boxes.

  Coordinates are normalized within image [0, 1].
  """
  left: float
  right: float
  top: float
  bottom: float

  def __post_init__(self):
    if not 0 <= self.left <= self.right <= 1:
      raise ValueError('Received invalid box (left, right) values outside [0,1]'
                       f' range: {(self.left, self.right)}')
    if not 0 <= self.top <= self.bottom <= 1:
      raise ValueError('Received invalid box (top, bottom) values outside [0,1]'
                       f' range: {(self.top, self.bottom)}')

  @property
  def xmin(self):
    return self.left

  @property
  def xmax(self):
    return self.right

  @property
  def ymin(self):
    return self.top

  @property
  def ymax(self):
    return self.bottom

  @property
  def width(self):
    """Relative width in range [0-1]."""
    return self.right - self.left

  @property
  def height(self):
    """Relative height in range [0-1]."""
    return self.bottom - self.top

  @property
  def aspect_ratio(self):
    """Relative aspect ratio. To be used together with image aspect ratio."""
    return self.width / self.height

  @property
  def area(self):
    """Relative area."""
    return self.width * self.height

  def area_px(self, image_height, image_width):
    """Returns box area in pixels.

    Converts to px coordinates and then counts pixels to preserve any rounding
    errors.

    Args:
      image_height: image height
      image_width: image width

    Returns:
      Number of pixels this box occupies.
    """
    ymin, xmin, ymax, xmax = self.get_absolute_coordinates(
        image_height, image_width)
    return (ymax - ymin) * (xmax - xmin)

  @property
  def x_center(self):
    """Horizontal center location."""
    return (self.left + self.right) / 2

  @property
  def y_center(self):
    """Vertical center location."""
    return (self.top + self.bottom) / 2

  def contains_center(self, other):
    """Check whether this box contains center of another box."""
    return (self.left <= other.x_center <= self.right and
            self.top <= other.y_center <= self.bottom)

  def get_absolute_coordinates(
      self, image_height, image_width):
    """Computes absolute integer coordinates given image size.

    xmin and ymin are inclusive in box while xmax and ymax are exclusive.

    Args:
      image_height: Image height.
      image_width: Image width.

    Returns:
      Tuple of coordinates in OD order: ymin, xmin, ymax, xmax.
    """
    xmin = int(round(image_width * self.left))
    xmax = int(round(image_width * self.right))
    ymin = int(round(image_height * self.top))
    ymax = int(round(image_height * self.bottom))
    return ymin, xmin, ymax, xmax

  def to_coco_format(self):
    """Convert to COCO format.

    http://cocodataset.org/#format-results
    Returns:
      list with 4 float numbers [xmin, ymin, width, height]
    """
    return [self.left, self.top, self.width, self.height]

  def to_object_det_format(self):
    """Convert to Object detection framework: [ymin, xmin, ymax, xmax].

    Returns:
      list with 4 float numbers [ymin, xmin, ymax, xmax]
    """
    return [self.top, self.left, self.bottom, self.right]

  def to_bounding_box_format(self):
    """Convert to ScreenAi BoundingBox proto object."""
    return dimension_pb2.BoundingBox(
        left=self.left, right=self.right, top=self.top, bottom=self.bottom)

  def replace(self, **changes):
    """Replace fields and return a copy."""
    return dc.replace(self, **changes)


@dc.dataclass(frozen=True)
class BBox(Box):
  """Immutable bounding box dataclass.

  Coordinates are normalized within image [0, 1].
  """
  ui_class: Optional[str] = None
  ui_label: Optional[int] = None
  score: Optional[float] = None
  # Field to store additional data related to a bbox.
  # For instance, it is not converted to other formats like COCO, OD, or ACUITI.
  metadata: Mapping[str, Any] = dc.field(default_factory=immutabledict)

  def __post_init__(self):
    super().__post_init__()
    if self.score is not None and not 0 <= self.score <= 1:
      raise ValueError('Received invalid prediction score outside [0,1] range: '
                       f'{self.score}')
    if not isinstance(self.metadata, Mapping):
      raise ValueError('metadate field must be a mapping. '
                       f'Received: {type(self.metadata)}.')
    if isinstance(self.metadata, dict):
      object.__setattr__(self, 'metadata', immutabledict(self.metadata))

  @classmethod
  def from_coco_format(cls,
                       box,
                       ui_class = None,
                       ui_label = None,
                       score = None):
    """Create BBox from COCO format: [xmin, ymin, width, height]."""
    kwargs = _remove_none(ui_class=ui_class, ui_label=ui_label, score=score)
    return cls(
        left=box[0],
        top=box[1],
        right=box[0] + box[2],
        bottom=box[1] + box[3],
        **kwargs)

  @classmethod
  def from_object_det_format(cls,
                             box,
                             ui_class = None,
                             ui_label = None,
                             score = None):
    """Create BBox from Object detection framework: [ymin, xmin, ymax, xmax]."""
    kwargs = _remove_none(ui_class=ui_class, ui_label=ui_label, score=score)
    return cls(top=box[0], left=box[1], bottom=box[2], right=box[3], **kwargs)

  @classmethod
  def from_bounding_box_format(cls,
                               box,
                               ui_class = None,
                               ui_label = None,
                               score = None):
    """Create BBox from ScreenAi BoundingBox proto object."""
    kwargs = _remove_none(ui_class=ui_class, ui_label=ui_label, score=score)
    return cls(
        left=box.left,
        right=box.right,
        top=box.top,
        bottom=box.bottom,
        **kwargs)

  # aliases
  from_od_format = from_object_det_format


def _remove_none(**kwargs):
  """Drop optional (None) arguments."""
  return {k: v for k, v in kwargs.items() if v is not None}


def intersect_area(box1, box2):
  """Compute the area of the intersection of two bounding boxes.

  Works on normalized coordinates.

  Args:
    box1: first bounding box.
    box2: second bounding box.

  Returns:
    area spanned by the intersection.
  """
  inter_top = max(box1.top, box2.top)
  inter_bottom = min(box1.bottom, box2.bottom)
  inter_left = max(box1.left, box2.left)
  inter_right = min(box1.right, box2.right)

  if inter_right <= inter_left or inter_bottom <= inter_top:
    return 0.0  # no intersection

  return (inter_right - inter_left) * (inter_bottom - inter_top)


def iou(box1, box2):
  """Intersection over union score for two bounding boxes.

  Works on normalized coordinates.

  Args:
    box1: first bounding box.
    box2: second bounding box.

  Returns:
    iou score between two boxes.
  """
  inter_area = intersect_area(box1, box2)
  if inter_area == 0.:
    return 0.

  box1_area = box1.width * box1.height
  box2_area = box2.width * box2.height

  iou_score = inter_area / (box1_area + box2_area - inter_area)
  assert 0.0 <= iou_score <= 1.0
  return iou_score


def overlap(inner, outer):
  """Returns how much of the area of inner is contained in outer box.

  Works on normalized coordinates.

  Args:
    inner: first bounding box.
    outer: second bounding box.

  Returns:
    the fraction of overlap of inner with outer box.
  """
  inter_area = intersect_area(inner, outer)
  if inter_area == 0.:
    return 0.

  overlap_fraction = inter_area / inner.area
  assert 0.0 <= overlap_fraction <= 1.0, (
      f'{inter_area}/{inner.area} should be between 0 '
      'and 1')
  return overlap_fraction


def smaller_box_overlap(box1, box2):
  """Same as #overlap(), but infers inner/outer box by size."""
  if box1.area > box2.area:
    box1, box2 = box2, box1
  return overlap(inner=box1, outer=box2)


def get_overlapping_box(box,
                        candidates,
                        threshold = 0.1):
  """Returns candidate with most area overlap with box (or None if none)."""
  overlapping = [(overlap(c, box), c) for c in candidates]
  overlapping = [pair for pair in overlapping if pair[0] >= threshold]
  if not overlapping:
    return None
  # Sort boxes by area of overlapping and output box with most overlap.
  overlapping.sort(reverse=True, key=lambda pair: pair[0])
  return overlapping[0][1]


def img_to_bytes(image,
                 img_format = 'PNG'):
  """Convert image to bytes for storing in tf.Example proto.

  Args:
   image: Either numpy array or PIL image to be serialized.
   img_format: Image serailization format.

  Returns:
    Serialized image.
  Raises:
    ValueError: if input image is empty.
  """
  if np.product(np.shape(image)) == 0:
    raise ValueError('Cannot serialize an empty image.')
  if not isinstance(image, Image.Image):
    if not np.issubdtype(image.dtype, np.integer):
      image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
  image_bytes = io.BytesIO()
  image.save(image_bytes, format=img_format)
  return image_bytes.getvalue()


def bytes_to_img_pil(image_bytes,
                     include_alpha = False):
  """Deserialize image from bytes."""
  image_bytes = io.BytesIO(image_bytes)
  image = Image.open(image_bytes)
  img_shape = np.shape(image)
  # Single channel images might only have 2 dimensions (HxW).
  n_channels = img_shape[-1] if len(np.shape(image)) == 3 else 1
  if n_channels == 4:
    if not include_alpha:
      # Use skimage for conversions since PIL struggles with icons images
      # where all information is in 4th channel.
      image = skimage.color.rgba2rgb(image)
      image = numpy_to_pil(image)
  elif n_channels != 3:
    # For images with a single (luminosity) or two channels
    # (luminosity+alpha) expand to RGBA and then to RGB.
    image = image.convert('RGBA')
    image = skimage.color.rgba2rgb(image)
    image = numpy_to_pil(image)
  return image


def pil_to_numpy(image):
  """Convert PIL image object (uint8) to numpy array (float64)."""
  return np.array(image) / 255


def numpy_to_pil(image):
  """Convert numpy array (float or int) to PIL image object (uint8)."""
  if not np.issubdtype(image.dtype, np.integer):
    # Scale float image to 0-255 range.
    image = (image * 255).astype(np.uint8)
  elif not np.issubdtype(image.dtype, np.uint8):
    # Alread in range 0-255, just cast to uint8.
    image = image.astype(np.uint8)

  return Image.fromarray(image)
