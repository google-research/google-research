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

"""Utility functions for tf.train.Example protos."""
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

from immutabledict import immutabledict
import numpy as np
from PIL import Image
import tensorflow as tf

from clay.utils import image_utils
from clay.utils import tf_example_fields
from clay.utils import view_hierarchy_fields
from clay.utils.tf_example_fields import TfExampleFields

Field = view_hierarchy_fields.Field
FieldName = Union[str, Field]
BBox = image_utils.BBox
PType = TypeVar('PType', bytes, str, int, np.integer, float, np.floating, bool)
RType = TypeVar('RType', bytes, str, int, float)
img_to_bytes = image_utils.img_to_bytes  # alias for convenient access

BBOX_GT_SCHEMA = immutabledict({
    'xmin': TfExampleFields.object_bbox_xmin,
    'xmax': TfExampleFields.object_bbox_xmax,
    'ymin': TfExampleFields.object_bbox_ymin,
    'ymax': TfExampleFields.object_bbox_ymax,
    'class_text': TfExampleFields.object_class_text,
    'class_label': TfExampleFields.object_class_label,
    'score': TfExampleFields.object_score,
})
BBOX_DT_SCHEMA = immutabledict({
    'xmin': TfExampleFields.detection_bbox_xmin,
    'xmax': TfExampleFields.detection_bbox_xmax,
    'ymin': TfExampleFields.detection_bbox_ymin,
    'ymax': TfExampleFields.detection_bbox_ymax,
    'class_text': TfExampleFields.detection_class_text,
    'class_label': TfExampleFields.detection_class_label,
    'score': TfExampleFields.detection_score,
})


def get_bboxes(
    example,
    *,
    skip_errors = False,
    box_schema = None,
    get_metadata = False,
):
  """Get bounding boxes from Example datapoint with label if it's present.

  Allows customizing field keys with prefix for box coordinates and custom keys
  for other fields. If metadata data is present in the tf.Example (prefixed with
  image/object/metadata/ or image/view_hierarchy/) and get_metadata is True, it
  is added to the bounding boxes.

  Args:
   example: example to extract boxes from.
   skip_errors: skip boxes with errors or throw errors.
   box_schema: tf.Example box field name schema. Can be passed to store GT or
     detected boxes. Defaults to GT boxes.
   get_metadata: whether to retrieve metadata information.

  Returns:
   List of bounding boxes.
  """
  box_schema = box_schema or BBOX_GT_SCHEMA
  features = example.features.feature
  bboxes = []
  class_texts = features[box_schema['class_text']].bytes_list.value
  labels = features[box_schema['class_label']].int64_list.value
  scores = features[box_schema['score']].float_list.value
  for i, bounds in enumerate(
      zip(features[box_schema['xmin']].float_list.value,
          features[box_schema['xmax']].float_list.value,
          features[box_schema['ymin']].float_list.value,
          features[box_schema['ymax']].float_list.value)):
    cls = str(class_texts[i], 'utf-8') if class_texts else None
    label = labels[i] if labels else None
    score = scores[i] if scores else None
    try:
      metadata = _get_metadata(example, i) if get_metadata else {}
      bbox = BBox(
          *bounds, ui_class=cls, ui_label=label, score=score, metadata=metadata)
      bboxes.append(bbox)

    except Exception:  # pylint: disable=broad-except
      if not skip_errors:
        raise

  return bboxes


def _get_metadata(example, idx):
  """Extracts metadata fields in tf.Example and returns them in a dictionary.

  Args:
   example: example to extract metadata from.
   idx: integer that indicates the position of the data in the feature vector.

  Returns:
    A dict that contains field names and their corresponding metadata.
  """
  metadata = {}
  for field_name in _get_metadata_keys(example):
    # We only remove the prefix if it starts with METADATA_PREFIX.
    if field_name.startswith(tf_example_fields.VIEW_HIERARCHY_PREFIX):
      field_name_no_metadata_prefix = field_name
    else:
      field_name_no_metadata_prefix = field_name.replace(
          tf_example_fields.METADATA_PREFIX, '', 1)
    features = get_feat_list(example, field_name)
    if features:
      assert len(features) > idx, f'{idx} >= feature length ({len(features)})'
      metadata[field_name_no_metadata_prefix] = features[idx]
  return metadata


def delete_bboxes(example,
                  box_schema = None):
  """Delete bounding boxes from example based on box_schema (in-place).

  Args:
    example: tf.Example from which the boxes will be removed.
    box_schema: tf.Example box field name schema. Can be passed to store GT or
      detected boxes. Defaults to GT boxes.
  """
  box_schema = box_schema or BBOX_GT_SCHEMA
  for field_name in box_schema.values():
    del_feat(example, field_name)
  _delete_metadata_fields(example)


def delete_detection_bboxes(example):
  """Delete detected boxes from example (in-place)."""
  delete_bboxes(example, box_schema=BBOX_DT_SCHEMA)


def _validate_bboxes(boxes):
  """Ensure that either all optional fields are set or None."""
  assert len(np.unique([b.ui_class is None for b in boxes])) <= 1
  assert len(np.unique([b.ui_label is None for b in boxes])) <= 1
  assert len(np.unique([b.score is None for b in boxes])) <= 1


def _validate_bboxes_for_insertion(boxes,
                                   example,
                                   box_schema):
  """Ensure that the optional fields are set (or not) consistently with the existing boxes."""

  def validate_field(field_name, box_fields):
    feats = get_feat_list(example, field_name)
    if feats:
      # If the feature exists in tf.Example, make sure the new boxes have it.
      assert np.sum([f is None for f in box_fields]) == 0
    elif get_feat_list(example, box_schema['ymin']):
      # If the feature does not exist (but some bounding boxes already exist),
      # verify that it is not set in the boxes to be added.
      assert np.sum([f is not None for f in box_fields]) == 0

  validate_field(box_schema['class_text'], [b.ui_class for b in boxes])
  validate_field(box_schema['class_label'], [b.ui_label for b in boxes])
  validate_field(box_schema['score'], [b.score for b in boxes])


def add_bboxes(example,
               boxes,
               *,
               validate_boxes=True,
               box_schema = None):
  """Add BBoxes to tf.Example object.

  Args:
   example: example to add bounding boxes to.
   boxes: new box(es) to be added.
   validate_boxes: whether to first validate boxes before insertion.
   box_schema: tf.Example box field name schema. Can be passed to store GT or
     detected boxes. Defaults to GT boxes.
  """
  box_schema = box_schema or BBOX_GT_SCHEMA
  if not isinstance(boxes, Iterable):
    boxes = [boxes]

  if validate_boxes:
    _validate_bboxes(boxes)
    _validate_bboxes_for_insertion(boxes, example, box_schema=box_schema)

  for box in boxes:
    add_feat(example, box_schema['xmin'], box.left)
    add_feat(example, box_schema['xmax'], box.right)
    add_feat(example, box_schema['ymin'], box.top)
    add_feat(example, box_schema['ymax'], box.bottom)
    add_feat(example, box_schema['class_text'], box.ui_class)
    add_feat(example, box_schema['class_label'], box.ui_label)
    add_feat(example, box_schema['score'], box.score)


def _get_metadata_keys(example):
  """Retrieves all keys that correspond to metadata fields."""
  return [
      key for key in example.features.feature.keys()
      if key.startswith(tf_example_fields.METADATA_PREFIX) or
      key.startswith(tf_example_fields.VIEW_HIERARCHY_PREFIX)
  ]


def _delete_metadata_fields(example):
  """Deletes metadata fields from tf.Example (in-place)."""
  for key in _get_metadata_keys(example):
    del_feat(example, key)


def get_image(example,
              new_size = None,
              include_alpha = False,
              field_name = TfExampleFields.image_encoded):
  """Get image numpy array with values ranging [0, 1].

  Args:
   example: example to extract the image from.
   new_size: tuple of int with new size (height, width), or None (no op).
   include_alpha: flag whether to include alpha (4th) channel.
   field_name: custom image field name to retrieve image from.

  Returns:
   The image (resized).
  """
  image = get_image_pil(
      example,
      new_size=new_size,
      include_alpha=include_alpha,
      field_name=field_name)
  return image_utils.pil_to_numpy(image)


def get_image_pil(
    example,
    new_size = None,
    include_alpha = False,
    field_name = TfExampleFields.image_encoded):
  """Returns a PIL image object deserialized from tf.Example field.

  Args:
   example: example to extract the image from.
   new_size: tuple of int with new size (height, width), or None (no op).
   include_alpha: flag whether to include alpha (4th) channel.
   field_name: custom image field name to retrieve image from.

  Returns:
   The image (resized).
  """
  image_bytes = example.features.feature[field_name].bytes_list.value[0]
  image = image_utils.bytes_to_img_pil(image_bytes, include_alpha=include_alpha)
  if new_size:
    new_size_pil = (new_size[1], new_size[0])
    image = image.resize(new_size_pil)
  return image


def get_image_tensor(
    example,
    new_size = None,
    include_alpha = False,
    field_name = TfExampleFields.image_encoded):
  """Returns image as a tensor with values ranging [0, 1].

  This function is not compatible with animated GIF images (if provided, it will
  return only the first frame).

  Args:
   example: example to extract the image from.
   new_size: tuple of int with new size (height, width), or None (no op).
   include_alpha: flag whether to include alpha (4th) channel.
   field_name: custom image field name to retrieve image from.

  Returns:
   The image (resized).
  """
  img_bytes = example.features.feature[field_name].bytes_list.value[0]
  img = tf.image.decode_image(img_bytes, expand_animations=False)
  if not include_alpha:
    img = img[Ellipsis, :3]  # take first three channels.
  img = tf.image.convert_image_dtype(img, tf.float32)
  if new_size:
    img = tf.image.resize(img, new_size, method='bilinear', name='resize_image')
  return img


def _get_shape_name(array_name, shape_name = None):
  """Either get shape name or create from array_name."""
  return shape_name if shape_name else f'{array_name}/shape'


def _get_field_name(name):
  """Unifies Fields and str types by converting Field to str."""
  if isinstance(name, Field):
    return name.name
  return name


def add_feat(
    example,
    name,
    feat,
):
  """Generic function to add feature to tf.Example.

  Empty Sequence is not stored in tf.Example.

  When storing 1D numpy array, the shape will not be stored. As a result,
  #get_np_feat will not work for data stored with #add_feat.

  Args:
   example: example to add feature to.
   name: name of the feature.
   feat: feature value or sequence of values or numpy array. None is skipped.
  """
  if feat is None:
    return
  name = _get_field_name(name)
  feature_obj = example.features.feature
  feat_type = type(feat)
  if np.issubdtype(feat_type, np.integer):
    feature_obj[name].int64_list.value.append(feat)
  elif np.issubdtype(feat_type, np.floating):
    feature_obj[name].float_list.value.append(feat)
  elif np.issubdtype(feat_type, np.str_):
    feature_obj[name].bytes_list.value.append(str(feat).encode('utf-8'))
  elif np.issubdtype(feat_type, np.bool_):
    feature_obj[name].int64_list.value.append(int(feat))
  elif isinstance(feat, bytes):
    feature_obj[name].bytes_list.value.append(feat)
  elif isinstance(feat, (Sequence, np.ndarray)):
    shape = np.shape(feat)
    if len(shape) > 1:
      raise ValueError('Passed multidimentional collection (shape: '
                       f'{np.shape(feat)}). Use #add_np_feat instead.')
    if shape:
      for f in feat:
        add_feat(example, name, f)
  else:
    raise AttributeError(f'Wrong feature type: {type(feat)} for key {name}.')


def set_feat(
    example,
    name,
    feat,
):
  """Generic function to set feature to tf.Example.

  If the feature already exists, it is replaced with the new value(s).

  Args:
   example: example to add feature to.
   name: name of the feature.
   feat: feature value or sequence of values or numpy array. None is skipped.
  """
  del_feat(example, name)
  add_feat(example, name, feat)


def get_feat_list(example,
                  name,
                  *,
                  decode = True,
                  infer_name = False):
  """Extract features from example as a list.

  Args:
    example: tf.Example to extract feature from.
    name: name of the feature.
    decode: if feature is bytes, should it be decoded to string (utf-8).
    infer_name: Enables partial feature name matching as long as it uniquely
      identifies a single feature. I.e. 'label' could extract
      'image/class/label' feature.

  Returns:
    A list of int, float, str, or bytes. Returns empty list if feature is not
      present in tf.Example.

  Raises:
    ValueError: When infer_name=True and name can refer to multiple fields.
  """
  name = _get_field_name(name)
  if name not in example.features.feature:
    if not infer_name:
      return []
    # try to find unique match for given partial name.
    partial_key_matches = 0
    last_match = None
    for ex_key in example.features.feature:
      if name in ex_key:
        partial_key_matches += 1
        last_match = ex_key
    if partial_key_matches == 1:
      name = last_match
    elif partial_key_matches > 1:
      raise ValueError(f'{partial_key_matches} matches for "{name}" name. '
                       f'Keys: {list(example.features.feature)}')
    else:
      return []

  feat = example.features.feature[name]
  feat_type = feat.WhichOneof('kind')
  if feat_type is None:
    return []

  feat_value_list = getattr(feat, feat_type).value
  if decode and feat_type == 'bytes_list':
    feat_value_list = [str(f, 'utf-8') for f in feat_value_list]

  return feat_value_list


def del_feat(
    example,
    name,
):
  """Delete feature if present."""
  name = _get_field_name(name)
  if name and name in example.features.feature:
    del example.features.feature[name]
