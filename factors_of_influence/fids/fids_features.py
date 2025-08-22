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

"""Defines possible features and their types.

The goal of this file is to keep features as consistent as possible across
different datasets. Hence some common features are defined.

To add a new feature, the following needs to be done:

- Add function in TfdsTypes class, which returns the tfds feature connector.
  IMPORTANT: The function should be a staticmethod.
- If applicable: add to FeatureUtils.FEATURES_REQUIRED_IN_METADICT.

- The goal is to remain a light module, without heavy imports. This module is
  imported in utils.
"""
from typing import Any, Dict, Iterable, Text, Union

import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class TfdsTypes:
  """This class standardizes feature names with tfds types.

  IMPORTANT: each features should be defined in a staticmethod.
  """

  @staticmethod
  def id():
    return tfds.features.Text()

  @staticmethod
  def image():
    return tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8)

  @staticmethod
  def scene_class(num_scene_classes=None, list_scene_classes=None):
    return tfds.features.ClassLabel(
        num_classes=num_scene_classes, names=list_scene_classes)

  @staticmethod
  def segmentation():
    return tfds.features.Image(
        shape=(None, None, 1), dtype=tf.uint16, use_colormap=True)

  @staticmethod
  def instance_segmentation():
    return tfds.features.Image(
        shape=(None, None, 1), dtype=tf.uint16, use_colormap=True)

  @staticmethod
  def segmentation_mseg():
    return tfds.features.Image(
        shape=(None, None, 1), dtype=tf.uint16, use_colormap=True)

  @staticmethod
  def segmentation_mseg_relabeled():
    return tfds.features.Image(
        shape=(None, None, 1), dtype=tf.uint16, use_colormap=True)

  @staticmethod
  def segmentation_mseg_universal():
    return tfds.features.Image(
        shape=(None, None, 1), dtype=tf.uint16, use_colormap=True)

  @staticmethod
  def drivable():
    return tfds.features.Image(shape=(None, None, 1), dtype=tf.uint16)

  @staticmethod
  def boxes():
    """Its tensor is Nx4. Represention: [row_min, col_min, row_max, col_max]."""
    return tfds.features.Sequence(tfds.features.BBoxFeature())

  @staticmethod
  def box_labels(num_box_labels):
    return tfds.features.Sequence(
        tfds.features.Tensor(shape=[num_box_labels], dtype=tf.float32))

  @staticmethod
  def depth():
    """Encodes the depth of a scene in centimeters (float)."""
    return tfds.features.Tensor(
        shape=(None, None, 1), dtype=tf.float32, encoding='bytes')

  @staticmethod
  def face_boxes():
    return tfds.features.Sequence(tfds.features.BBoxFeature())

  @staticmethod
  def person_keypoints(num_keypoints):
    """Sequence of keypoint annotations: [row, col, is_visible].

    is_visible follows the COCO format and can have the values:
      0: not annotated.
      1: annotated and not visible.
      2: annotated and visible.

    Args:
      num_keypoints: number of keypoints.

    Returns:
      tfds FeatureConnector.
    """
    return tfds.features.Sequence(
        tfds.features.Tensor(shape=[num_keypoints, 3], dtype=tf.float32))

  @staticmethod
  def person_boxes():
    return tfds.features.Sequence(tfds.features.BBoxFeature())

  @staticmethod
  def person_position(dim_per_position):
    return tfds.features.Sequence(
        tfds.features.Tensor(shape=[dim_per_position], dtype=tf.float32))

  @classmethod
  def get_feature_names(cls):
    """Returns all feature names.

    This assumes that all feature functions are static.
    """
    feature_names = [
        fn_name for fn_name, fn in cls.__dict__.items()
        if isinstance(fn, staticmethod)
    ]
    return feature_names


class FeatureUtils:
  """Defines utility functions for features."""

  FEATURES_REQUIRED_IN_METADICT = {
      'box_labels': [],
      'depth': ['default_clip_min', 'default_clip_max'],
      'drivable': [],
      'person_keypoints': ['keypoint_names'],
      'segmentation': [],
      'segmentation_mseg': [],
      'segmentation_mseg_relabeled': [],
      'segmentation_mseg_universal': [],
  }

  def __init__(self,
               feature_args = None):
    """Constructor.

    Args:
      feature_args: dictionary which maps feature_name to named arguments. See
        functions in TfdsTypes for corresponding arguments.
    """
    self._tfds_types = TfdsTypes()
    self.feature_names = self._tfds_types.get_feature_names()
    self._feature_args = {f: {} for f in self.feature_names}

    if feature_args is not None:
      for feature_name, named_args in feature_args.items():
        # Fail at construction time if feature_args are wrong.
        _ = self._get_tfds_type(feature_name, named_args)

        self._feature_args[feature_name] = named_args

  def _get_tfds_type(
      self, feature_name,
      named_args):
    """Return tfds type for feature_name, given named arguments."""
    tfds_type_fn = getattr(self._tfds_types, feature_name)
    tfds_type = tfds_type_fn(**named_args)
    return tfds_type

  def _get_fake_feature_for_type(self, feature_type):
    """Get fake feature given TFDS feature type."""
    if isinstance(feature_type, tfds.features.Image):
      im_shape = feature_type.shape
      im_shape = [s if s is not None else 1 for s in im_shape]
      dtype = feature_type.dtype.as_numpy_dtype
      return np.zeros(im_shape, dtype=dtype)
    elif isinstance(feature_type, tfds.features.Sequence):
      return [self._get_fake_feature_for_type(feature_type.feature)]
    elif isinstance(feature_type, tfds.features.BBoxFeature):
      return tfds.features.BBox(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
    elif isinstance(feature_type, tfds.features.Tensor):
      tensor_shape = feature_type.shape
      tensor_shape = [s if s is not None else 1 for s in tensor_shape]
      dtype = feature_type.dtype.as_numpy_dtype
      return np.zeros(tensor_shape, dtype=dtype)
    else:
      raise NotImplementedError(f'for {feature_type}')

  def get_tfds_type(self, feature_name):
    """Returns tfds type for feature_name."""
    return self._get_tfds_type(feature_name, self._feature_args[feature_name])

  def get_fake_feature(self, feature):
    """Return fake TFDS feature, useful for missing values."""
    feature_type = self.get_tfds_type(feature)
    return self._get_fake_feature_for_type(feature_type)

  def get_tfds_features_dict(self, features):
    """Return features dict used in TFDS info structure.

    Ensures that the feature names also have an 'is_present' entry, which is a
    FeaturesDict of booleans. This provides a default mechanism for handling
    possible missing annotations and/or input data.

    Args:
      features: list of feature names.

    Returns:
      features.FeaturesDict, describing features and types of this dataset.
      This is used to create the TFDS DatasetInfo.
    """
    dataset_features = {f: self.get_tfds_type(f) for f in features}
    if 'id' not in dataset_features:
      dataset_features['id'] = self.get_tfds_type('id')

    is_present = {}
    for name in dataset_features:
      if name != 'id':
        is_present[name] = tfds.features.Tensor(shape=[], dtype=tf.bool)
    dataset_features['is_present'] = tfds.features.FeaturesDict(is_present)

    return tfds.features.FeaturesDict(dataset_features)
