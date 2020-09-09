# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Keypoint profile class and utility functions."""

import abc
import enum

import six
import tensorflow as tf

from poem.core import keypoint_utils


class LeftRightType(enum.Enum):
  """Keypoint/segment left/right type."""
  UNKNOWN = 0
  CENTRAL = 1
  LEFT = 2
  RIGHT = 3


def infer_keypoint_left_right_type(left_right_types, indices):
  """Infers keypoint left/right type.

  The inferred left/right type is decided as follows:
  1. If either type is UNKNOWN, returns UNKNOWN.
  2. If both types are the same, returns this type.
  3. If one type is CENTRAL, and the other type is LEFT or RIGHT, returns the
     other type.
  4. If one type is LEFT and the other type is RIGHT, returns CENTRAL.

  Args:
    left_right_types: A list of LeftRightType enum values for all keypoints.
    indices: A list of integers for keypoint indices.

  Returns:
    A LeftRightType enum value for inferred type.

  Raises:
    ValueError: If any index is out of range.
  """
  if not indices:
    return LeftRightType.UNKNOWN

  def lookup(i):
    if i < 0 or i >= len(left_right_types):
      raise ValueError('Left/right type index is out of range: %d.' % i)
    return left_right_types[i]

  if len(indices) == 1:
    return lookup(indices[0])

  output_type = LeftRightType.CENTRAL
  for i in indices:
    current_type = lookup(i)
    if current_type == LeftRightType.UNKNOWN:
      return LeftRightType.UNKNOWN
    if output_type == LeftRightType.CENTRAL:
      output_type = current_type
    elif current_type != LeftRightType.CENTRAL and current_type != output_type:
      output_type = LeftRightType.CENTRAL
  return output_type


def infer_segment_left_right_type(left_right_types, start_indices, end_indices):
  """Infers segment left/right type.

  The inferred left/right type is decided as follows:
  1. If either type is UNKNOWN, returns UNKNOWN.
  2. If both types are the same, returns this type.
  3. If one type is CENTRAL, and the other type is LEFT or RIGHT, returns the
     other type.
  4. If one type is LEFT and the other type is RIGHT, returns CENTRAL.

  Args:
    left_right_types: A list of LeftRightType enum values for all keypoints.
    start_indices: A list of integers for LHS keypoint indices.
    end_indices: A list of integers for RHS keypoint indices.

  Returns:
    A LeftRightType enum value for inferred type.
  """
  lhs_type = infer_keypoint_left_right_type(left_right_types, start_indices)
  rhs_type = infer_keypoint_left_right_type(left_right_types, end_indices)
  if lhs_type == LeftRightType.UNKNOWN or rhs_type == LeftRightType.UNKNOWN:
    return LeftRightType.UNKNOWN
  if lhs_type == LeftRightType.CENTRAL:
    return rhs_type
  if rhs_type == LeftRightType.CENTRAL:
    return lhs_type
  return lhs_type if lhs_type == rhs_type else LeftRightType.CENTRAL


class KeypointProfile(six.with_metaclass(abc.ABCMeta, object)):
  """Keypoint profile base class."""

  def __init__(self,
               name,
               keypoint_names,
               offset_keypoint_names,
               scale_keypoint_name_pairs,
               scale_distance_reduction_fn,
               scale_unit,
               segment_name_pairs,
               head_keypoint_name=None,
               neck_keypoint_name=None,
               left_shoulder_keypoint_name=None,
               right_shoulder_keypoint_name=None,
               left_elbow_keypoint_name=None,
               right_elbow_keypoint_name=None,
               left_wrist_keypoint_name=None,
               right_wrist_keypoint_name=None,
               spine_keypoint_name=None,
               pelvis_keypoint_name=None,
               left_hip_keypoint_name=None,
               right_hip_keypoint_name=None,
               left_knee_keypoint_name=None,
               right_knee_keypoint_name=None,
               left_ankle_keypoint_name=None,
               right_ankle_keypoint_name=None):
    """Initializer."""
    self._name = name
    self._keypoint_names = [name for name, _ in keypoint_names]
    self._keypoint_left_right_types = [
        left_right_type for _, left_right_type in keypoint_names
    ]

    self._offset_keypoint_index = [
        self._keypoint_names.index(keypoint_name)
        for keypoint_name in offset_keypoint_names
    ]

    self._scale_keypoint_index_pairs = []
    for start_names, end_names in scale_keypoint_name_pairs:
      self._scale_keypoint_index_pairs.append(
          ([self._keypoint_names.index(name) for name in start_names],
           [self._keypoint_names.index(name) for name in end_names]))
    self._scale_distance_reduction_fn = scale_distance_reduction_fn
    self._scale_unit = scale_unit

    self._segment_index_pairs = []
    for start_names, end_names in segment_name_pairs:
      self._segment_index_pairs.append(
          ([self._keypoint_names.index(name) for name in start_names],
           [self._keypoint_names.index(name) for name in end_names]))

    self._head_keypoint_name = head_keypoint_name
    self._neck_keypoint_name = neck_keypoint_name
    self._left_shoulder_keypoint_name = left_shoulder_keypoint_name
    self._right_shoulder_keypoint_name = right_shoulder_keypoint_name
    self._left_elbow_keypoint_name = left_elbow_keypoint_name
    self._right_elbow_keypoint_name = right_elbow_keypoint_name
    self._left_wrist_keypoint_name = left_wrist_keypoint_name
    self._right_wrist_keypoint_name = right_wrist_keypoint_name
    self._spine_keypoint_name = spine_keypoint_name
    self._pelvis_keypoint_name = pelvis_keypoint_name
    self._left_hip_keypoint_name = left_hip_keypoint_name
    self._right_hip_keypoint_name = right_hip_keypoint_name
    self._left_knee_keypoint_name = left_knee_keypoint_name
    self._right_knee_keypoint_name = right_knee_keypoint_name
    self._left_ankle_keypoint_name = left_ankle_keypoint_name
    self._right_ankle_keypoint_name = right_ankle_keypoint_name

  @property
  def name(self):
    """Gets keypoint profile name."""
    return self._name

  @property
  def keypoint_names(self):
    """Gets keypoint names."""
    return self._keypoint_names

  @property
  @abc.abstractmethod
  def keypoint_dim(self):
    """Gets keypoint dimensionality."""
    raise NotImplementedError

  @property
  def keypoint_num(self):
    """Gets number of keypoints."""
    return len(self._keypoint_names)

  def keypoint_left_right_type(self, keypoint_index):
    """Gets keypoint left/right type given index."""
    if isinstance(keypoint_index, int):
      keypoint_index = [keypoint_index]
    return infer_keypoint_left_right_type(self._keypoint_left_right_types,
                                          keypoint_index)

  def segment_left_right_type(self, start_index, end_index):
    """Gets segment left/right type given index."""
    if isinstance(start_index, int):
      start_index = [start_index]
    if isinstance(end_index, int):
      end_index = [end_index]
    return infer_segment_left_right_type(self._keypoint_left_right_types,
                                         start_index, end_index)

  @property
  def offset_keypoint_index(self):
    """Gets offset keypoint index."""
    return self._offset_keypoint_index

  @property
  def scale_keypoint_index_pairs(self):
    """Gets scale keypoint index pairs."""
    return self._scale_keypoint_index_pairs

  @property
  def scale_unit(self):
    """Gets scale unit."""
    return self._scale_unit

  @property
  def segment_index_pairs(self):
    """Gets segment index pairs."""
    return self._segment_index_pairs

  @property
  def keypoint_affinity_matrix(self):
    """Gets keypoint affinity matrix.

    If a segment has multi-point end, all pairs of relevant points are
    considered as in affinity.

    Returns:
      matrix: A double list of floats for the keypoint affinity matrix.

    Raises:
      ValueError: If affinity matrix has any isolated node.
    """
    matrix = [[0.0
               for _ in range(self.keypoint_num)]
              for _ in range(self.keypoint_num)]

    # Self-affinity.
    for i in range(self.keypoint_num):
      matrix[i][i] = 1.0

    for lhs_index, rhs_index in self._segment_index_pairs:
      for i in lhs_index:
        for j in lhs_index:
          matrix[i][j] = 1.0
          matrix[j][i] = 1.0

      for i in rhs_index:
        for j in rhs_index:
          matrix[i][j] = 1.0
          matrix[j][i] = 1.0

      for i in lhs_index:
        for j in rhs_index:
          matrix[i][j] = 1.0
          matrix[j][i] = 1.0

    # Check if the affinity matrix is valid, i.e., each node must have degree
    # greater than 1 (no isolated node).
    for row in matrix:
      if sum(row) <= 1.0:
        raise ValueError(
            'Affinity matrix has a node with degree less than 2: %s.' %
            str(matrix))

    return matrix

  def keypoint_index(self, keypoint_name, raise_error_if_not_found=False):
    """Gets keypoint index given name.

    If `raise_error_if_not_found` is True, raises ValueError if keypoint does
    not exist. Otherwise, returns -1 if keypoint does not exist.

    Args:
      keypoint_name: A string for keypoint name to find index of.
      raise_error_if_not_found: A boolean for whether to raise ValueError if
        keypoint does not exist.

    Returns:
      An integer for keypoint index.

    Raises:
      ValueError: If keypoint does not exist and `raise_error_if_not_found` is
        True.
    """
    if keypoint_name in self._keypoint_names:
      return self._keypoint_names.index(keypoint_name)
    if raise_error_if_not_found:
      raise ValueError('Failed to find keypoint: `%s`.' % str(keypoint_name))
    return -1

  @property
  def head_keypoint_index(self):
    """Gets head keypoint index."""
    if not self._head_keypoint_name:
      raise ValueError('Head keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._head_keypoint_name
    ]

  @property
  def neck_keypoint_index(self):
    """Gets neck keypoint index."""
    if not self._neck_keypoint_name:
      raise ValueError('Neck keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._neck_keypoint_name
    ]

  @property
  def left_shoulder_keypoint_index(self):
    """Gets left shoulder keypoint index."""
    if not self._left_shoulder_keypoint_name:
      raise ValueError('Left shoulder keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._left_shoulder_keypoint_name
    ]

  @property
  def right_shoulder_keypoint_index(self):
    """Gets right shoulder keypoint index."""
    if not self._right_shoulder_keypoint_name:
      raise ValueError('Right shoulder keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._right_shoulder_keypoint_name
    ]

  @property
  def left_elbow_keypoint_index(self):
    """Gets left elbow keypoint index."""
    if not self._left_elbow_keypoint_name:
      raise ValueError('Left elbow keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._left_elbow_keypoint_name
    ]

  @property
  def right_elbow_keypoint_index(self):
    """Gets right elbow keypoint index."""
    if not self._right_elbow_keypoint_name:
      raise ValueError('Right elbow keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._right_elbow_keypoint_name
    ]

  @property
  def left_wrist_keypoint_index(self):
    """Gets left wrist keypoint index."""
    if not self._left_wrist_keypoint_name:
      raise ValueError('Left wrist keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._left_wrist_keypoint_name
    ]

  @property
  def right_wrist_keypoint_index(self):
    """Gets right wrist keypoint index."""
    if not self._right_wrist_keypoint_name:
      raise ValueError('Right wrist keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._right_wrist_keypoint_name
    ]

  @property
  def spine_keypoint_index(self):
    """Gets spine keypoint index."""
    if not self._spine_keypoint_name:
      raise ValueError('Spine keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._spine_keypoint_name
    ]

  @property
  def pelvis_keypoint_index(self):
    """Gets pelvis keypoint index."""
    if not self._pelvis_keypoint_name:
      raise ValueError('Pelvis keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._pelvis_keypoint_name
    ]

  @property
  def left_hip_keypoint_index(self):
    """Gets left hip keypoint index."""
    if not self._left_hip_keypoint_name:
      raise ValueError('Left hip keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._left_hip_keypoint_name
    ]

  @property
  def right_hip_keypoint_index(self):
    """Gets right hip keypoint index."""
    if not self._right_hip_keypoint_name:
      raise ValueError('Right hip keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._right_hip_keypoint_name
    ]

  @property
  def left_knee_keypoint_index(self):
    """Gets left knee keypoint index."""
    if not self._left_knee_keypoint_name:
      raise ValueError('Left knee keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._left_knee_keypoint_name
    ]

  @property
  def right_knee_keypoint_index(self):
    """Gets right knee keypoint index."""
    if not self._right_knee_keypoint_name:
      raise ValueError('Right knee keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._right_knee_keypoint_name
    ]

  @property
  def left_ankle_keypoint_index(self):
    """Gets left ankle keypoint index."""
    if not self._left_ankle_keypoint_name:
      raise ValueError('Left ankle keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._left_ankle_keypoint_name
    ]

  @property
  def right_ankle_keypoint_index(self):
    """Gets right ankle keypoint index."""
    if not self._right_ankle_keypoint_name:
      raise ValueError('Right ankle keypoint is not specified.')
    return [
        self.keypoint_index(name, raise_error_if_not_found=True)
        for name in self._right_ankle_keypoint_name
    ]

  @property
  def standard_part_names(self):
    """Gets all standard part names."""
    return [
        'HEAD', 'NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
        'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'SPINE', 'PELVIS',
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
        'RIGHT_ANKLE'
    ]

  def get_standard_part_index(self, part_name):
    """Gets part index by standardized name."""
    if part_name.upper() == 'HEAD':
      return self.head_keypoint_index
    if part_name.upper() == 'NECK':
      return self.neck_keypoint_index
    if part_name.upper() == 'LEFT_SHOULDER':
      return self.left_shoulder_keypoint_index
    if part_name.upper() == 'RIGHT_SHOULDER':
      return self.right_shoulder_keypoint_index
    if part_name.upper() == 'LEFT_ELBOW':
      return self.left_elbow_keypoint_index
    if part_name.upper() == 'RIGHT_ELBOW':
      return self.right_elbow_keypoint_index
    if part_name.upper() == 'LEFT_WRIST':
      return self.left_wrist_keypoint_index
    if part_name.upper() == 'RIGHT_WRIST':
      return self.right_wrist_keypoint_index
    if part_name.upper() == 'SPINE':
      return self.spine_keypoint_index
    if part_name.upper() == 'PELVIS':
      return self.pelvis_keypoint_index
    if part_name.upper() == 'LEFT_HIP':
      return self.left_hip_keypoint_index
    if part_name.upper() == 'RIGHT_HIP':
      return self.right_hip_keypoint_index
    if part_name.upper() == 'LEFT_KNEE':
      return self.left_knee_keypoint_index
    if part_name.upper() == 'RIGHT_KNEE':
      return self.right_knee_keypoint_index
    if part_name.upper() == 'LEFT_ANKLE':
      return self.left_ankle_keypoint_index
    if part_name.upper() == 'RIGHT_ANKLE':
      return self.right_ankle_keypoint_index
    raise ValueError('Unsupported part name: `%s`.' % part_name)

  def normalize(self, keypoints, keypoint_masks=None):
    """Normalizes keypoints."""
    del keypoint_masks
    return keypoint_utils.normalize_points(
        keypoints,
        offset_point_indices=self._offset_keypoint_index,
        scale_distance_point_index_pairs=self._scale_keypoint_index_pairs,
        scale_distance_reduction_fn=self._scale_distance_reduction_fn,
        scale_unit=self._scale_unit)

  def denormalize(self,
                  normalized_keypoints,
                  offset_points,
                  scale_distances,
                  keypoint_masks=None):
    """Denormalizes keypoints."""
    del keypoint_masks
    return (normalized_keypoints / self._scale_unit * scale_distances +
            offset_points)


class KeypointProfile3D(KeypointProfile):
  """3D keypoint profile base class."""

  def __init__(self,
               name,
               keypoint_names,
               offset_keypoint_names,
               scale_keypoint_name_pairs,
               segment_name_pairs,
               scale_distance_reduction_fn=tf.math.reduce_sum,
               scale_unit=1.0,
               head_keypoint_name=None,
               neck_keypoint_name=None,
               left_shoulder_keypoint_name=None,
               right_shoulder_keypoint_name=None,
               left_elbow_keypoint_name=None,
               right_elbow_keypoint_name=None,
               left_wrist_keypoint_name=None,
               right_wrist_keypoint_name=None,
               spine_keypoint_name=None,
               pelvis_keypoint_name=None,
               left_hip_keypoint_name=None,
               right_hip_keypoint_name=None,
               left_knee_keypoint_name=None,
               right_knee_keypoint_name=None,
               left_ankle_keypoint_name=None,
               right_ankle_keypoint_name=None):
    """Initializer."""
    super(KeypointProfile3D, self).__init__(
        name=name,
        keypoint_names=keypoint_names,
        offset_keypoint_names=offset_keypoint_names,
        scale_keypoint_name_pairs=scale_keypoint_name_pairs,
        scale_distance_reduction_fn=scale_distance_reduction_fn,
        scale_unit=scale_unit,
        segment_name_pairs=segment_name_pairs,
        head_keypoint_name=head_keypoint_name,
        neck_keypoint_name=neck_keypoint_name,
        left_shoulder_keypoint_name=left_shoulder_keypoint_name,
        right_shoulder_keypoint_name=right_shoulder_keypoint_name,
        left_elbow_keypoint_name=left_elbow_keypoint_name,
        right_elbow_keypoint_name=right_elbow_keypoint_name,
        left_wrist_keypoint_name=left_wrist_keypoint_name,
        right_wrist_keypoint_name=right_wrist_keypoint_name,
        spine_keypoint_name=spine_keypoint_name,
        pelvis_keypoint_name=pelvis_keypoint_name,
        left_hip_keypoint_name=left_hip_keypoint_name,
        right_hip_keypoint_name=right_hip_keypoint_name,
        left_knee_keypoint_name=left_knee_keypoint_name,
        right_knee_keypoint_name=right_knee_keypoint_name,
        left_ankle_keypoint_name=left_ankle_keypoint_name,
        right_ankle_keypoint_name=right_ankle_keypoint_name)

  @property
  def keypoint_dim(self):
    """Gets keypoint dimensionality."""
    return 3


class KeypointProfile2D(KeypointProfile):
  """2D keypoint profile base class."""

  def __init__(self,
               name,
               keypoint_names,
               offset_keypoint_names,
               scale_keypoint_name_pairs,
               segment_name_pairs,
               compatible_keypoint_name_dict=None,
               scale_distance_reduction_fn=tf.math.reduce_max,
               scale_unit=0.5,
               head_keypoint_name=None,
               neck_keypoint_name=None,
               left_shoulder_keypoint_name=None,
               right_shoulder_keypoint_name=None,
               left_elbow_keypoint_name=None,
               right_elbow_keypoint_name=None,
               left_wrist_keypoint_name=None,
               right_wrist_keypoint_name=None,
               spine_keypoint_name=None,
               pelvis_keypoint_name=None,
               left_hip_keypoint_name=None,
               right_hip_keypoint_name=None,
               left_knee_keypoint_name=None,
               right_knee_keypoint_name=None,
               left_ankle_keypoint_name=None,
               right_ankle_keypoint_name=None):
    """Initializer."""
    super(KeypointProfile2D, self).__init__(
        name=name,
        keypoint_names=keypoint_names,
        offset_keypoint_names=offset_keypoint_names,
        scale_keypoint_name_pairs=scale_keypoint_name_pairs,
        scale_distance_reduction_fn=scale_distance_reduction_fn,
        scale_unit=scale_unit,
        segment_name_pairs=segment_name_pairs,
        head_keypoint_name=head_keypoint_name,
        neck_keypoint_name=neck_keypoint_name,
        left_shoulder_keypoint_name=left_shoulder_keypoint_name,
        right_shoulder_keypoint_name=right_shoulder_keypoint_name,
        left_elbow_keypoint_name=left_elbow_keypoint_name,
        right_elbow_keypoint_name=right_elbow_keypoint_name,
        left_wrist_keypoint_name=left_wrist_keypoint_name,
        right_wrist_keypoint_name=right_wrist_keypoint_name,
        spine_keypoint_name=spine_keypoint_name,
        pelvis_keypoint_name=pelvis_keypoint_name,
        left_hip_keypoint_name=left_hip_keypoint_name,
        right_hip_keypoint_name=right_hip_keypoint_name,
        left_knee_keypoint_name=left_knee_keypoint_name,
        right_knee_keypoint_name=right_knee_keypoint_name,
        left_ankle_keypoint_name=left_ankle_keypoint_name,
        right_ankle_keypoint_name=right_ankle_keypoint_name)
    self._compatible_keypoint_name_dict = {}
    if compatible_keypoint_name_dict is not None:
      for _, compatible_keypoint_names in compatible_keypoint_name_dict.items():
        if len(compatible_keypoint_names) != len(self._keypoint_names):
          raise ValueError('Compatible keypoint names must be of the same size '
                           'as keypoint names.')
      self._compatible_keypoint_name_dict = compatible_keypoint_name_dict

  @property
  def keypoint_dim(self):
    """Gets keypoint dimensionality."""
    return 2

  @property
  def compatible_keypoint_name_dict(self):
    """Gets compatible keypoint name dictionary."""
    return self._compatible_keypoint_name_dict


class Std16KeypointProfile3D(KeypointProfile3D):
  """Standard 3D 16-keypoint profile."""

  def __init__(self):
    """Initializer."""
    super(Std16KeypointProfile3D,
          self).__init__(
              name='3DSTD16',
              keypoint_names=[('HEAD', LeftRightType.CENTRAL),
                              ('NECK', LeftRightType.CENTRAL),
                              ('LEFT_SHOULDER', LeftRightType.LEFT),
                              ('RIGHT_SHOULDER', LeftRightType.RIGHT),
                              ('LEFT_ELBOW', LeftRightType.LEFT),
                              ('RIGHT_ELBOW', LeftRightType.RIGHT),
                              ('LEFT_WRIST', LeftRightType.LEFT),
                              ('RIGHT_WRIST', LeftRightType.RIGHT),
                              ('SPINE', LeftRightType.CENTRAL),
                              ('PELVIS', LeftRightType.CENTRAL),
                              ('LEFT_HIP', LeftRightType.LEFT),
                              ('RIGHT_HIP', LeftRightType.RIGHT),
                              ('LEFT_KNEE', LeftRightType.LEFT),
                              ('RIGHT_KNEE', LeftRightType.RIGHT),
                              ('LEFT_ANKLE', LeftRightType.LEFT),
                              ('RIGHT_ANKLE', LeftRightType.RIGHT)],
              offset_keypoint_names=['PELVIS'],
              scale_keypoint_name_pairs=[(['NECK'], ['SPINE']),
                                         (['SPINE'], ['PELVIS'])],
              segment_name_pairs=[(['HEAD'], ['NECK']),
                                  (['NECK'], ['LEFT_SHOULDER']),
                                  (['NECK'], ['RIGHT_SHOULDER']),
                                  (['NECK'], ['SPINE']),
                                  (['LEFT_SHOULDER'], ['LEFT_ELBOW']),
                                  (['RIGHT_SHOULDER'], ['RIGHT_ELBOW']),
                                  (['LEFT_ELBOW'], ['LEFT_WRIST']),
                                  (['RIGHT_ELBOW'], ['RIGHT_WRIST']),
                                  (['SPINE'], ['PELVIS']),
                                  (['PELVIS'], ['LEFT_HIP']),
                                  (['PELVIS'], ['RIGHT_HIP']),
                                  (['LEFT_HIP'], ['LEFT_KNEE']),
                                  (['RIGHT_HIP'], ['RIGHT_KNEE']),
                                  (['LEFT_KNEE'], ['LEFT_ANKLE']),
                                  (['RIGHT_KNEE'], ['RIGHT_ANKLE'])],
              head_keypoint_name=['HEAD'],
              neck_keypoint_name=['NECK'],
              left_shoulder_keypoint_name=['LEFT_SHOULDER'],
              right_shoulder_keypoint_name=['RIGHT_SHOULDER'],
              left_elbow_keypoint_name=['LEFT_ELBOW'],
              right_elbow_keypoint_name=['RIGHT_ELBOW'],
              left_wrist_keypoint_name=['LEFT_WRIST'],
              right_wrist_keypoint_name=['RIGHT_WRIST'],
              spine_keypoint_name=['SPINE'],
              pelvis_keypoint_name=['PELVIS'],
              left_hip_keypoint_name=['LEFT_HIP'],
              right_hip_keypoint_name=['RIGHT_HIP'],
              left_knee_keypoint_name=['LEFT_KNEE'],
              right_knee_keypoint_name=['RIGHT_KNEE'],
              left_ankle_keypoint_name=['LEFT_ANKLE'],
              right_ankle_keypoint_name=['RIGHT_ANKLE'])


class Std13KeypointProfile3D(KeypointProfile3D):
  """Standard 3D 13-keypoint profile."""

  def __init__(self):
    """Initializer."""
    super(Std13KeypointProfile3D, self).__init__(
        name='3DSTD13',
        keypoint_names=[('HEAD', LeftRightType.CENTRAL),
                        ('LEFT_SHOULDER', LeftRightType.LEFT),
                        ('RIGHT_SHOULDER', LeftRightType.RIGHT),
                        ('LEFT_ELBOW', LeftRightType.LEFT),
                        ('RIGHT_ELBOW', LeftRightType.RIGHT),
                        ('LEFT_WRIST', LeftRightType.LEFT),
                        ('RIGHT_WRIST', LeftRightType.RIGHT),
                        ('LEFT_HIP', LeftRightType.LEFT),
                        ('RIGHT_HIP', LeftRightType.RIGHT),
                        ('LEFT_KNEE', LeftRightType.LEFT),
                        ('RIGHT_KNEE', LeftRightType.RIGHT),
                        ('LEFT_ANKLE', LeftRightType.LEFT),
                        ('RIGHT_ANKLE', LeftRightType.RIGHT)],
        offset_keypoint_names=['LEFT_HIP', 'RIGHT_HIP'],
        scale_keypoint_name_pairs=[(['LEFT_SHOULDER', 'RIGHT_SHOULDER'],
                                    ['LEFT_HIP', 'RIGHT_HIP'])],
        segment_name_pairs=[
            (['HEAD'], ['LEFT_SHOULDER', 'RIGHT_SHOULDER']),
            (['LEFT_SHOULDER', 'RIGHT_SHOULDER'], ['LEFT_SHOULDER']),
            (['LEFT_SHOULDER', 'RIGHT_SHOULDER'], ['RIGHT_SHOULDER']),
            (['LEFT_SHOULDER', 'RIGHT_SHOULDER'],
             ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']),
            (['LEFT_SHOULDER'], ['LEFT_ELBOW']),
            (['RIGHT_SHOULDER'], ['RIGHT_ELBOW']),
            (['LEFT_ELBOW'], ['LEFT_WRIST']),
            (['RIGHT_ELBOW'], ['RIGHT_WRIST']),
            (['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP',
              'RIGHT_HIP'], ['LEFT_HIP', 'RIGHT_HIP']),
            (['LEFT_HIP', 'RIGHT_HIP'], ['LEFT_HIP']),
            (['LEFT_HIP', 'RIGHT_HIP'], ['RIGHT_HIP']),
            (['LEFT_HIP'], ['LEFT_KNEE']), (['RIGHT_HIP'], ['RIGHT_KNEE']),
            (['LEFT_KNEE'], ['LEFT_ANKLE']), (['RIGHT_KNEE'], ['RIGHT_ANKLE'])
        ],
        head_keypoint_name=['HEAD'],
        neck_keypoint_name=['LEFT_SHOULDER', 'RIGHT_SHOULDER'],
        left_shoulder_keypoint_name=['LEFT_SHOULDER'],
        right_shoulder_keypoint_name=['RIGHT_SHOULDER'],
        left_elbow_keypoint_name=['LEFT_ELBOW'],
        right_elbow_keypoint_name=['RIGHT_ELBOW'],
        left_wrist_keypoint_name=['LEFT_WRIST'],
        right_wrist_keypoint_name=['RIGHT_WRIST'],
        spine_keypoint_name=[
            'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP'
        ],
        pelvis_keypoint_name=['LEFT_HIP', 'RIGHT_HIP'],
        left_hip_keypoint_name=['LEFT_HIP'],
        right_hip_keypoint_name=['RIGHT_HIP'],
        left_knee_keypoint_name=['LEFT_KNEE'],
        right_knee_keypoint_name=['RIGHT_KNEE'],
        left_ankle_keypoint_name=['LEFT_ANKLE'],
        right_ankle_keypoint_name=['RIGHT_ANKLE'])


class LegacyH36m17KeypointProfile3D(KeypointProfile3D):
  """Legacy Human3.6M 3D 17-keypoint profile."""

  def __init__(self):
    """Initializer."""
    super(LegacyH36m17KeypointProfile3D, self).__init__(
        name='LEGACY_3DH36M17',
        keypoint_names=[('Hip', LeftRightType.CENTRAL),
                        ('Head', LeftRightType.CENTRAL),
                        ('Neck/Nose', LeftRightType.CENTRAL),
                        ('Thorax', LeftRightType.CENTRAL),
                        ('LShoulder', LeftRightType.LEFT),
                        ('RShoulder', LeftRightType.RIGHT),
                        ('LElbow', LeftRightType.LEFT),
                        ('RElbow', LeftRightType.RIGHT),
                        ('LWrist', LeftRightType.LEFT),
                        ('RWrist', LeftRightType.RIGHT),
                        ('Spine', LeftRightType.CENTRAL),
                        ('LHip', LeftRightType.LEFT),
                        ('RHip', LeftRightType.RIGHT),
                        ('LKnee', LeftRightType.LEFT),
                        ('RKnee', LeftRightType.RIGHT),
                        ('LFoot', LeftRightType.LEFT),
                        ('RFoot', LeftRightType.RIGHT)],
        offset_keypoint_names=['Hip'],
        scale_keypoint_name_pairs=[(['Hip'], ['Spine']),
                                   (['Spine'], ['Thorax'])],
        segment_name_pairs=[(['Hip'], ['Spine']), (['Hip'], ['LHip']),
                            (['Hip'], ['RHip']), (['Spine'], ['Thorax']),
                            (['LHip'], ['LKnee']), (['RHip'], ['RKnee']),
                            (['LKnee'], ['LFoot']), (['RKnee'], ['RFoot']),
                            (['Thorax'], ['Neck/Nose']),
                            (['Thorax'], ['LShoulder']),
                            (['Thorax'], ['RShoulder']),
                            (['Neck/Nose'], ['Head']),
                            (['LShoulder'], ['LElbow']),
                            (['RShoulder'], ['RElbow']),
                            (['LElbow'], ['LWrist']), (['RElbow'], ['RWrist'])],
        head_keypoint_name=['Head'],
        neck_keypoint_name=['Thorax'],
        left_shoulder_keypoint_name=['LShoulder'],
        right_shoulder_keypoint_name=['RShoulder'],
        left_elbow_keypoint_name=['LElbow'],
        right_elbow_keypoint_name=['RElbow'],
        left_wrist_keypoint_name=['LWrist'],
        right_wrist_keypoint_name=['RWrist'],
        spine_keypoint_name=['Spine'],
        pelvis_keypoint_name=['Hip'],
        left_hip_keypoint_name=['LHip'],
        right_hip_keypoint_name=['RHip'],
        left_knee_keypoint_name=['LKnee'],
        right_knee_keypoint_name=['RKnee'],
        left_ankle_keypoint_name=['LFoot'],
        right_ankle_keypoint_name=['RFoot'])


class LegacyH36m13KeypointProfile3D(KeypointProfile3D):
  """Legacy Human3.6M 3D 13-keypoint profile."""

  def __init__(self):
    """Initializer."""
    super(LegacyH36m13KeypointProfile3D, self).__init__(
        name='LEGACY_3DH36M13',
        keypoint_names=[('Head', LeftRightType.CENTRAL),
                        ('LShoulder', LeftRightType.LEFT),
                        ('RShoulder', LeftRightType.RIGHT),
                        ('LElbow', LeftRightType.LEFT),
                        ('RElbow', LeftRightType.RIGHT),
                        ('LWrist', LeftRightType.LEFT),
                        ('RWrist', LeftRightType.RIGHT),
                        ('LHip', LeftRightType.LEFT),
                        ('RHip', LeftRightType.RIGHT),
                        ('LKnee', LeftRightType.LEFT),
                        ('RKnee', LeftRightType.RIGHT),
                        ('LFoot', LeftRightType.LEFT),
                        ('RFoot', LeftRightType.RIGHT)],
        offset_keypoint_names=['LHip'],
        scale_keypoint_name_pairs=[
            (['LHip', 'RHip'], ['LShoulder', 'RShoulder']),
        ],
        segment_name_pairs=[(['LHip', 'RHip'], ['LShoulder', 'RShoulder']),
                            (['LHip', 'RHip'], ['LHip']),
                            (['LHip', 'RHip'], ['RHip']), (['LHip'], ['LKnee']),
                            (['RHip'], ['RKnee']), (['LKnee'], ['LFoot']),
                            (['RKnee'], ['RFoot']),
                            (['LShoulder', 'RShoulder'], ['Head']),
                            (['LShoulder', 'RShoulder'], ['LShoulder']),
                            (['LShoulder', 'RShoulder'], ['RShoulder']),
                            (['LShoulder'], ['LElbow']),
                            (['RShoulder'], ['RElbow']),
                            (['LElbow'], ['LWrist']), (['RElbow'], ['RWrist'])],
        head_keypoint_name=['Head'],
        neck_keypoint_name=['LShoulder', 'RShoulder'],
        left_shoulder_keypoint_name=['LShoulder'],
        right_shoulder_keypoint_name=['RShoulder'],
        left_elbow_keypoint_name=['LElbow'],
        right_elbow_keypoint_name=['RElbow'],
        left_wrist_keypoint_name=['LWrist'],
        right_wrist_keypoint_name=['RWrist'],
        spine_keypoint_name=['LShoulder', 'RShoulder', 'LHip', 'RHip'],
        pelvis_keypoint_name=['LHip', 'RHip'],
        left_hip_keypoint_name=['LHip'],
        right_hip_keypoint_name=['RHip'],
        left_knee_keypoint_name=['LKnee'],
        right_knee_keypoint_name=['RKnee'],
        left_ankle_keypoint_name=['LFoot'],
        right_ankle_keypoint_name=['RFoot'])


class LegacyMpii3dhp17KeypointProfile3D(KeypointProfile3D):
  """Legacy MPII-3DHP 3D 17-keypoint profile."""

  def __init__(self):
    """Initializer."""
    super(LegacyMpii3dhp17KeypointProfile3D, self).__init__(
        name='LEGACY_3DMPII3DHP17',
        keypoint_names=[('pelvis', LeftRightType.CENTRAL),
                        ('head', LeftRightType.CENTRAL),
                        ('neck', LeftRightType.CENTRAL),
                        ('head_top', LeftRightType.CENTRAL),
                        ('left_shoulder', LeftRightType.LEFT),
                        ('right_shoulder', LeftRightType.RIGHT),
                        ('left_elbow', LeftRightType.LEFT),
                        ('right_elbow', LeftRightType.RIGHT),
                        ('left_wrist', LeftRightType.LEFT),
                        ('right_wrist', LeftRightType.RIGHT),
                        ('spine', LeftRightType.CENTRAL),
                        ('left_hip', LeftRightType.LEFT),
                        ('right_hip', LeftRightType.RIGHT),
                        ('left_knee', LeftRightType.LEFT),
                        ('right_knee', LeftRightType.RIGHT),
                        ('left_ankle', LeftRightType.LEFT),
                        ('right_ankle', LeftRightType.RIGHT)],
        offset_keypoint_names=['pelvis'],
        scale_keypoint_name_pairs=[(['pelvis'], ['spine']),
                                   (['spine'], ['neck'])],
        segment_name_pairs=[(['pelvis'], ['spine']), (['pelvis'], ['left_hip']),
                            (['pelvis'], ['right_hip']), (['spine'], ['neck']),
                            (['left_hip'], ['left_knee']),
                            (['right_hip'], ['right_knee']),
                            (['left_knee'], ['left_ankle']),
                            (['right_knee'], ['right_ankle']),
                            (['neck'], ['head']), (['neck'], ['left_shoulder']),
                            (['neck'], ['right_shoulder']),
                            (['head'], ['head_top']),
                            (['left_shoulder'], ['left_elbow']),
                            (['right_shoulder'], ['right_elbow']),
                            (['left_elbow'], ['left_wrist']),
                            (['right_elbow'], ['right_wrist'])],
        head_keypoint_name=['head'],
        neck_keypoint_name=['neck'],
        left_shoulder_keypoint_name=['left_shoulder'],
        right_shoulder_keypoint_name=['right_shoulder'],
        left_elbow_keypoint_name=['left_elbow'],
        right_elbow_keypoint_name=['right_elbow'],
        left_wrist_keypoint_name=['left_wrist'],
        right_wrist_keypoint_name=['right_wrist'],
        spine_keypoint_name=['spine'],
        pelvis_keypoint_name=['pelvis'],
        left_hip_keypoint_name=['left_hip'],
        right_hip_keypoint_name=['right_hip'],
        left_knee_keypoint_name=['left_knee'],
        right_knee_keypoint_name=['right_knee'],
        left_ankle_keypoint_name=['left_ankle'],
        right_ankle_keypoint_name=['right_ankle'])


class Std13KeypointProfile2D(KeypointProfile2D):
  """Standard 2D 13-keypoint profile."""

  def __init__(self):
    """Initializer."""
    super(Std13KeypointProfile2D, self).__init__(
        name='2DSTD13',
        keypoint_names=[('NOSE_TIP', LeftRightType.CENTRAL),
                        ('LEFT_SHOULDER', LeftRightType.LEFT),
                        ('RIGHT_SHOULDER', LeftRightType.RIGHT),
                        ('LEFT_ELBOW', LeftRightType.LEFT),
                        ('RIGHT_ELBOW', LeftRightType.RIGHT),
                        ('LEFT_WRIST', LeftRightType.LEFT),
                        ('RIGHT_WRIST', LeftRightType.RIGHT),
                        ('LEFT_HIP', LeftRightType.LEFT),
                        ('RIGHT_HIP', LeftRightType.RIGHT),
                        ('LEFT_KNEE', LeftRightType.LEFT),
                        ('RIGHT_KNEE', LeftRightType.RIGHT),
                        ('LEFT_ANKLE', LeftRightType.LEFT),
                        ('RIGHT_ANKLE', LeftRightType.RIGHT)],
        offset_keypoint_names=['LEFT_HIP', 'RIGHT_HIP'],
        scale_keypoint_name_pairs=[(['LEFT_SHOULDER'], ['RIGHT_SHOULDER']),
                                   (['LEFT_SHOULDER'], ['LEFT_HIP']),
                                   (['LEFT_SHOULDER'], ['RIGHT_HIP']),
                                   (['RIGHT_SHOULDER'], ['LEFT_HIP']),
                                   (['RIGHT_SHOULDER'], ['RIGHT_HIP']),
                                   (['LEFT_HIP'], ['RIGHT_HIP'])],
        segment_name_pairs=[(['NOSE_TIP'], ['LEFT_SHOULDER']),
                            (['NOSE_TIP'], ['RIGHT_SHOULDER']),
                            (['LEFT_SHOULDER'], ['RIGHT_SHOULDER']),
                            (['LEFT_SHOULDER'], ['LEFT_ELBOW']),
                            (['RIGHT_SHOULDER'], ['RIGHT_ELBOW']),
                            (['LEFT_ELBOW'], ['LEFT_WRIST']),
                            (['RIGHT_ELBOW'], ['RIGHT_WRIST']),
                            (['LEFT_SHOULDER'], ['LEFT_HIP']),
                            (['RIGHT_SHOULDER'], ['RIGHT_HIP']),
                            (['LEFT_HIP'], ['RIGHT_HIP']),
                            (['LEFT_HIP'], ['LEFT_KNEE']),
                            (['RIGHT_HIP'], ['RIGHT_KNEE']),
                            (['LEFT_KNEE'], ['LEFT_ANKLE']),
                            (['RIGHT_KNEE'], ['RIGHT_ANKLE'])],
        compatible_keypoint_name_dict={
            '3DSTD16': [
                'HEAD', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
                'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP',
                'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                'RIGHT_ANKLE'
            ],
            '3DSTD13': [
                'HEAD', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
                'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP',
                'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                'RIGHT_ANKLE'
            ],
            'LEGACY_3DH36M17': [
                'Head', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist',
                'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LFoot', 'RFoot'
            ],
            'LEGACY_3DMPII3DHP17': [
                'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
                'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
                'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                'right_ankle'
            ],
        },
        head_keypoint_name=['NOSE_TIP'],
        neck_keypoint_name=['LEFT_SHOULDER', 'RIGHT_SHOULDER'],
        left_shoulder_keypoint_name=['LEFT_SHOULDER'],
        right_shoulder_keypoint_name=['RIGHT_SHOULDER'],
        left_elbow_keypoint_name=['LEFT_ELBOW'],
        right_elbow_keypoint_name=['RIGHT_ELBOW'],
        left_wrist_keypoint_name=['LEFT_WRIST'],
        right_wrist_keypoint_name=['RIGHT_WRIST'],
        spine_keypoint_name=[
            'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP'
        ],
        pelvis_keypoint_name=['LEFT_HIP', 'RIGHT_HIP'],
        left_hip_keypoint_name=['LEFT_HIP'],
        right_hip_keypoint_name=['RIGHT_HIP'],
        left_knee_keypoint_name=['LEFT_KNEE'],
        right_knee_keypoint_name=['RIGHT_KNEE'],
        left_ankle_keypoint_name=['LEFT_ANKLE'],
        right_ankle_keypoint_name=['RIGHT_ANKLE'])


class LegacyCoco13KeypointProfile2D(Std13KeypointProfile2D):
  """Legacy COCO 2D 13-keypoint profile.

  This profile is the same as the `2DSTD13` profil, except the name.
  """

  def __init__(self):
    """Initializer."""
    super(LegacyCoco13KeypointProfile2D, self).__init__()
    self._name = 'LEGACY_2DCOCO13'


class LegacyH36m13KeypointProfile2D(KeypointProfile2D):
  """Legacy Human3.6M 2D 13-keypoint profile."""

  def __init__(self):
    """Initializer."""
    super(LegacyH36m13KeypointProfile2D,
          self).__init__(
              name='LEGACY_2DH36M13',
              keypoint_names=[('Head', LeftRightType.CENTRAL),
                              ('LShoulder', LeftRightType.LEFT),
                              ('RShoulder', LeftRightType.RIGHT),
                              ('LElbow', LeftRightType.LEFT),
                              ('RElbow', LeftRightType.RIGHT),
                              ('LWrist', LeftRightType.LEFT),
                              ('RWrist', LeftRightType.RIGHT),
                              ('LHip', LeftRightType.LEFT),
                              ('RHip', LeftRightType.RIGHT),
                              ('LKnee', LeftRightType.LEFT),
                              ('RKnee', LeftRightType.RIGHT),
                              ('LFoot', LeftRightType.LEFT),
                              ('RFoot', LeftRightType.RIGHT)],
              offset_keypoint_names=['LHip', 'RHip'],
              scale_keypoint_name_pairs=[(['LShoulder'], ['RShoulder']),
                                         (['LShoulder'], ['LHip']),
                                         (['LShoulder'], ['RHip']),
                                         (['RShoulder'], ['LHip']),
                                         (['RShoulder'], ['RHip']),
                                         (['LHip'], ['RHip'])],
              segment_name_pairs=[(['Head'], ['LShoulder']),
                                  (['Head'], ['RShoulder']),
                                  (['LShoulder'], ['LElbow']),
                                  (['LElbow'], ['LWrist']),
                                  (['RShoulder'], ['RElbow']),
                                  (['RElbow'], ['RWrist']),
                                  (['LShoulder'], ['LHip']),
                                  (['RShoulder'], ['RHip']),
                                  (['LHip'], ['LKnee']), (['LKnee'], ['LFoot']),
                                  (['RHip'], ['RKnee']), (['RKnee'], ['RFoot']),
                                  (['LShoulder'], ['RShoulder']),
                                  (['LHip'], ['RHip'])],
              compatible_keypoint_name_dict={
                  '3DSTD16': [
                      'HEAD', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
                      'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP',
                      'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                      'RIGHT_ANKLE'
                  ],
                  '3DSTD13': [
                      'HEAD', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
                      'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP',
                      'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                      'RIGHT_ANKLE'
                  ],
                  'LEGACY_3DH36M17': [
                      'Head', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
                      'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee',
                      'LFoot', 'RFoot'
                  ],
                  'LEGACY_3DMPII3DHP17': [
                      'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
                      'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
                      'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                      'right_ankle'
                  ],
              },
              head_keypoint_name=['Head'],
              neck_keypoint_name=['LShoulder', 'RShoulder'],
              left_shoulder_keypoint_name=['LShoulder'],
              right_shoulder_keypoint_name=['RShoulder'],
              left_elbow_keypoint_name=['LElbow'],
              right_elbow_keypoint_name=['RElbow'],
              left_wrist_keypoint_name=['LWrist'],
              right_wrist_keypoint_name=['RWrist'],
              spine_keypoint_name=['LShoulder', 'RShoulder', 'LHip', 'RHip'],
              pelvis_keypoint_name=['LHip', 'RHip'],
              left_hip_keypoint_name=['LHip'],
              right_hip_keypoint_name=['RHip'],
              left_knee_keypoint_name=['LKnee'],
              right_knee_keypoint_name=['RKnee'],
              left_ankle_keypoint_name=['LFoot'],
              right_ankle_keypoint_name=['RFoot'])


def create_keypoint_profile_or_die(keypoint_profile_name):
  """Creates keypoint profile based on name.

  Args:
    keypoint_profile_name: A string for keypoint profile name.

  Returns:
    A keypint profile class object.

  Raises:
    ValueError: If keypoint profile name is unsupported.
  """
  if keypoint_profile_name == '3DSTD16':
    return Std16KeypointProfile3D()
  if keypoint_profile_name == '3DSTD13':
    return Std13KeypointProfile3D()
  if keypoint_profile_name == 'LEGACY_3DH36M17':
    return LegacyH36m17KeypointProfile3D()
  if keypoint_profile_name == 'LEGACY_3DH36M13':
    return LegacyH36m13KeypointProfile3D()
  if keypoint_profile_name == 'LEGACY_3DMPII3DHP17':
    return LegacyMpii3dhp17KeypointProfile3D()
  if keypoint_profile_name == '2DSTD13':
    return Std13KeypointProfile2D()
  if keypoint_profile_name == 'LEGACY_2DCOCO13':
    return LegacyCoco13KeypointProfile2D()
  if keypoint_profile_name == 'LEGACY_2DH36M13':
    return LegacyH36m13KeypointProfile2D()

  raise ValueError('Unsupported keypoint profile name: `%s`.' %
                   str(keypoint_profile_name))
