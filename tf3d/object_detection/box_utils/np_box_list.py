# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Numpy 3D BoxList classes and functions."""

import logging
import numpy as np


class BoxList3d(object):
  """Box collection.

  BoxList represents a list of bounding boxes. Additional fields can be added to
  BoxList3d (such as objectness/classification scores).
  """

  def __init__(self,
               length,
               height,
               width,
               center_x,
               center_y,
               center_z,
               rotation_matrix=None,
               rotation_z_radians=None):
    """Constructs box collection.

    Args:
      length: A numpy array of shape [N].
      height: A numpy array of shape [N].
      width: A numpy array of shape [N].
      center_x: A numpy array of shape [N].
      center_y: A numpy array of shape [N].
      center_z: A numpy array of shape [N].
      rotation_matrix: A numpy array of shape [N, 3, 3].
      rotation_z_radians: A numpy array of shape [N] or None.

    Raises:
      ValueError: If arguments are not of the right size or type.
      ValueError: If none of the rotation_matrix, rotation_y_radians is set.
    """
    self.data = {}

    self.data['length'] = length
    if not isinstance(length, np.ndarray):
      raise ValueError('Length must be a numpy array.')
    if len(length.shape) != 1:
      raise ValueError('Invalid dimensions for length.')

    self.data['height'] = height
    if not isinstance(height, np.ndarray):
      raise ValueError('Height must be a numpy array.')
    if len(height.shape) != 1:
      raise ValueError('Invalid dimensions for height.')

    self.data['width'] = width
    if not isinstance(width, np.ndarray):
      raise ValueError('Width must be a numpy array.')
    if len(width.shape) != 1:
      raise ValueError('Invalid dimensions for width.')

    self.data['center_x'] = center_x
    if not isinstance(center_x, np.ndarray):
      raise ValueError('center_x must be a numpy array.')
    if len(center_x.shape) != 1:
      raise ValueError('Invalid dimensions for center_x.')

    self.data['center_y'] = center_y
    if not isinstance(center_y, np.ndarray):
      raise ValueError('center_y must be a numpy array.')
    if len(center_y.shape) != 1:
      raise ValueError('Invalid dimensions for center_y.')

    self.data['center_z'] = center_z
    if not isinstance(center_z, np.ndarray):
      raise ValueError('center_z must be a numpy array.')
    if len(center_z.shape) != 1:
      raise ValueError('Invalid dimensions for center_z.')

    if rotation_matrix is None and rotation_z_radians is None:
      raise ValueError('Rotation should be set.')
    self.data['rotation_matrix'] = rotation_matrix
    self.data['rotation_z_radians'] = rotation_z_radians
    if rotation_matrix is not None:
      if not isinstance(rotation_matrix, np.ndarray):
        raise ValueError('rotation_matrix must be a numpy array.')
      if len(rotation_matrix.shape) != 3:
        raise ValueError('Invalid dimensions for rotation_matrix.')
      if rotation_matrix.shape[1] != 3 or rotation_matrix.shape[2] != 3:
        raise ValueError('rotation_matrix should be of size [n, 3, 3].')
    if rotation_z_radians is not None:
      if not isinstance(rotation_z_radians, np.ndarray):
        raise ValueError('rotation_z_radians must be a numpy array.')
      if len(rotation_z_radians.shape) != 1:
        raise ValueError('Invalid dimensions for rotation_z_radians.')
    if not self._is_valid_boxes():
      raise ValueError('Invalid box format.')

  def _is_valid_boxes(self):
    """Checks whether box format fullfills the format.

    Returns:
      A boolean indicating whether box format is valid.
    """
    n = self.data['length'].shape[0]
    if self.data['height'].shape[0] != n:
      logging.info('**********************')
      logging.info('Invalid height')
      return False
    if self.data['width'].shape[0] != n:
      logging.info('**********************')
      logging.info('Invalid width')
      return False
    if self.data['center_x'].shape[0] != n:
      logging.info('**********************')
      logging.info('Invalid center_x')
      return False
    if self.data['center_y'].shape[0] != n:
      logging.info('**********************')
      logging.info('Invalid center_y')
      return False
    if self.data['center_z'].shape[0] != n:
      logging.info('**********************')
      logging.info('Invalid center_z')
      return False
    if self.data['rotation_z_radians'] is not None:
      if self.data['rotation_z_radians'].shape[0] != n:
        logging.info('**********************')
        logging.info('Invalid rotation_z_radians')
        return False
    for i in range(n):
      if (self.data['length'][i] <= 0 or self.data['height'][i] <= 0 or
          self.data['width'][i] <= 0):
        logging.info('**********************')
        logging.info('Negative length, height or width.')
        return False
    return True

  def num_boxes(self):
    """Return number of boxes held in collections."""
    return self.get_length().shape[0]

  def has_field(self, field):
    return field in self.data

  def add_field(self, field, field_data):
    """Add data to a specified field.

    Args:
      field: a string parameter used to speficy a related field to be accessed.
      field_data: a numpy array of [N, ...] representing the data associated
          with the field.
    Raises:
      ValueError: if the field is already exist or the dimension of the field
          data does not matches the number of boxes.
    """
    if self.has_field(field):
      raise ValueError('Field ' + field + 'already exists')
    if len(field_data.shape) < 1 or field_data.shape[0] != self.num_boxes():
      raise ValueError('Invalid dimensions for field data')
    self.data[field] = field_data

  def get_field(self, field):
    """Accesses data associated with the specified field in the box collection.

    Args:
      field: a string parameter used to specify a related field to be accessed.
    Returns:
      a numpy 1-d array representing data of an associated field
    Raises:
      ValueError: if invalid field
    """
    if not self.has_field(field):
      raise ValueError('field {} does not exist'.format(field))
    return self.data[field]

  def get(self):
    raise ValueError('Get function should not be called for this class.')

  def get_length(self):
    return self.get_field('length')

  def get_height(self):
    return self.get_field('height')

  def get_width(self):
    return self.get_field('width')

  def get_center_x(self):
    return self.get_field('center_x')

  def get_center_y(self):
    return self.get_field('center_y')

  def get_center_z(self):
    return self.get_field('center_z')

  def get_center(self):
    return np.stack(
        [self.get_center_x(),
         self.get_center_y(),
         self.get_center_z()], axis=1)

  def get_rotation_matrix(self):
    return self.get_field('rotation_matrix')

  def get_rotation_z_radians(self):
    return self.get_field('rotation_z_radians')

  def get_extra_fields(self):
    """Return all non-box fields."""
    field_list = [
        'length', 'height', 'width', 'center_x', 'center_y', 'center_z',
        'rotation_matrix', 'rotation_z_radians'
    ]
    return [k for k in self.data.keys() if k not in field_list]


def boxlist_3d_from_7param_boxes(boxes):
  """Creates and returns a BoxList3d class from the 7 parameter boxes.

  Args:
    boxes: A np.float32 array of size [N, 7]. Each row contains the following
      entries [ry, length, width, height, center_x, center_y, center_z].

  Returns:
    A BoxList3d object.
  """
  rotation_z_radians = boxes[:, 0]
  length = boxes[:, 1]
  width = boxes[:, 2]
  height = boxes[:, 3]
  center_x = boxes[:, 4]
  center_y = boxes[:, 5]
  center_z = boxes[:, 6]
  return BoxList3d(
      length=length,
      height=height,
      width=width,
      center_x=center_x,
      center_y=center_y,
      center_z=center_z,
      rotation_z_radians=rotation_z_radians)


def boxlist_3d_to_7param_boxes(boxlist):
  """Creates and returns a BoxList3d class from the 7 parameter boxes.

  Args:
    boxlist: A BoxList3d object.

  Returns:
    boxes: A np.float32 array of size [N, 7]. Each row contains the following
      entries [ry, length, width, height, center_x, center_y, center_z].
  """
  return np.stack([
      boxlist.get_rotation_z_radians(),
      boxlist.get_length(),
      boxlist.get_width(),
      boxlist.get_height(),
      boxlist.get_center_x(),
      boxlist.get_center_y(),
      boxlist.get_center_z()
  ], axis=1)


def boxlist_3d_from_9param_boxes(boxes):
  """Creates and returns a BoxList3d class from the 9 parameter boxes.

  Args:
    boxes: A np.float32 array of size [N, 9]. Each row contains the following
      [rotation_matrix, length, width, height, center_x, center_y, center_z].

  Returns:
    A BoxList3d object.
  """
  rotation_matrix = np.reshape(boxes[:, 0:9], [-1, 3, 3])
  length = boxes[:, 9]
  width = boxes[:, 10]
  height = boxes[:, 11]
  center_x = boxes[:, 12]
  center_y = boxes[:, 13]
  center_z = boxes[:, 14]
  return BoxList3d(
      length=length,
      height=height,
      width=width,
      center_x=center_x,
      center_y=center_y,
      center_z=center_z,
      rotation_matrix=rotation_matrix)


def boxlist_3d_to_9param_boxes(boxlist):
  """Creates and returns a BoxList3d class from the 9 parameter boxes.

  Args:
    boxlist: A BoxList3d object.

  Returns:
    boxes: A np.float32 array of size [N, 9]. Each row contains the following
      entries [ry, length, width, height, center_x, center_y, center_z].
  """
  rotation_values = np.reshape(boxlist.get_rotation_matrix(), [-1, 9])
  size_center_values = np.stack([
      boxlist.get_length(),
      boxlist.get_width(),
      boxlist.get_height(),
      boxlist.get_center_x(),
      boxlist.get_center_y(),
      boxlist.get_center_z()
  ], axis=1)
  return np.concatenate([rotation_values, size_center_values], axis=1)
