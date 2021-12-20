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

"""Loads the SYMSOL dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.spatial.transform import Rotation
import tensorflow as tf
import tensorflow_datasets as tfds


SHAPE_NAMES = [
    'tet', 'cube', 'icosa', 'cone', 'cyl', 'tetX', 'cylO', 'sphereX'
]


def load_symsol(shapes, mode='train', downsample_continuous_gt=0, mock=False):
  """Loads the symmetric_solids dataset.

  Args:
    shapes: Can be 'symsol1' or any subset from the 8 shapes in SHAPE_NAMES.
    mode: 'train' or 'test', determining the split of the dataset.
    downsample_continuous_gt: An integer, the amount to downsample the
      continuous symmetry ground truths, if any.  The gt rotations for the cone
      and cyl have been discretized to 1 degree increments, but this can be
      overkill for evaluation during training. If 0, use the full annotation.
    mock: Make random data to avoid downloading it.

  Returns:
    tf.data.Dataset of images with the associated rotation matrices.
  """
  shape_inds = [SHAPE_NAMES.index(shape) for shape in shapes]
  dataset_loaded = False
  if not dataset_loaded:
    if mock:
      with tfds.testing.mock_data(num_examples=100):
        dataset = tfds.load('symmetric_solids', split=mode)
    else:
      dataset = tfds.load('symmetric_solids', split=mode)

  # Filter the dataset by shape index, and use the full set of equivalent
  # rotations only if mode == test
  dataset = dataset.filter(
      lambda x: tf.reduce_any(tf.equal(x['label_shape'], shape_inds)))

  annotation_key = 'rotation' if mode == 'train' else 'rotations_equivalent'

  dataset = dataset.map(
      lambda example: (example['image'], example[annotation_key]),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)


  dataset = dataset.map(
      lambda im, rots: (tf.image.convert_image_dtype(im, tf.float32), rots),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if mode == 'test' and downsample_continuous_gt and shape_inds[0] in [3, 4]:
    # Downsample the full set of equivalent rotations for the cone and cyl.
    dataset = dataset.map(
        lambda im, rots: (im, rots[::downsample_continuous_gt]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return dataset


def compute_symsol_symmetries(num_steps_around_continuous=360):
  """Return the GT rotation matrices for the symmetric solids.

  We provide this primarily for the ability to generate the symmetry rotations
  for the cone and cylinder at arbitrary resolutions.

  The first matrix returned for each is the identity.

  Args:
    num_steps_around_continuous: The number of steps taken around each great
    circle of equivalent poses for the cylinder and cone.

  Returns:
    A dictionary, indexed by shape name, for the five solids of the SYMSOL
    dataset.  The values in the dictionary are [N, 3, 3] rotation matrices,
    where N is 12 for tet, 24 for cube, 60 for icosa,
    num_steps_around_continuous for cone, and 2*num_steps_around_continuous for
    cyl.
  """
  # Tetrahedron
  tet_seeds = [np.eye(3)]
  for i in range(3):
    tet_seeds.append(np.diag(np.roll([-1, -1, 1], i)))
  tet_syms = []
  for rotation_matrix in tet_seeds:
    tet_syms.append(rotation_matrix)
    tet_syms.append(np.roll(rotation_matrix, 1, axis=0))
    tet_syms.append(np.roll(rotation_matrix, -1, axis=0))

  tet_syms = np.stack(tet_syms, 0)
  # The syms are specific to the object coordinate axes used during rendering,
  # and for the tet the canonical frames were 45 deg from corners of a cube
  correction_rot = Rotation.from_euler('xyz',
                                       np.float32([0, 0, np.pi / 4.0])).as_dcm()
  # So we rotate to the cube frame, where the computed syms (above) are valid
  # and then rotate back
  tet_syms = correction_rot @ tet_syms @ correction_rot.T

  # Cube
  cube_seeds = [np.eye(3)]
  cube_seeds.append(np.float32([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]))
  for i in range(3):
    cube_seeds.append(np.diag(np.roll([-1, -1, 1], i)))
    cube_seeds.append(np.diag(np.roll([-1, 1, 1], i)) @ np.float32([[0, 0, 1],
                                                                    [0, 1, 0],
                                                                    [1, 0, 0]]))
  cube_syms = []
  for rotation_matrix in cube_seeds:
    cube_syms.append(rotation_matrix)
    cube_syms.append(np.roll(rotation_matrix, 1, axis=0))
    cube_syms.append(np.roll(rotation_matrix, -1, axis=0))
  cube_syms = np.stack(cube_syms, 0)

  # Icosahedron
  golden_ratio = (1 + np.sqrt(5)) / 2.
  a, b = np.float32([1, golden_ratio]) / np.sqrt(1 + golden_ratio**2)
  icosa_verts = np.float32([[-a, b, 0],
                            [a, b, 0],
                            [-a, -b, 0],
                            [a, -b, 0],
                            [0, -a, b],
                            [0, a, b],
                            [0, -a, -b],
                            [0, a, -b],
                            [b, 0, -a],
                            [b, 0, a],
                            [-b, 0, -a],
                            [-b, 0, a]])
  icosa_syms = [np.eye(3)]
  for ind1 in range(12):
    for ind2 in range(ind1+1, 12):
      icosa_vert1 = icosa_verts[ind1]
      icosa_vert2 = icosa_verts[ind2]
      if np.abs(np.dot(icosa_vert1, icosa_vert2)) == 1:
        continue
      for angle1 in np.arange(3) * 2 * np.pi / 5:
        for angle2 in np.arange(1, 3) * 2 * np.pi / 5:
          rot = Rotation.from_rotvec(
              angle1 * icosa_vert1).as_dcm() @ Rotation.from_rotvec(
                  angle2 * icosa_vert2).as_dcm()
          icosa_syms.append(rot)

  # Remove duplicates
  icosa_syms = np.stack(icosa_syms, 0)
  trs = np.trace((icosa_syms[np.newaxis] @ np.transpose(
      icosa_syms, [0, 2, 1])[:, np.newaxis]),
                 axis1=2,
                 axis2=3)
  good_inds = []
  bad_inds = []
  eps = 1e-9
  for i in range(icosa_syms.shape[0]):
    if i not in bad_inds:
      good_inds.append(i)
    dups = np.where(trs[i, :] > (3 - eps))
    _ = [bad_inds.append(j) for j in dups[0]]
  icosa_syms = icosa_syms[good_inds]

  # Cone
  cone_syms = []
  for sym_val in np.linspace(0, 2*np.pi, num_steps_around_continuous):
    sym_rot = Rotation.from_euler('xyz', np.float32([0, 0, sym_val])).as_dcm()
    cone_syms.append(sym_rot)
  cone_syms = np.stack(cone_syms, 0)

  # Cylinder
  cyl_syms = []
  for sym_val in np.linspace(0, 2*np.pi, num_steps_around_continuous):
    for x_rot in [0., np.pi]:
      sym_rot = Rotation.from_euler('xyz', np.float32([x_rot, 0,
                                                       sym_val])).as_dcm()
      cyl_syms.append(sym_rot)
  cyl_syms = np.stack(cyl_syms, 0)

  return dict(tet=tet_syms,
              cube=cube_syms,
              icosa=icosa_syms,
              cyl=cyl_syms,
              cone=cone_syms)

