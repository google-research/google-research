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

"""IO Utilities."""
import os

import numpy as np
from PIL import Image
import plyfile
from tensorflow.io import gfile


def load_masked_ply(file_name):
  """Load masked ply."""
  with gfile.GFile(file_name, 'rb') as f:
    mesh = plyfile.PlyData.read(f)

  vertices_x = np.asarray(mesh['vertex']['x'], dtype=np.double)
  vertices_y = np.asarray(mesh['vertex']['y'], dtype=np.double)
  vertices_z = np.asarray(mesh['vertex']['z'], dtype=np.double)

  vertices = np.concatenate(
      (vertices_x[:, None], vertices_y[:, None], vertices_z[:, None]), axis=-1
  )

  faces = None
  if 'face' in mesh:
    faces = mesh['face']['vertex_indices']
    faces = [np.asarray(fc, dtype=np.int_) for fc in faces]
    faces = np.stack(faces, axis=-1)

  mask = np.asarray(mesh['face']['quality'], dtype=np.int_)

  return vertices, np.transpose(faces, (1, 0)), mask


def load_npz(file_name, pickle=True):
  """Load from npz."""
  direc = {}
  with gfile.GFile(file_name, 'rb') as f:
    with np.load(f, allow_pickle=pickle) as npz_archive:
      for k in npz_archive:
        direc[k] = npz_archive[k]
  return direc


def load_png(file_name):
  """Load png."""
  with gfile.GFile(file_name, 'rb') as f:
    out = np.array(Image.open(f))
  return out


def save_png(x, file_name):
  """Save png."""
  gfile.makedirs(os.path.dirname(file_name))
  with gfile.GFile(file_name, 'wb') as f:
    Image.fromarray(x).save(f)
