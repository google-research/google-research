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

"""RIO (tfds) feature specifications."""

import tensorflow as tf
import tensorflow_datasets as tfds


def mesh_feature_spec(with_annotations=True):
  """Feature specification of mesh data."""
  feature_spec = {
      # vertex data of N points.
      'vertices': {
          # Vertex positions (Nx3).
          'positions': tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
          # Vertex normals (Nx3).
          'normals': tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
      },
      # face data of F faces.
      'faces': {
          # Face polygons (Fx3).
          'polygons': tfds.features.Tensor(shape=(None, 3), dtype=tf.int32),
      },
  }
  if with_annotations:
    feature_spec['vertices'].update({
        # Vertex semantic labels (Nx1).
        'semantic_labels':
            tfds.features.Tensor(shape=(None, 1), dtype=tf.int64),
        # Vertex instance labels (Nx1).
        'instance_labels':
            tfds.features.Tensor(shape=(None, 1), dtype=tf.int64),
    })
  return tfds.features.FeaturesDict(feature_spec)


def scene_feature_spec(with_annotations=True):
  """Feature specification of scene data.

  Each scene data stores reconstructed 3D mesh of the whole scene.

  Args:
    with_annotations: If true semantic and instance labels for mesh vertices are
      also present. This is the default (True) for training data.

  Returns:
    Feature specification (tfds) for a single scene data.
  """
  return tfds.features.FeaturesDict({
      # A unique name that identifies the scene.
      'scene_name': tfds.features.Text(),
      # Reference name of the initial scan.
      'reference_name': tfds.features.Text(),
      # Scene mesh data.
      'mesh': mesh_feature_spec(with_annotations)
  })
