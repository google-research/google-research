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

"""ScanNet (tfds) feature specifications."""

import tensorflow as tf
import tensorflow_datasets as tfds

from tf3d.datasets.utils import image_features


def camera_feature_spec(
    with_annotations = True):
  """Feature specification of camera data.

  A single frame can have multiple cameras. We assume pin-hole camera model:
  x_{image} = K [R | t] X_{frame}.

  Args:
    with_annotations: If true semantic and instance label images are also
      present. This is the default (True) for training data.

  Returns:
    Feature specification (tfds) for a single camera data.
  """
  feature_spec = {
      # camera intrinsics.
      'intrinsics': {
          # camera intrinsics matrix K (3x3 matrix).
          'K': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
      },
      # camera extrinsics w.r.t frame.
      'extrinsics': {
          'R': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
          't': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
      },
      # color image data.
      'color_image': tfds.features.Image(shape=(None, None, 3)),
      # depth image data.
      'depth_image': image_features.Depth(dtype=tf.float32),
  }
  if with_annotations:
    feature_spec.update({
        # semantic segmentation label image.
        'semantic_image':
            tfds.features.Image(
                shape=(None, None, 1), dtype=tf.uint8, encoding_format='png'),
        # instance segmentation label image.
        'instance_image':
            tfds.features.Image(
                shape=(None, None, 1), dtype=tf.uint8, encoding_format='png')
    })
  return tfds.features.FeaturesDict(feature_spec)


def frame_feature_spec(with_annotations=True):
  """Frame feature specification.

  Each frame contains sensor data at particular timestamp. For ScanNet each
  frame data only has a single RGBD camera. All transformations like camera
  extrinsics are w.r.t local coordinate frame. To obtaine extrinsics w.r.t scene
  we also need to incorporate the frame pose w.r.t scene.


  Args:
    with_annotations: If true semantic and instance label images are also
      present. This is the default (True) for training data.

  Returns:
    Feature specification (tfds) for a single frame data.
  """
  # Feature specification of frame dataset.
  return tfds.features.FeaturesDict({
      # A unique name that identifies the sequence the frame is from.
      'scene_name': tfds.features.Text(),
      # A unique name that identifies this particular frame.
      'frame_name': tfds.features.Text(),
      # Frame timestamp. 0-based index of this frame among all scene frames.
      'timestamp': tf.int64,
      # Camera sensor data.
      'cameras': {
          'rgbd_camera': camera_feature_spec(with_annotations),
      },
      # Frame pose w.r.t scene: X_{scene} = R * X_{frame} + t.
      'pose': {
          'R': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
          't': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
      },
  })


def frame_info_feature_spec():
  """Feature specification of lightweight information about a single frame."""
  return tfds.features.FeaturesDict({
      # A unique name that identifies a particular frame. This can be used to
      # lookup a frame example from frame dataset.
      'frame_name': tfds.features.Text(),
      # Frame pose w.r.t scene: X_{scene} = R * X_{frame} + t.
      'pose': {
          'R': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
          't': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
      },
  })


def mesh_feature_spec(with_annotations=True):
  """Feature specification of mesh data."""
  feature_spec = {
      # vertex data of N points.
      'vertices': {
          # Vertex positions (Nx3).
          'positions': tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
          # Vertex normals (Nx3).
          'normals': tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
          # Vertex colors RGBA (Nx4).
          'colors': tfds.features.Tensor(shape=(None, 4), dtype=tf.uint8),
      },
      # face data of F faces.
      'faces': {
          # Face polygons (Fx3).
          'polygons': tfds.features.Tensor(shape=(None, 3), dtype=tf.uint32),
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
      # Scene mesh data.
      'mesh': mesh_feature_spec(with_annotations),
      # Frame information. A scene have several frames.
      'frames': tfds.features.Sequence(frame_info_feature_spec()),
  })
