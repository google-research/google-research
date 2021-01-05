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

"""Feature specification for Waymo Frame Dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds


# Composite FeatureConnetor for camera data.
CAMERA_FEATURE_SPEC = tfds.features.FeaturesDict({
    # camera id (e.g 0, 3).
    'id': tf.int64,
    # camera name (e.g. FRONT, LEFT).
    'name': tfds.features.Text(),
    # image data.
    'image': tfds.features.Image(shape=(None, None, 3), encoding_format='jpeg'),
    # camera instrinsics.
    'intrinsics': {
        # Camera intrinsics matrix K (3x3 matrix).
        'K': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
        # Distortion coeffiecients (k1, k2, p1, p2, k3).
        'distortion': tfds.features.Tensor(shape=(5,), dtype=tf.float32),
    },
    # camera extrinsics.
    'extrinsics': {
        'R': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
        't': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
    },
    # camera shutter type
    'shutter_type': tfds.features.Text(),
})


# Composite FeatureConnetor for Lidar data.
LIDAR_FEATURE_SPEC = tfds.features.FeaturesDict({
    # lidar id (e.g 1, 3).
    'id': tf.int64,
    # lidar name (e.g. TOP, REAR).
    'name': tfds.features.Text(),
    # 3D pointcloud data from the lidar with N points.
    'pointcloud': {
        # Pointcloud positions (Nx3).
        'positions': tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
        # Pointcloud intensity (Nx1).
        'intensity': tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
        # Pointcloud elongation (Nx1).
        'elongation': tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
        # Pointcloud No Label Zone information (NLZ) (Nx1).
        'inside_nlz': tfds.features.Tensor(shape=(None, 1), dtype=tf.bool),
        # Pointcloud return number (first return, second return, or third)
        # TODO(alirezafathi): Find a better name for this field.
        'return_number': tfds.features.Tensor(shape=(None, 1), dtype=tf.int32),
    },
    # lidar extrinsics.
    'extrinsics': {
        'R': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
        't': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
    },
    # lidar pointcloud to camera image correspondence for N lidar points.
    'camera_projections': {
        # Camera id each 3D point projects to (Nx1).
        'ids': tfds.features.Tensor(shape=(None, 1), dtype=tf.int64),
        # Image location (x, y) of each 3D point's projection (Nx2).
        'positions': tfds.features.Tensor(shape=(None, 2), dtype=tf.float32),
    }
})


# Object category labels.
OBJECT_CATEGORY_LABELS = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']


# Composite FeatureConnetor for object data.
OBJECT_FEATURE_SPEC = tfds.features.FeaturesDict({
    # object id
    'id': tf.int64,
    # object name
    'name': tfds.features.Text(),
    # object category (class).
    'category': {
        # integer label id
        'label': tfds.features.ClassLabel(names=OBJECT_CATEGORY_LABELS),
        # text label id
        'text': tfds.features.Text(),
    },
    # object shape
    'shape': {
        # object size (length, width, height) along object's (x, y, z) axes.
        'dimension': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
    },
    # object pose
    'pose': {
        'R': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
        't': tfds.features.Tensor(shape=(3,), dtype=tf.float32)
    },
    # object difficulty level
    # The higher the level, the harder it is.
    'difficulty_level': {
        'detection': tf.int64,
        'tracking': tf.int64,
    },
})


# Feature specification of frame dataset.
FRAME_FEATURE_SPEC = tfds.features.FeaturesDict({
    # A unique name that identifies the sequence the frame is from.
    'scene_name': tfds.features.Text(),
    # A unique name that identifies this particular frame.
    'frame_name': tfds.features.Text(),
    # Frame *start* time (set to the timestamp of the first top laser spin).
    'timestamp': tf.int64,
    # Day, Dawn/Dusk, or Night, determined from sun elevation.
    'time_of_day': tfds.features.Text(),
    # Human readable location (e.g. CHD, SF) of the run segment.
    'location': tfds.features.Text(),
    # Sunny or Rain.
    'weather': tfds.features.Text(),
    # Camera sensor data.
    'cameras': {
        'front': CAMERA_FEATURE_SPEC,
        'front_left': CAMERA_FEATURE_SPEC,
        'front_right': CAMERA_FEATURE_SPEC,
        'side_left': CAMERA_FEATURE_SPEC,
        'side_right': CAMERA_FEATURE_SPEC,
    },
    # lidar sensor data.
    'lidars': {
        'top': LIDAR_FEATURE_SPEC,
        'front': LIDAR_FEATURE_SPEC,
        'side_left': LIDAR_FEATURE_SPEC,
        'side_right': LIDAR_FEATURE_SPEC,
        'rear': LIDAR_FEATURE_SPEC,
    },
    # objects annotations data.
    'objects': tfds.features.Sequence(OBJECT_FEATURE_SPEC)
})
