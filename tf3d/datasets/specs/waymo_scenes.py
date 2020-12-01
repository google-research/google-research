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

"""Feature specification for Waymo `scene` Dataset.

A single `scene` data describes a 3D scene  which have been constructed from
collection of individual `frame` level data. Apart from some context features
common to all frames in the scene, we also store a lightweight sequence of frame
information corresponding to all frames in the scene.
"""

import tensorflow as tf
import tensorflow_datasets as tfds


# Feature specification of Frame information.
FRAME_INFO_FEATURE_SPEC = tfds.features.FeaturesDict({
    # A unique name that identifies a particular frame. This can be used to look
    # up a frame example from frame dataset.
    'frame_name': tfds.features.Text(),
    # Frame pose w.r.t scene: X_{scene} = R * X_{frame} + t.
    'pose': {
        'R': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
        't': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
    },
})

# Feature specification of an scene dataset.
SCENE_FEATURE_SPEC = tfds.features.FeaturesDict({
    # A unique name that identifies the scene.
    'scene_name': tfds.features.Text(),
    # Day, Dawn/Dusk, or Night, determined from sun elevation.
    'time_of_day': tfds.features.Text(),
    # Human readable location (e.g. CHD, SF) of the run segment.
    'location': tfds.features.Text(),
    # Sunny or Rain.
    'weather': tfds.features.Text(),
    # A scene have several frames. Sequence of frame information.
    'frames': tfds.features.Sequence(FRAME_INFO_FEATURE_SPEC),
})
