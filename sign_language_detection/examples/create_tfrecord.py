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

"""Example code to create tfrecord for training."""

import numpy as np
import tensorflow as tf

with tf.io.TFRecordWriter('example.tfrecord') as writer:
  for _ in range(5):  # Iterate over 5 examples
    frames = 100  # Number of frames in the example video
    fps = 25  # FPS in the example video

    is_signing = np.random.randint(
        low=0, high=1, size=(frames), dtype='byte').tobytes()
    data = tf.io.serialize_tensor(
        tf.random.normal(shape=(frames, 1, 137, 2), dtype=tf.float32)).numpy()
    confidence = tf.io.serialize_tensor(
        tf.random.normal(shape=(frames, 1, 137), dtype=tf.float32)).numpy()

    features = {
        'fps':
            tf.train.Feature(int64_list=tf.train.Int64List(value=[fps])),
        'pose_data':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
        'pose_confidence':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[confidence])),
        'is_signing':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_signing]))
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())
