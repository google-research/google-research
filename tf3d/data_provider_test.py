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

"""Tests for tf3d.data_provider."""

import random
import tensorflow as tf
from tf3d import data_provider


class DatasetsTest(tf.test.TestCase,
                   ):

  def test_get_tf_data_dataset_tfrecord(self):
    dataset = data_provider.get_tf_data_dataset(
        dataset_name='waymo_object_per_frame',
        split_name='val',
        batch_size=1,
        is_training=True,
        preprocess_fn=None,
        feature_keys=None,
        label_keys=None,
        num_readers=1,
        filenames_shuffle_buffer_size=2,
        num_epochs=0,
        read_block_length=1,
        shuffle_buffer_size=2,
        num_parallel_batches=1,
        num_prefetch_batches=1,
        dataset_format='tfrecord',
    )
    tfrecord_features = next(iter(dataset))
    self.assertAllEqual(tfrecord_features['cameras/front/extrinsics/R'].shape,
                        [1, 3, 3])


if __name__ == '__main__':
  tf.test.main()
