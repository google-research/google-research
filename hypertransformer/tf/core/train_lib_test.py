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

"""Tests for `train_lib.py`."""

import numpy as np
import tensorflow.compat.v1 as tf

from hypertransformer.tf.core import common_ht as common
from hypertransformer.tf.core import train_lib


class TrainLibTest(tf.test.TestCase):

  def test_make_dataset_helper(self):
    """Tests `make_dataset_helper` function."""
    batch_size = 32
    num_labels = 4
    num_transformer_samples = 8
    num_cnn_samples = 16
    images = tf.zeros(shape=(batch_size, 2, 2, 1), dtype=tf.int8)
    labels = list(np.arange(num_labels)) * (batch_size // num_labels)
    labels = tf.constant(np.array(labels).astype(np.int32))
    ds = tf.data.Dataset.from_tensor_slices({'image': images,
                                             'label': labels})
    model_config = common.LayerwiseModelConfig(
        num_transformer_samples=num_transformer_samples,
        num_cnn_samples=num_cnn_samples,
        image_size=4)
    dataset_info = common.DatasetInfo(num_labels=num_labels,
                                      num_samples_per_label=8,
                                      transpose_images=False)
    data_config = common.DatasetConfig(dataset_name='dataset',
                                       ds=ds,
                                       dataset_info=dataset_info)
    with self.session() as sess:
      samples, _ = train_lib.make_dataset(model_config=model_config,
                                          data_config=data_config,
                                          shuffle_labels=True)
      sess.run(tf.global_variables_initializer())
      outputs = sess.run({'train_images': samples.transformer_images,
                          'train_labels': samples.transformer_labels,
                          'cnn_images': samples.cnn_images,
                          'cnn_labels': samples.cnn_labels})
      self.assertEqual(outputs['train_images'].shape,
                       (num_transformer_samples, 4, 4, 1))
      self.assertEqual(outputs['train_labels'].shape,
                       (num_transformer_samples,))
      self.assertEqual(outputs['cnn_images'].shape, (num_cnn_samples, 4, 4, 1))
      self.assertEqual(outputs['cnn_labels'].shape, (num_cnn_samples,))


if __name__ == '__main__':
  tf.test.main()
