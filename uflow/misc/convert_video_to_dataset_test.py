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

"""Tests for convert_video_to_dataset."""

from absl.testing import absltest
import tensorflow as tf

from uflow.data import generic_flow_dataset
from uflow.misc import convert_video_to_dataset


class ConvertVideoToDatasetTest(absltest.TestCase):

  def test_video_parsing(self):
    """Test that we can convert a video to a dataset and load it correctly."""
    filepath = 'uflow/files/billiard_clip.mp4'
    output_dir = '/tmp/dataset'
    convert_video_to_dataset.convert_video(
        video_file_path=filepath,
        output_folder=output_dir)
    dataset = generic_flow_dataset.make_dataset(path=output_dir, mode='test')
    data_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    count = 0
    for element in data_iterator:
      image1, image2 = element
      count += 1
      self.assertEqual(image1.shape[0], image2.shape[0])
      self.assertEqual(image1.shape[1], image2.shape[1])
      self.assertEqual(image1.shape[2], 3)
    self.assertEqual(count, 299)

if __name__ == '__main__':
  absltest.main()
