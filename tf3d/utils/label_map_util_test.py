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

"""Tests for tf3d.utils.label_map_util."""
import os
import tensorflow as tf
from tf3d.utils import label_map_util


class LabelMapUtilTest(tf.test.TestCase):

  def test_load_bad_label_map(self):
    label_map_string = """
      item {
        id:0
        name:'class that should not be indexed at zero'
      }
      item {
        id:2
        name:'cat'
      }
      item {
        id:1
        name:'dog'
      }
    """
    label_map_path = os.path.join(self.get_temp_dir(), 'label_map.pbtxt')
    with tf.io.gfile.GFile(label_map_path, 'wb') as f:
      f.write(label_map_string)

    with self.assertRaises(ValueError):
      label_map_util.load_labelmap(label_map_path)


if __name__ == '__main__':
  tf.test.main()
