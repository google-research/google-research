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

"""Unit tests for train_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
from internal import configs
from internal import train_utils


class TrainUtilsTest(parameterized.TestCase):

  @parameterized.parameters((dict,), (flax.core.frozen_dict.FrozenDict,))
  def test_clip_gradients(self, dict_fn):
    """Confirm that gradient clipping returns the same type as the input."""
    gradient_data = {
        'params': {
            'foo': 1,
            'bar': -100,
            'baz': 100,
        },
    }

    clipping_config = configs.Config()
    input_dict = dict_fn(gradient_data)
    output_dict = train_utils.clip_gradients(input_dict, clipping_config)

    self.assertEqual(type(output_dict), type(input_dict))


if __name__ == '__main__':
  absltest.main()
