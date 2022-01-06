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

"""Tests for training_hparams.py."""
import copy
import dataclasses
import json

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax


from aqt.jax.wmt_mlperf import train
from aqt.jax.wmt_mlperf import training_hparams
from aqt.jax.wmt_mlperf.hparams_configs import base_config
from aqt.utils import hparams_utils as os_hparams_utils


FLAGS = flags.FLAGS


class HParamsTest(parameterized.TestCase):

  def test_dynamic_hparams(self):
    # Check that a JITed function sees mutations to hparams arguments. We check
    # that mutating the learning rate (arbitrary choice) of an hparams instance
    # causes Jax to appropriately recompile. This test is useful since Jax
    # caches static arguments to JITed functions and we want to make sure that
    # Jax's recommended workflow of deep-copying an hparams instance that is
    # modified between training steps works correctly.
    def return_learning_rate(hparams):
      return hparams.learning_rate_schedule.base_learning_rate

    return_learning_rate_jit = jax.jit(
        return_learning_rate, static_argnums=(0,))
    hparams = os_hparams_utils.load_dataclass_from_config_dict(
        training_hparams.TrainingHParams,
        base_config.get_config(
            n_layers=3,
            quant_target=base_config.QuantTarget.weights_and_auto_acts))
    hparams.learning_rate_schedule.base_learning_rate = 1.0
    learning_rate = return_learning_rate_jit(hparams)
    self.assertEqual(learning_rate, 1.0)
    hparams = copy.deepcopy(hparams)
    hparams.learning_rate_schedule.base_learning_rate = 2.0
    mutated_learning_rate = return_learning_rate_jit(hparams)
    self.assertEqual(mutated_learning_rate, 2.0)


  def test_convert_lists_to_tuples(self):
    test_input = {
        'a': [
            {  #
                'b': [1, 2]
            },  #
            {
                'b': [3, 4]
            }
        ],
        'c': 'd',
        'e': [],
        'f': (1,)
    }
    test_output = os_hparams_utils._convert_lists_to_tuples(test_input)
    self.assertEqual(
        test_output,
        {
            'a': (
                {  #
                    'b': (1, 2)
                },  #
                {
                    'b': (3, 4)
                }  #
            ),
            'c': 'd',
            'e': (),
            'f': (1,)
        })
    test_input = [1, 2, 3]
    test_output = os_hparams_utils._convert_lists_to_tuples(test_input)
    self.assertEqual(test_output, (1, 2, 3))
    test_input = 1
    test_output = os_hparams_utils._convert_lists_to_tuples(test_input)
    self.assertEqual(test_output, 1)


if __name__ == '__main__':
  absltest.main()
