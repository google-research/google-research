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

"""Tests for aqt.summary_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from aqt.jax.stats import Stats
from aqt.utils import summary_utils


class SummaryUtilsTest(parameterized.TestCase):

  def assertNestedDictEqual(self, a, b):
    np.testing.assert_equal(a.keys(), b.keys())
    for key in a:
      np.testing.assert_array_equal(a[key], b[key])

  @parameterized.named_parameters(
      dict(testcase_name='no_keys', keys=[]),
      dict(testcase_name='key_not_in_dict', keys=['bounds']))
  def test_empty_get_state_dict_summary(self, keys):
    state_dict = {}
    distr_summary = summary_utils.get_state_dict_summary(state_dict, keys=keys)
    self.assertEmpty(distr_summary)

  @parameterized.named_parameters(
      dict(
          testcase_name='keys_in_dict',
          keys=['bounds', 'min_per_ch', 'max_per_ch'],
          expected_summary={
              '/decoder/attention/dense_out/bounds':
                  np.array([[1., 2.], [2., 4.], [3., 6.]]),
              '/decoder/attention/dense_out/min_per_ch':
                  np.array([-6., -5., -4.]),
              '/decoder/attention/dense_out/max_per_ch':
                  np.array([20., 21., 22.]),
          }),
      dict(
          testcase_name='key_not_in_dict',
          keys=['other_key'],
          expected_summary={})
    )
  def test_get_state_dict_summary(self, keys, expected_summary):
    state_dict = {
        'decoder': {
            'attention': {
                'dense_out': {
                    'bounds':
                        jnp.array([[1., 2.], [2., 4.], [3., 6.]]),
                    'min_per_ch':
                        jnp.array([-6., -5., -4.]),
                    'max_per_ch':
                        jnp.array([20., 21., 22.]),
                    'stats':
                        Stats(
                            n=1,
                            mean=jnp.ones(()),
                            mean_abs=jnp.ones(()),
                            mean_sq=jnp.ones(()),
                            mean_batch_maximum=jnp.ones(()),
                            mean_batch_minimum=jnp.ones(()))
                }
            },
            'mlp': {
                'dense_1': {
                    'stats':
                        Stats(
                            n=1,
                            mean=jnp.ones(()),
                            mean_abs=jnp.ones(()),
                            mean_sq=jnp.ones(()),
                            mean_batch_maximum=jnp.ones(()),
                            mean_batch_minimum=jnp.ones(()))
                }
            },
        }
    }

    summary = summary_utils.get_state_dict_summary(state_dict, keys=keys)
    self.assertNestedDictEqual(summary, expected_summary)


if __name__ == '__main__':
  absltest.main()
