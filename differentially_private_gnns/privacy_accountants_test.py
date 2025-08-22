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

"""Tests for privacy_accountants."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from differentially_private_gnns import privacy_accountants


def get_privacy_accountant(training_type):
  """Gets the right accountant for the given training type."""
  if training_type in ['sgd', 'adam']:
    return privacy_accountants.dpsgd_privacy_accountant
  if training_type in ['multiterm-sgd', 'multiterm-adam']:
    return privacy_accountants.multiterm_dpsgd_privacy_accountant
  raise ValueError(f'Unsupported training_type: {training_type}.')


class PrivacyAccountantsTest(parameterized.TestCase):

  @parameterized.product(
      optimizer=['adam', 'sgd'],
      num_training_steps=[1, 10],
      noise_multiplier=[1],
      target_delta=[1e-5],
      batch_size=[10, 20],
      num_samples=[1000, 2000],
      max_terms_per_node=[1])
  def test_multiterm_vs_standard_dpsgd(self, optimizer,
                                       num_training_steps,
                                       noise_multiplier,
                                       target_delta,
                                       batch_size,
                                       num_samples,
                                       max_terms_per_node):
    multiterm_privacy_accountant = get_privacy_accountant('multiterm-' +
                                                          optimizer)
    multiterm_epsilon = multiterm_privacy_accountant(
        num_training_steps=num_training_steps,
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
        num_samples=num_samples,
        batch_size=batch_size,
        max_terms_per_node=max_terms_per_node)

    standard_privacy_accountant = get_privacy_accountant(optimizer)
    standard_epsilon = standard_privacy_accountant(
        num_training_steps=num_training_steps,
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
        sampling_probability=batch_size / num_samples)
    self.assertLessEqual(standard_epsilon, multiterm_epsilon)

  @parameterized.product(
      optimizer=['adam', 'sgd'],
      noise_multiplier=[-1, 0],
      target_delta=[1e-5],
      num_samples=[1000],
      batch_size=[10],
      max_terms_per_node=[1])
  def test_low_noise_multiplier_multiterm_dpsgd(self, optimizer, *args,
                                                **kwargs):
    privacy_accountant = get_privacy_accountant('multiterm-' + optimizer)
    self.assertEqual(privacy_accountant(num_training_steps=10, *args, **kwargs),
                     np.inf)

  @parameterized.product(
      optimizer=['adam', 'sgd'],
      noise_multiplier=[-1, 0],
      target_delta=[1e-5],
      sampling_probability=[0.1],
  )
  def test_low_noise_multiplier_dpsgd(self, optimizer, *args, **kwargs):
    privacy_accountant = get_privacy_accountant(optimizer)
    self.assertEqual(privacy_accountant(num_training_steps=10, *args, **kwargs),
                     np.inf)


if __name__ == '__main__':
  absltest.main()
