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

"""Test for the main executable."""

import random

from absl.testing import absltest
import jax
import numpy as np
import tensorflow_datasets as tfds

from d3pm.text import configs
from d3pm.text import main


class MainTest(absltest.TestCase):

  def test_small_training_job(self):
    experiment_dir = self.create_tempdir().full_path

    # Disable compiler optimizations for faster compile time.
    jax.config.update('jax_disable_most_optimizations', True)

    # Seed the random number generators.
    random.seed(0)
    np.random.seed(0)

    # Construct a test config with a small number of steps.
    configs.gin_load('lm1b_tiny')

    with tfds.testing.mock_data(num_examples=2048):
      # Make sure we can train without any exceptions.
      main.run_experiment(
          experiment_dir,
          batch_size_per_device=1,
          max_train_steps=1,
          validate_every=5,
          train_summary_frequency=5,
          num_eval_steps=5,
          num_predict_steps=1,
          restore_checkpoint=False,
          checkpoint_frequency=None,
      )


if __name__ == '__main__':
  absltest.main()
