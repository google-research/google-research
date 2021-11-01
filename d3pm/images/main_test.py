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

"""Test for the main executable."""

import random

from absl.testing import absltest
from flax import linen as nn
import jax
import numpy as onp

from d3pm.images import main
from d3pm.images import main_test_config
from d3pm.images import model
from d3pm.images import utils


class MainTest(absltest.TestCase):

  def test_small_training_job(self):
    experiment_dir = self.create_tempdir().full_path
    work_unit_dir = self.create_tempdir().full_path

    # Disable compiler optimizations for faster compile time.
    jax.config.update('jax_disable_most_optimizations', True)

    # Seed the random number generators.
    random.seed(0)
    onp.random.seed(0)
    rng = utils.RngGen(jax.random.PRNGKey(0))

    # Construct a test config with a small number of steps.
    config = main_test_config.get_config()

    # Patch normalization so that we don't try to apply GroupNorm with more
    # groups than test channels.
    orig_normalize = model.Normalize
    try:
      model.Normalize = nn.LayerNorm

      # Make sure we can train without any exceptions.
      main.run_train(
          config=config,
          experiment_dir=experiment_dir,
          work_unit_dir=work_unit_dir,
          rng=rng)

    finally:
      model.Normalize = orig_normalize


if __name__ == '__main__':
  absltest.main()
