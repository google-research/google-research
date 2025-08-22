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

"""Tests for diffusion_unet."""

from absl.testing import absltest
from jax import random
import numpy as np

from simulation_research.diffusion.unet import UNet
from simulation_research.diffusion.unet import unet_64_config


class UnetTest(absltest.TestCase):

  def test_model_forward1d(self):
    traj_len = 64
    x = np.random.randn(2, traj_len, 3)
    t = np.linspace(0, 1, 2)
    model = UNet(unet_64_config(x.shape[-1]))
    params = model.init(random.PRNGKey(42), x=x, t=t, train=True)

    y = model.apply(params, x=x, t=t, train=True)
    assert y.shape == x.shape, f"{y.shape} != {x.shape}"

  def test_model_forward2d(self):
    height = width = 64
    x = np.random.randn(2, height, width, 3)
    t = np.linspace(0, 1, 2)
    model = UNet(unet_64_config(x.shape[-1]))
    params = model.init(random.PRNGKey(42), x=x, t=t, train=True)

    y = model.apply(params, x=x, t=t, train=True)
    assert y.shape == x.shape, f"{y.shape} != {x.shape}"


if __name__ == "__main__":
  absltest.main()
