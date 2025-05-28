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

"""score_sde tests."""
import unittest

from score_prior import utils
from score_prior.configs import score_model_config

# Parameters for shape of dummy data:
_IMAGE_SIZE = 8
_N_CHANNELS = 1
_IMAGE_SHAPE = (_IMAGE_SIZE, _IMAGE_SIZE, _N_CHANNELS)


class ScoreSdeTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    # Set up config for DPI.
    config = score_model_config.get_config()
    config.data.image_size = _IMAGE_SIZE
    config.data.num_channels = _N_CHANNELS
    config.training.batch_size = 1
    # Reduce model size to make test faster.
    config.model.name = 'ddpm'
    config.model.nf = 32
    config.model.ch_mult = (1, 1, 1, 1)
    config.model.attn_resolutions = (1,)
    config.model.num_res_blocks = 1
    self.config = config

  def test_init_score_model(self):
    config = self.config
    state, score_model, tx = utils.initialize_training_state(config)
    return state, score_model, tx


if __name__ == '__main__':
  unittest.main()
