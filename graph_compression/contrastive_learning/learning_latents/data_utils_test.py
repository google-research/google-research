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

"""Tests for data_utils."""

import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf

from graph_compression.contrastive_learning.learning_latents import data_utils


# create some fake datasets with the same structure as the real ones
NUM_EXAMPLES = 5
FAKE_DSPRITES_DF = pd.DataFrame(
    np.concatenate([
        np.random.randint(0, 2, (NUM_EXAMPLES, 7)),
        np.random.randn(NUM_EXAMPLES, 4)
    ],
                   axis=1),
    columns=data_utils.DSPRITES_SHAPE_NAMES + data_utils.DSPRITES_LABEL_NAMES +
    data_utils.DSPRITES_VALUE_NAMES)
FAKE_THREEDIDENT_DF = pd.DataFrame(
    np.concatenate([
        np.arange(NUM_EXAMPLES).reshape(-1, 1),
        np.random.randn(NUM_EXAMPLES, 10)
    ],
                   axis=1),
    columns=['id'] + data_utils.THREEDIDENT_VALUE_NAMES)


class DataTest(tf.test.TestCase):

  def test_dsprites_simple_noise_fn(self):
    df = FAKE_DSPRITES_DF
    result = data_utils.dsprites_simple_noise_fn(df.iloc[0], df)
    for latent_name in data_utils.DSPRITES_SHAPE_NAMES + data_utils.DSPRITES_LABEL_NAMES:
      self.assertIn(latent_name, result.keys())

  def test_threedident_simple_noise_fn(self):
    df = FAKE_THREEDIDENT_DF
    result = data_utils.threedident_simple_noise_fn(df.iloc[0], df)
    for latent_name in data_utils.THREEDIDENT_VALUE_NAMES:
      self.assertIn(latent_name, result.keys())


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
