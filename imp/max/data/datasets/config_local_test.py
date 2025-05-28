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

r"""Tests for datasets.

Run these only if/when you add an actual dataset with real tfrecord.

"""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from imp.max.data import config as data_config
from imp.max.data import tokenizers
from imp.max.data.datasets import config as datasets_config
from imp.max.data.datasets import dataloader

VOCABULARY = tokenizers.VOCABULARY


class DatasetsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (dataset.name, dataset) for dataset in datasets_config.ALL_DATASETS
  )
  def test_dataset_loading(self, config):
    """Locally-run smoke test for datasets."""

    logging.info('Testing dataset %s', config.name)

    batch_size = 32


    experiment = data_config.ExperimentData(
        vision_spatial_size=(32, 32),
        vision_spatial_patch_size=(16, 16),
        vision_temporal_size=1,
        vision_temporal_patch_size=1,
        waveform_temporal_size=256,
        waveform_temporal_patch_size=128,
        spectrogram_temporal_patch_size=2,
        spectrogram_spectoral_patch_size=16,
        text_size=64,
        num_epochs=1,
        loaders=(
            data_config.Loader(dataset=config,
                               batch_size=batch_size,
                               shuffle=False),
        ),
        is_training=False)

    loaders = dataloader.create_data(experiment)
    dataset = loaders[0]['loader']
    ds_iter = iter(dataset)
    example = next(ds_iter)
    self.assertNotEmpty(example)

    # Spot-check that outputs look reasonable
    logging.info('Example:\n%s', example)

    if 'vision' in example and 'label' in example['vision']:
      logging.info('Label:\n%s', example['vision']['label'].sum(-1))

    # Ensure the strings are not empty
    if 'text' in example:
      tokens = VOCABULARY.decode_tf(example['text']['token_id'])
      logging.info('Tokens:\n%s', tokens)
      self.assertNotEmpty(tokens[0])

    # Ensure that multiple examples can be retrieved.
    for _ in range(10):
      next(ds_iter)


if __name__ == '__main__':
  absltest.main()
