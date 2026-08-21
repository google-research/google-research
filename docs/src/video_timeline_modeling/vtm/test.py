# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Test functions for debugging.

To run these test functions:
1. Add 'python3 vtm/test.py' to command list in xm_launch.py
2. Launch the job.
"""

from absl import app
from absl import flags
from absl import logging
import torch
from torch.utils.data import DataLoader
from vtm.dataset import collate_topics
from vtm.dataset import TimelineDataset
from vtm.dataset import TimelineDatasetTest
from vtm.model.attention_head import AttentionHead
from vtm.model.attention_head import Encoder
from vtm.model.attention_head import TimelineModel

_DATA_PATH = flags.DEFINE_string('data_path', None, 'The dataset path.')


def test_encoder():
  """Test the Transformer based encoder."""

  logging.info('========================================')
  logging.info('===Test the Transformer based encoder')
  model = Encoder(16, 128, 2, 8, 0.1)
  x_input = torch.randn(32, 20, 16)
  logging.info('Initialized the model')
  x_encoder = model(x_input)
  logging.info('3rd dimension of x_encoder: %d', x_encoder.shape[-1])
  assert x_encoder.shape[-1] == 16
  logging.info('========================================')


def test_attn_head():
  """Test the attention head model."""

  logging.info('========================================')
  logging.info('===Test the Attention Head model')
  model = AttentionHead(128)
  x_key = torch.randn(16, 20, 128)
  x_query = torch.randn(16, 30, 128)
  logging.info('Initialized the model')
  log_score = model(x_query, x_key)
  logging.info('2nd dimension of attention score: %d', log_score.shape[1])
  logging.info('3rd dimension of attention score: %d', log_score.shape[2])
  assert log_score.shape[1] == 30
  assert log_score.shape[2] == 20
  logging.info('========================================')


def test_timeline_mode():
  """Test the whole Timeline model."""

  logging.info('========================================')
  logging.info('===Test the whole Timeline model.')
  model = TimelineModel(24, 30, 128, 60, 256, 8, 4)
  batch_video_x = torch.randn(16, 30, 60)
  batch_video_padding_mask = torch.randint(0, 2, (16, 30), dtype=torch.bool)
  logging.info('Initialized the model')
  log_score = model(batch_video_x, batch_video_padding_mask)
  logging.info('2nd dimension of attention score: %d', log_score.shape[1])
  logging.info('3rd dimension of attention score: %d', log_score.shape[2])
  assert log_score.shape[1] == 30
  assert log_score.shape[2] == 24
  logging.info('========================================')


def test_timeline_dataset():
  """Test the Timeline dataset (especially the padding collate function `collate_topics`)."""

  logging.info('========================================')
  logging.info('===Test the collate function.')
  dataset = TimelineDatasetTest()
  loader = DataLoader(
      dataset, batch_size=4, shuffle=False, collate_fn=collate_topics)
  for batch_data in loader:
    logging.info('2nd dimension of the first batch_data (video_features): %d',
                 batch_data['video_features'].shape[1])
    assert batch_data['video_features'].shape[1] == 4
    assert batch_data['video_cluster_label'].shape[1] == 4
    assert batch_data['video_padding_mask'].shape[1] == 4
    assert torch.equal(batch_data['video_features'][2, -1, :],
                       torch.Tensor([0, 0, 0, 0]))
    assert torch.equal(batch_data['video_cluster_label'][0, 1:],
                       torch.Tensor([-1, -1, -1]).to(torch.long))
    assert torch.equal(batch_data['video_padding_mask'][0],
                       torch.Tensor([0, 1, 1, 1]).to(torch.bool))
    break
  logging.info('========================================')
  logging.info('===Test the Timeline dataset.')
  train_dataset = TimelineDataset(partition='train', data_path=_DATA_PATH.value)
  ## We are trying to split the collected dataset into 80%/10%/10% roughly.
  ## The numbers are not exact due to some failure samples
  assert train_dataset[0]['cluster_text_features'].shape[0] == 23
  assert train_dataset[0]['video_features'].shape[0] == 106
  assert train_dataset[0]['video_features'].shape[-1] == 256
  logging.info('========================================')


def main(_):
  test_encoder()
  test_attn_head()
  test_timeline_mode()
  test_timeline_dataset()


if __name__ == '__main__':
  app.run(main)
