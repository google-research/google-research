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

"""Construct PyTorch dataset for timeline modeling."""

import glob
import os
from typing import Dict
import tensorflow as tf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

NUM_OF_TRAINING_SAMPLES = 9936
NUM_OF_VALIDATION_SAMPLES = 1255
NUM_OF_TEST_SAMPLES = 1220
MAX_NUM_CLUSTERS = 24


def collate_topics(topic_dicts,
                   max_num_cluster=MAX_NUM_CLUSTERS):
  """Customized batch collate function for padding variable-length video tokens.

  For example, if a batch has two data sample. The first one has 3 videos and 2
  clusters, while the second one has 4 videos and 3 clusters. Then, the
  data_batch['video_padding_mask'] for this batch is a (2,4) tensor: [[0, 0, 0,
  1], [0, 0, 0, 0]], where 1 denotes video padding token. The
  data_batch['cluster_non_padding_mask'] for this batch is a (2, 24) tensor:
  [[1, 1, 0, ..., 0], [1, 1, 1, 0, ..., 0]], where 0 denotes the cluster padding
  token. The features and labels are also padded accordingly.

  Args:
    topic_dicts (list[dict[str, torch.Tensor]]): the list of data to be batched,
      where each data is a dict with keys 'video_features',
      'cluster_text_features', and 'video_cluster_label'.
    max_num_cluster: the maximum number of clusters in the dataset, which is
      fixed to be 24 in our dataset.

  Returns:
    A dict with keys 'video_features', 'cluster_text_features',
    'video_cluster_label', 'video_padding_mask', and 'cluster_non_padding_mask'.
    Each value is a tensor. The first dimension of each value is batch_size,
    which is also the length of the input .
  """
  data_batch = {}
  data_batch['video_features'] = pad_sequence(
      [topic_dict['video_features'] for topic_dict in topic_dicts],
      batch_first=True)
  mask = [
      torch.zeros_like(
          topic_dict['video_features'][Ellipsis, 0].squeeze(-1),
          dtype=torch.bool).view(-1) for topic_dict in topic_dicts
  ]
  data_batch['video_padding_mask'] = pad_sequence(
      mask, batch_first=True, padding_value=1)
  data_batch['video_cluster_label'] = pad_sequence(
      [topic_dict['video_cluster_label'] for topic_dict in topic_dicts],
      batch_first=True,
      padding_value=-1)
  # Pad the first cluster sequence to the max_num_cluster length
  ## This ensures that the padded cluster num in each batch is max_num_cluster
  cluster_non_padding_mask = [
      torch.ones_like(
          topic_dict['cluster_text_features'][Ellipsis, 0].squeeze(-1),
          dtype=torch.bool).view(-1) for topic_dict in topic_dicts
  ]
  cluster_non_padding_mask[0] = torch.cat(
      (cluster_non_padding_mask[0],
       torch.zeros(
           max_num_cluster - cluster_non_padding_mask[0].shape[0],
           dtype=torch.bool).view(-1)),
      dim=0)
  data_batch['cluster_non_padding_mask'] = pad_sequence(
      cluster_non_padding_mask, batch_first=True, padding_value=0)
  topic_dicts[0]['cluster_text_features'] = torch.cat(
      (topic_dicts[0]['cluster_text_features'],
       torch.zeros(
           (max_num_cluster - topic_dicts[0]['cluster_text_features'].shape[0],
            topic_dicts[0]['cluster_text_features'].shape[-1]),
           dtype=torch.float)),
      dim=0)
  data_batch['cluster_text_features'] = pad_sequence(
      [topic_dict['cluster_text_features'] for topic_dict in topic_dicts],
      batch_first=True, padding_value=0)

  return data_batch


class TimelineDataset(Dataset):
  """The timline modeling dataset."""

  def __init__(self,
               partition='train',
               feature_key='vca_video_features_pulsar_embedding',
               feature_dim=256,
               data_path=None):
    super().__init__()
    # Data paths on google cloud storage
    if partition == 'train':
      path = os.path.join(data_path, 'train-*.tfrecord')
    elif partition == 'valid':
      path = os.path.join(data_path, 'val-*.tfrecord')
    elif partition == 'test':
      path = os.path.join(data_path, 'test-*.tfrecord')
    filenames = glob.glob(path)
    self.dataset = []
    raw_dataset = tf.data.TFRecordDataset(filenames)
    for raw_record in raw_dataset:
      data = {}
      (video_features, video_cluster_label, timeline_url,
       cluster_text_features) = self.parse_function(raw_record, feature_key)
      # Ignore the data sample without valid features.
      if video_features.shape[-1] == feature_dim:
        data['video_features'] = video_features
        data['video_cluster_label'] = video_cluster_label
        data['timeline_url'] = timeline_url.decode('ascii')
        data['cluster_text_features'] = cluster_text_features
        self.dataset.append(data)
    if partition == 'train':
      assert len(self.dataset) == NUM_OF_TRAINING_SAMPLES
    elif partition == 'valid':
      assert len(self.dataset) == NUM_OF_VALIDATION_SAMPLES
    elif partition == 'test':
      assert len(self.dataset) == NUM_OF_TEST_SAMPLES

  def parse_function(self, raw_record, feature_key):
    context_description = {
        'video_to_moment': tf.io.VarLenFeature(dtype=tf.int64),
        'webpage_url': tf.io.FixedLenFeature([], dtype=tf.string)
    }
    sequence_description = {
        feature_key: tf.io.VarLenFeature(dtype=tf.float32),
        'moment_newsembed_embedding': tf.io.VarLenFeature(dtype=tf.float32)
    }
    contexts, feature_lists = tf.io.parse_single_sequence_example(
        raw_record,
        context_features=context_description,
        sequence_features=sequence_description)
    video_features = torch.from_numpy(
        tf.sparse.to_dense(feature_lists[feature_key]).numpy())
    cluster_text_features = torch.from_numpy(
        tf.sparse.to_dense(feature_lists['moment_newsembed_embedding']).numpy())
    video_cluster_label = torch.from_numpy(
        tf.sparse.to_dense(contexts['video_to_moment']).numpy())
    timeline_url = contexts['webpage_url'].numpy()
    return (video_features, video_cluster_label,
            timeline_url, cluster_text_features)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    return self.dataset[index]


class TimelineDatasetTest(Dataset):
  """A random dataset used for testing the collate_topics function only."""

  def __init__(self):
    super().__init__()
    self.dataset = []
    # We randomly generated several data with certain number of videos and
    # clusters. The we can verify if the batched data via collate_topics
    # function is correct or not, in the test functions.
    for i in range(10):
      data = {}
      data['video_features'] = torch.randn(i + 1, 4)
      data['video_cluster_label'] = torch.randint(0, 2, (i + 1,))
      data['cluster_text_features'] = torch.randn(i + 5, 8)
      self.dataset.append(data)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    return self.dataset[index]
