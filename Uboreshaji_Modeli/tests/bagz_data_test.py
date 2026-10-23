# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

import itertools
from unittest import mock

from absl.testing import absltest
from grain.python import fast_proto
import ml_collections
import numpy as np

from Uboreshaji_Modeli.common import audio_utils
from Uboreshaji_Modeli.common import bagz_data


def _records(n):
  return [
      fast_proto.make_tf_example({
          "audio/bytes": [b"audio"],
          "transcription": [f"r{i}".encode()],
      }).SerializeToString()
      for i in range(n)
  ]


def _collate(examples):
  texts = [e["messages"][2]["content"][0]["text"] for e in examples]
  return {"input_ids": texts}


def _take(n, **kwargs):
  stream = bagz_data.TrainStream(_records(10), _collate, seed=3, **kwargs)
  return [b["input_ids"] for b in itertools.islice(stream, n)]


class TrainStreamTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            audio_utils, "get_audio_array", return_value=np.zeros(4)
        )
    )

  def test_train_source_skips_empty_shards(self):
    root = self.create_tempdir()
    shard = root.create_file("x-train-00000-of-00002.bagz", content="x")
    root.create_file("x-train-00001-of-00002.bagz")
    cfg = ml_collections.ConfigDict({
        "dataset": {"dataset_path": root.full_path, "train_split": "train"}
    })
    with mock.patch.object(bagz_data.grain, "BagDataSource") as source:
      self.assertIs(bagz_data.train_source(cfg), source.return_value)
    source.assert_called_once_with([shard.full_path])
    cfg.dataset.train_split = "val"
    with self.assertRaises(FileNotFoundError):
      bagz_data.train_source(cfg)

  def test_resume_continues_where_fresh_stream_is_after_start_step(self):
    fresh = _take(3 * 2 + 4, batch_size=2, grad_accum=3)
    resumed = _take(4, batch_size=2, grad_accum=3, start_step=2)
    self.assertEqual(resumed, fresh[6:])

  def test_ranks_split_each_epoch_exactly_once(self):
    kwargs = dict(batch_size=1, grad_accum=1, world_size=2)
    records = sum(_take(5, rank=0, **kwargs), []) + sum(
        _take(5, rank=1, **kwargs), []
    )
    self.assertCountEqual(records, [f"r{i}" for i in range(10)])

  def test_undecodable_audio_raises(self):
    audio_utils.get_audio_array.return_value = None
    with (
        mock.patch.object(
            bagz_data.torch_data.IterableDataset, "__init__"
        ) as super_init,
        self.assertRaises(ValueError),
    ):
      _take(1, batch_size=1, grad_accum=1)
    super_init.assert_called_once_with()


if __name__ == "__main__":
  absltest.main()
