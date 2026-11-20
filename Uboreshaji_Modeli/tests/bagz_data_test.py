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
import json
from unittest import mock

from absl.testing import absltest
import bagz
import ml_collections
import numpy as np

from Uboreshaji_Modeli.common import audio_utils
from Uboreshaji_Modeli.common import bagz_data
from Uboreshaji_Modeli.common import config
from tensorflow.core.example import example_pb2


def _example(**features) -> bytes:
  ex = example_pb2.Example()
  for k, vals in features.items():
    ex.features.feature[k].bytes_list.value.extend(vals)
  return ex.SerializeToString()


def _records(n):
  return [
      _example(
          **{"audio/bytes": [b"audio"], "transcription": [f"r{i}".encode()]}
      )
      for i in range(n)
  ]


def _collate(examples):
  return {"input_ids": [e["text"] for e in examples]}


def _take(n, **kwargs):
  kwargs.setdefault("decode_fn", bagz_data.decode_asr)
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

  def test_split_source_skips_empty_shards_and_supports_oss_reader(self):
    root = self.create_tempdir()
    shard = root.create_file("x-train-00000-of-00002.bagz")
    with bagz.Writer(shard.full_path) as w:
      w.write(b"rec0")
    root.create_file("x-train-00001-of-00002.bagz")
    root.create_file("x-valid-00000-of-00001.bagz", content="y")
    cfg = ml_collections.ConfigDict(
        {"dataset": {"dataset_path": root.full_path}}
    )
    with mock.patch.object(bagz_data.grain, "BagDataSource") as source:
      self.assertIs(bagz_data.split_source(cfg, "train"), source.return_value)
    source.assert_called_once_with([shard.full_path])

    with mock.patch.object(bagz_data.grain, "BagDataSource", None):
      reader = bagz_data.split_source(cfg, "train")
      self.assertEqual(list(reader), [b"rec0"])

    with self.assertRaises(FileNotFoundError):
      bagz_data.split_source(cfg, "val")

  def test_parse_tf_example_oss_pb2_fallback(self):
    raw = _example(b=[b"hi", b"there"])
    with mock.patch.object(bagz_data, "_parse_tf_example", None):
      self.assertEqual(bagz_data.parse_tf_example(raw)["b"], [b"hi", b"there"])

  def test_resume_continues_where_fresh_stream_is_after_start_step(self):
    fresh = _take(3 * 2 + 4, batch_size=2, grad_accum=3)
    resumed = _take(4, batch_size=2, grad_accum=3, start_step=2)
    self.assertLen(fresh, 10)  # 20 records from 10: the stream repeats.
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

  def test_eval_ranks_get_equal_disjoint_repeatable_batches(self):
    def batches(rank):
      stream = bagz_data.eval_stream(
          _records(10),
          _collate,
          seed=3,
          num_records=7,
          batch_size=2,
          rank=rank,
          world_size=2,
          decode_fn=bagz_data.decode_asr,
      )
      first, second = list(stream), list(stream)
      self.assertEqual(first, second)
      return [b["input_ids"] for b in first]

    rank0, rank1 = batches(0), batches(1)
    self.assertLen(rank0, 1)
    self.assertLen(rank1, 1)
    self.assertEmpty(set(rank0[0]) & set(rank1[0]))

  def test_task_decoders_and_preprocess_pipeline(self):
    asr_cfg = ml_collections.ConfigDict({"task_type": config.TaskType.ASR})
    self.assertIs(bagz_data.get_task_decoder(asr_cfg), bagz_data.decode_asr)
    txt_cfg = ml_collections.ConfigDict({"task_type": config.TaskType.TEXT_SFT})
    msgs = [{"role": "user", "content": "hi"}]
    raw_txt = _example(messages=[json.dumps(msgs).encode()])
    self.assertEqual(
        bagz_data.get_task_decoder(txt_cfg)(raw_txt), {"messages": msgs}
    )
    raw_pr = _example(prompt=[b"hi"], response=[b"yo"])
    self.assertEqual(
        bagz_data.decode_text(raw_pr)["messages"][1],
        {"role": "assistant", "content": "yo"},
    )

    # Dict source with decode_fn=None + preprocess_fn runs the same pipeline.
    dicts = [{"text": "a"}, {"text": "b"}]
    stream = bagz_data.TrainStream(
        dicts,
        list,
        seed=0,
        batch_size=2,
        grad_accum=1,
        decode_fn=None,
        preprocess_fn=lambda d: d["text"].upper(),
    )
    self.assertCountEqual(next(iter(stream)), ["A", "B"])


if __name__ == "__main__":
  absltest.main()
