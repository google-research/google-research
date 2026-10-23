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

"""Bagz shards as a resumable, per-rank Grain stream of SFT batches."""

from collections.abc import Callable, Iterator
import threading
from typing import Any

from absl import logging
from etils import epath
import grain.python as grain
from grain.python import fast_proto
import ml_collections
from torch.utils import data as torch_data
import transformers

from Uboreshaji_Modeli.common import audio_utils

_SAMPLE_RATE = 16000


def train_source(cfg):
  """Returns the non-empty `*-<train_split>-*.bagz` shards as one source."""
  root = epath.Path(cfg.dataset.dataset_path)
  pattern = f"*-{cfg.dataset.train_split}-*.bagz"
  shards = [str(p) for p in sorted(root.glob(pattern)) if p.stat().length]
  if not shards:
    raise FileNotFoundError(f"No non-empty Bagz shards in {root / pattern}.")
  return grain.BagDataSource(shards)


def _example(record):
  """Decodes a serialized ASR tf.Example into an SFT chat example."""
  features = fast_proto.parse_tf_example_grain_dict(record)
  audio = audio_utils.get_audio_array(
      features["audio/bytes"][0], target_sr=_SAMPLE_RATE
  )
  if audio is None:
    raise ValueError("Could not decode Bagz audio record.")
  messages = [
      {
          "role": "system",
          "content": "You are an assistant that transcribes speech accurately.",
      },
      {
          "role": "user",
          "content": [
              {"type": "audio", "audio": audio},
              {"type": "text", "text": "Please transcribe this audio."},
          ],
      },
      {
          "role": "assistant",
          "content": [{
              "type": "text",
              "text": features["transcription"][0].decode(),
          }],
      },
  ]
  return {
      "messages": messages,
      "audio": {"array": audio, "sampling_rate": _SAMPLE_RATE},
  }


class TrainStream(torch_data.IterableDataset, transformers.TrainerCallback):
  """Endless per-rank stream of collated batches that starts at `start_step`.

  Rank r reads records `start + r, start + r + W, ...` of the shuffled,
  repeated source, so the stream position after N optimizer steps is a pure
  function of N: resuming skips ahead without decoding anything. Resume assumes
  the same world size, batch size and gradient accumulation.

  As a callback, it logs the trainer metrics (loss, eval_loss) on rank 0.
  """

  def __init__(
      self,
      source,
      collate_fn,
      *,
      seed,
      batch_size,
      grad_accum,
      start_step = 0,
      rank = 0,
      world_size = 1,
  ):
    super().__init__()
    start = start_step * grad_accum * batch_size * world_size
    lock = threading.Lock()

    def collate(examples):
      with lock:  # HF fast tokenizers are not thread-safe.
        return collate_fn(examples)

    self._dataset = (
        grain.MapDataset.source(source)
        .shuffle(seed=seed)
        .repeat()
        .slice(slice(start + rank, None, world_size))
        .map(_example)
        .batch(batch_size, drop_remainder=True, batch_fn=collate)
        .to_iter_dataset(grain.ReadOptions(prefetch_buffer_size=2 * grad_accum))
    )
    self._rank = rank

  def __iter__(self):
    return iter(self._dataset)

  def on_log(self, args, state, control, logs=None, **kwargs):
    if logs and self._rank == 0:
      logging.info("step %d: %s", state.global_step, logs)
