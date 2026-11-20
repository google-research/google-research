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

"""Grain batch streams and model-blind task decoders for Bagz shards."""

from collections.abc import Callable, Iterator, Mapping, Sequence
import functools
import io
import json
import threading
from typing import Any

from absl import logging
from etils import epath
import grain.python as grain
import ml_collections
import numpy as np
from PIL import Image
from PIL import ImageOps
from torch.utils import data as torch_data
import transformers

from Uboreshaji_Modeli.common import audio_utils
from Uboreshaji_Modeli.common import config

_SAMPLE_RATE = 16000
# Shared by the train and eval streams: HF fast tokenizers are not thread-safe.
_COLLATE_LOCK = threading.Lock()

_parse_tf_example = None


def parse_tf_example(record):
  """Parses a serialized `tf.train.Example` into feature sequences."""
  if _parse_tf_example is not None:
    return _parse_tf_example(record)  # pylint: disable=not-callable
  try:
    from tensorflow.core.example import example_pb2  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    raise ImportError(
        "Reading Bagz tf.Example shards outside Google requires tensorflow"
        " (or tensorflow-cpu); otherwise use Parquet or Hugging Face input."
    ) from e
  out: dict[str, Any] = {}
  example = example_pb2.Example.FromString(record)
  for k, feat in example.features.feature.items():
    kind = feat.WhichOneof("kind")
    if kind:
      out[k] = list(getattr(feat, kind).value)
  return out


def split_source(cfg, split):
  """Returns the non-empty `*-<split>-*.bagz` shards as one random-access source."""
  reader_cls = getattr(grain, "BagDataSource", None)
  if reader_cls is None:
    try:
      import bagz  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      raise ImportError(
          "Reading Bagz shards requires the `bagz` package; otherwise use"
          " Parquet or Hugging Face input."
      ) from e
    reader_cls = bagz.Reader
  root = epath.Path(cfg.dataset.dataset_path)
  pattern = f"*-{split}-*.bagz"
  shards = [str(p) for p in sorted(root.glob(pattern)) if p.stat().length]
  if not shards:
    raise FileNotFoundError(f"No non-empty Bagz shards in {root / pattern}.")
  return reader_cls(shards)


def _locked(collate_fn):
  @functools.wraps(collate_fn)
  def collate(examples):
    with _COLLATE_LOCK:
      return collate_fn(examples)

  return collate


def decode_asr(record):
  """Decodes a serialized ASR `tf.Example` into canonical `{audio, text}`."""
  features = parse_tf_example(record)
  audio = audio_utils.get_audio_array(
      features["audio/bytes"][0], target_sr=_SAMPLE_RATE
  )
  if audio is None:
    raise ValueError("Could not decode Bagz audio record.")
  return {
      "audio": {"array": audio, "sampling_rate": _SAMPLE_RATE},
      "text": features["transcription"][0].decode(),
  }


def decode_text(record):
  """Decodes a serialized text SFT `tf.Example` into canonical `{messages}`."""
  features = parse_tf_example(record)
  if "messages" in features:
    return {"messages": json.loads(features["messages"][0].decode())}
  return {
      "messages": [
          {"role": "user", "content": features["prompt"][0].decode()},
          {"role": "assistant", "content": features["response"][0].decode()},
      ]
  }


def decode_detection(
    record,
    class_names,
    exclude_classes = (),
):
  """Decodes a detection `tf.Example` into `{image_id, image, objects}`."""
  if not class_names:
    raise ValueError("Detection Bagz decoding requires dataset.class_names.")
  features = parse_tf_example(record)
  image_id = features.get("image_info/image_id", [None])[0]
  image = Image.open(io.BytesIO(features["image"][0]))
  image = ImageOps.exif_transpose(image).convert("RGB")
  size = (
      int(features["image_info/width"][0]),
      int(features["image_info/height"][0]),
  )
  if image.size != size:
    raise ValueError(
        f"Image {image_id}: decoded size {image.size} != recorded {size}."
    )
  bbox = np.asarray(features.get("annotations/bbox", ()), dtype=np.float32)
  labels = [b.decode() for b in features.get("annotations/label", ())]
  if bbox.size != 4 * len(labels):
    raise ValueError(
        f"Image {image_id}: {bbox.size} bbox values for {len(labels)} labels;"
        " expected 4 floats per label."
    )
  excluded = frozenset(exclude_classes)
  index = {name: i for i, name in enumerate(class_names)}
  if unknown := sorted(set(labels) - index.keys() - excluded):
    raise ValueError(
        f"Image {image_id}: labels {unknown} not in class_names"
        f" {list(class_names)}."
    )
  boxes_list = bbox.reshape(-1, 4).tolist()
  kept = [
      (box, index[label])
      for box, label in zip(boxes_list, labels)
      if label not in excluded
  ]
  return {
      "image_id": int(image_id) if image_id is not None else 0,
      "image": image,
      "objects": {
          "bbox": [b for b, _ in kept],
          "category": [c for _, c in kept],
      },
  }


def get_task_decoder(
    cfg,
):
  """Returns the model-blind Bagz record -> canonical task dict decoder."""
  task = cfg.get("task_type")
  if task in (config.TaskType.VISION_DETECTION, config.TaskType.DETECTION):
    names = cfg.dataset.get("class_names")
    if not names:
      raise ValueError("Detection Bagz requires cfg.dataset.class_names.")
    return functools.partial(
        decode_detection,
        class_names=tuple(names),
        exclude_classes=tuple(cfg.dataset.get("exclude_classes", ())),
    )
  asr_tasks = (
      config.TaskType.ASR,
      config.TaskType.CTC_ASR,
      config.TaskType.AST,
  )
  if task in asr_tasks:
    return decode_asr
  if task == config.TaskType.TEXT_SFT:
    return decode_text
  raise ValueError(f"Unsupported Bagz task_type: {task!r}.")


def _build_pipeline(
    source,
    slc,
    *,
    seed,
    repeat,
    batch_size,
    drop_remainder,
    collate_fn,
    decode_fn = None,
    preprocess_fn = None,
    read_opts = None,
):
  """Builds the shuffled, sliced, mapped, and collated Grain stream."""
  ds = grain.MapDataset.source(source).shuffle(seed=seed)
  if repeat:
    ds = ds.repeat()
  ds = ds.slice(slc)
  if decode_fn is not None:
    ds = ds.map(decode_fn)
  if preprocess_fn is not None:
    ds = ds.map(preprocess_fn)
  ds = ds.batch(
      batch_size, drop_remainder=drop_remainder, batch_fn=_locked(collate_fn)
  )
  return ds.to_iter_dataset(read_opts) if read_opts else ds.to_iter_dataset()


class TrainStream(torch_data.IterableDataset, transformers.TrainerCallback):
  """Endless per-rank stream of collated batches that starts at `start_step`."""

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
      decode_fn = None,
      preprocess_fn = None,
  ):
    super().__init__()
    start = start_step * grad_accum * batch_size * world_size
    # NOMUTANTS -- prefetch depth only; batch contents and order are unchanged.
    read_opts = grain.ReadOptions(prefetch_buffer_size=max(16, 2 * grad_accum))
    self._dataset = _build_pipeline(
        source,
        slice(start + rank, None, world_size),
        seed=seed,
        repeat=True,
        batch_size=batch_size,
        drop_remainder=True,
        collate_fn=collate_fn,
        decode_fn=decode_fn,
        preprocess_fn=preprocess_fn,
        read_opts=read_opts,
    )
    self._rank = rank

  def __iter__(self):
    return iter(self._dataset)

  def on_log(self, args, state, control, logs=None, **kwargs):
    if logs and self._rank == 0:
      logging.info("step %d: %s", state.global_step, logs)


def eval_stream(
    source,
    collate_fn,
    *,
    seed,
    num_records,
    batch_size,
    rank = 0,
    world_size = 1,
    decode_fn = None,
    preprocess_fn = None,
):
  """Fixed subset of `num_records` shuffled records as prefetched batches."""
  n = min(num_records, len(source))
  end = (
      n // (world_size * batch_size) * world_size * batch_size
      or n // world_size * world_size
  )
  return _build_pipeline(
      source,
      slice(rank, end, world_size),
      seed=seed,
      repeat=False,
      batch_size=batch_size,
      drop_remainder=False,
      collate_fn=collate_fn,
      decode_fn=decode_fn,
      preprocess_fn=preprocess_fn,
  )
