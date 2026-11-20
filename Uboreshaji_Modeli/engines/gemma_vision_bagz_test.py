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

"""Tests for detection Bagz decoding and Gemma vision preprocessing/streams."""

import io
import re

from absl.testing import absltest
import ml_collections
from PIL import Image
import torch

from Uboreshaji_Modeli.common import bagz_data
from Uboreshaji_Modeli.common import config as common_config
from Uboreshaji_Modeli.engines import gemma_vision
from tensorflow.core.example import example_pb2

_CLASSES = ["leaf", "stem", "weed"]
_PROMPT = "detect leaf ; stem ; weed"
_W, _H = 200, 100
_CATS = {1: "leaf", 2: "stem", 5: "weed"}
_ANNS = (
    (1, 5, [100.0, 50.0, 50.0, 25.0]),
    (2, 1, [20.0, 10.0, 40.0, 30.0]),
)
_EXPECTED = (
    "<loc0102><loc0102><loc0409><loc0306> leaf ; "
    "<loc0511><loc0511><loc0767><loc0767> weed"
)
_END = "<end_of_turn>\n"
_SPECIAL = {
    "<bos>": 2, "<start_of_turn>": 105, "<end_of_turn>": 106, "<|turn>": 107
}
_SPECIAL_RE = re.compile("|".join(re.escape(t) for t in _SPECIAL))
_REV_SPECIAL = {v: k for k, v in _SPECIAL.items()}
_GEMMA3_TPL = (
    "<bos><start_of_turn>user\n{image}{prompt}<end_of_turn>\n"
    "<start_of_turn>model\n{target}<end_of_turn>\n"
)


def _jpeg(width=_W, height=_H, orientation=None):
  buf, exif = io.BytesIO(), Image.Exif()
  if orientation is not None:
    exif[0x0112] = orientation
  Image.new("RGB", (width, height), (10, 20, 30)).save(
      buf, format="JPEG", exif=exif
  )
  return buf.getvalue()


def _record(anns=_ANNS, *, img_bytes=None, width=_W, height=_H):
  ex = example_pb2.Example()
  f = ex.features.feature
  raw = _jpeg() if img_bytes is None else img_bytes
  f["image"].bytes_list.value.append(raw)
  f["image_info/height"].int64_list.value.append(height)
  f["image_info/width"].int64_list.value.append(width)
  f["image_info/image_id"].int64_list.value.append(7)
  f["annotations/bbox"].float_list.value.extend(())
  f["annotations/label_id"].int64_list.value.extend(())
  f["annotations/label"].bytes_list.value.extend(())
  f["annotations/id"].int64_list.value.extend(())
  for ann_id, cat_id, bbox in anns:
    f["annotations/bbox"].float_list.value.extend(bbox)
    f["annotations/label_id"].int64_list.value.append(cat_id)
    f["annotations/label"].bytes_list.value.append(_CATS[cat_id].encode())
    f["annotations/id"].int64_list.value.append(ann_id)
  return ex.SerializeToString()


class _FakeTokenizer:
  pad_token_id = 0
  padding_side = "right"

  def encode(self, text, add_special_tokens=False):
    del add_special_tokens
    ids, pos = [], 0
    for m in _SPECIAL_RE.finditer(text):
      ids += [ord(c) + 1000 for c in text[pos : m.start()]]
      ids.append(_SPECIAL[m.group()])
      pos = m.end()
    return ids + [ord(c) + 1000 for c in text[pos:]]


class _FakeProcessor:
  name_or_path = "fake"

  def __init__(self, template=_GEMMA3_TPL):
    self.tokenizer = _FakeTokenizer()
    self.template = template
    self.images = []

  def apply_chat_template(self, messages, tokenize, add_generation_prompt):
    del tokenize, add_generation_prompt
    user, assistant = messages
    return self.template.format(
        image="####",
        prompt=user["content"][1]["text"],
        target=assistant["content"][0]["text"],
    )

  def __call__(self, text, images, return_tensors, padding, do_pan_and_scan):
    del return_tensors, padding, do_pan_and_scan
    self.images.append(images)
    ids = torch.tensor([self.tokenizer.encode(text)])
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "token_type_ids": (ids == ord("#") + 1000).long(),
        "pixel_values": torch.zeros((1, 3, 4, 4)),
    }


def _decode(ids):
  return "".join(_REV_SPECIAL.get(t) or chr(t - 1000) for t in ids)


def _target(ex):
  return _decode(ex["labels"][ex["labels"] != -100].tolist())


def _cfg(**overrides):
  c = ml_collections.ConfigDict({
      "task_type": common_config.TaskType.VISION_DETECTION,
      "dataset": {"class_names": _CLASSES, "image_size": 0},
      "detection_format": "loc",
      "max_seq_length": 4096,
      "training": {},
  })
  c.update(overrides)
  return c


def _pipeline(processor=None, cfg=None):
  cfg = cfg or _cfg()
  dec = bagz_data.get_task_decoder(cfg)
  prep = gemma_vision.GemmaVisionEngine().get_preprocess_fn(
      processor or _FakeProcessor(), cfg=cfg
  )
  return lambda r: prep(dec(r))


class GemmaVisionBagzTest(absltest.TestCase):

  def test_decode_round_trip_zero_boxes_exclude_and_exif(self):
    ex = bagz_data.decode_detection(_record(), _CLASSES)
    self.assertEqual(ex["image_id"], 7)
    self.assertEqual(ex["image"].size, (_W, _H))
    self.assertEqual(
        ex["objects"],
        {
            "bbox": [[100.0, 50.0, 50.0, 25.0], [20.0, 10.0, 40.0, 30.0]],
            "category": [2, 0],
        },
    )
    filtered = bagz_data.decode_detection(
        _record(), ["leaf", "stem"], exclude_classes=["weed"]
    )
    self.assertEqual(
        filtered["objects"],
        {"bbox": [[20.0, 10.0, 40.0, 30.0]], "category": [0]},
    )
    self.assertEqual(
        bagz_data.decode_detection(_record(()), _CLASSES)["objects"],
        {"bbox": [], "category": []},
    )
    stripped = example_pb2.Example.FromString(_record(()))
    del stripped.features.feature["annotations/bbox"]
    del stripped.features.feature["annotations/label"]
    self.assertEqual(
        bagz_data.decode_detection(stripped.SerializeToString(), _CLASSES)[
            "objects"
        ],
        {"bbox": [], "category": []},
    )
    rotated = _record((), img_bytes=_jpeg(_H, _W, orientation=6))
    self.assertEqual(
        bagz_data.decode_detection(rotated, _CLASSES)["image"].size,
        (_W, _H),
    )

  def test_decode_validation_errors(self):
    with self.assertRaisesRegex(ValueError, "decoded size"):
      bagz_data.decode_detection(_record(width=_W + 1), _CLASSES)
    with self.assertRaisesRegex(ValueError, "not in class_names"):
      bagz_data.decode_detection(_record(), ["leaf", "stem"])
    bad = example_pb2.Example.FromString(_record())
    bad.features.feature["annotations/bbox"].float_list.value.append(1.0)
    with self.assertRaisesRegex(ValueError, "4 floats per label"):
      bagz_data.decode_detection(bad.SerializeToString(), _CLASSES)

  def test_loc_tokens_resize_and_prompt_masking(self):
    proc, cfg = _FakeProcessor(), _cfg()
    cfg.dataset.image_size = 64
    ex = _pipeline(proc, cfg)(_record())
    self.assertEqual(_target(ex), _EXPECTED + _END)
    self.assertEqual(proc.images[-1].size, (64, 64))
    ids, labels = ex["input_ids"].tolist(), ex["labels"].tolist()
    first = next(i for i, v in enumerate(labels) if v != -100)
    self.assertEqual(
        _decode(ids[:first]),
        f"<bos><start_of_turn>user\n####{_PROMPT}<end_of_turn>\n"
        "<start_of_turn>model\n",
    )
    self.assertEqual(_target(_pipeline()(_record(()))), _END)

    g4_tpl = (
        "<bos><|turn>user\n<|turn>model\n{image}{prompt}\n<|turn>model\n"
        "{target}"
    )
    g4_cfg = _cfg(model_flavor=common_config.ModelFlavor.GEMMA_4)
    self.assertEqual(
        _target(_pipeline(_FakeProcessor(g4_tpl), g4_cfg)(_record())), _EXPECTED
    )
    with self.assertRaisesRegex(ValueError, "turn marker"):
      _pipeline(_FakeProcessor("{image}{prompt}\n{target}"))(_record())

  def test_config_and_length_guards(self):
    with self.assertRaisesRegex(ValueError, "max_seq_length=20"):
      _pipeline(cfg=_cfg(max_seq_length=20))(_record())
    cfg = _cfg()
    del cfg.dataset.class_names
    with self.assertRaisesRegex(ValueError, "class_names"):
      bagz_data.get_task_decoder(cfg)
    with self.assertRaisesRegex(ValueError, "class_names"):
      gemma_vision.GemmaVisionEngine().get_preprocess_fn(
          _FakeProcessor(), cfg=cfg
      )

  def test_streams_with_separate_decode_and_preprocess(self):
    proc, cfg = _FakeProcessor(), _cfg()
    dec = bagz_data.get_task_decoder(cfg)
    prep = gemma_vision.GemmaVisionEngine().get_preprocess_fn(proc, cfg=cfg)
    collate = gemma_vision.GemmaVisionCollate()
    records = [_record(), _record(())] * 2
    ts = bagz_data.TrainStream(
        records,
        collate,
        seed=0,
        batch_size=2,
        grad_accum=1,
        decode_fn=dec,
        preprocess_fn=prep,
    )
    batch = next(iter(ts))
    self.assertEqual(batch["pixel_values"].shape, (2, 3, 4, 4))
    self.assertEqual(batch["token_type_ids"].shape, batch["input_ids"].shape)
    self.assertEqual(
        batch["token_type_ids"][batch["attention_mask"] == 0].sum().item(), 0
    )
    es = bagz_data.eval_stream(
        records,
        collate,
        seed=0,
        num_records=4,
        batch_size=2,
        decode_fn=dec,
        preprocess_fn=prep,
    )
    self.assertLen(list(es), 2)


if __name__ == "__main__":
  absltest.main()
