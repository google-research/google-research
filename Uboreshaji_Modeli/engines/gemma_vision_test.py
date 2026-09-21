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

import importlib.metadata
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import datasets
import ml_collections
from PIL import Image
import torch
import transformers

from Uboreshaji_Modeli.engines import gemma_vision


class GemmaEngineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.engine = gemma_vision.GemmaVisionEngine()

  @parameterized.named_parameters(
      dict(
          testcase_name="default",
          config_updates={},
          expected_overrides={
              "dataset_kwargs": {"skip_prepare_dataset": True},
              "gradient_checkpointing_kwargs": {"use_reentrant": False},
          },
      ),
      dict(
          testcase_name="with_max_seq_length",
          config_updates={"max_seq_length": 256},
          expected_overrides={
              "dataset_kwargs": {"skip_prepare_dataset": True},
              "gradient_checkpointing_kwargs": {"use_reentrant": False},
              "max_seq_length": 256,
          },
      ),
  )
  def test_get_sft_config_overrides(self, config_updates, expected_overrides):
    cfg = ml_collections.ConfigDict(config_updates)
    overrides = self.engine.get_sft_config_overrides(cfg)
    self.assertEqual(overrides, expected_overrides)

  def test_collate_fn_padding(self):
    """Tests collate with uniform patch counts (no Pan-and-Scan)."""
    collate_fn = self.engine.get_collate_fn()
    # Without PnS, each item has pixel_values of shape [1, C, H, W].
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "pixel_values": torch.ones((1, 3, 224, 224)),
        },
        {
            "input_ids": torch.tensor([4, 5]),
            "labels": torch.tensor([4, 5]),
            "attention_mask": torch.tensor([1, 1]),
            "pixel_values": torch.ones((1, 3, 224, 224)),
        },
    ]
    collated = collate_fn(batch)
    with self.subTest(name="Test input_ids shape"):
      self.assertEqual(collated["input_ids"].shape, (2, 3))
    with self.subTest(name="Test labels shape"):
      self.assertEqual(collated["labels"].shape, (2, 3))
    with self.subTest(name="Test attention_mask shape"):
      self.assertEqual(collated["attention_mask"].shape, (2, 3))
    with self.subTest(name="Test pixel_values shape"):
      # torch.cat of [1, C, H, W] * 2 → [2, C, H, W]
      self.assertEqual(collated["pixel_values"].shape, (2, 3, 224, 224))

    with self.subTest(name="Test padding values"):
      self.assertEqual(collated["input_ids"][1, 2].item(), 0)
      self.assertEqual(collated["labels"][1, 2].item(), -100)
      self.assertEqual(collated["attention_mask"][1, 2].item(), 0)

  def test_collate_fn_tpu_static_padding(self):
    """Tests that TPU collate pads to max_seq_length, not dynamic max."""
    cfg = ml_collections.ConfigDict({
        "training": {"device": "tpu"},
        "max_seq_length": 8,
    })
    collate_fn = self.engine.get_collate_fn(cfg=cfg)
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "pixel_values": torch.ones((1, 3, 224, 224)),
        },
        {
            "input_ids": torch.tensor([4, 5]),
            "labels": torch.tensor([4, 5]),
            "attention_mask": torch.tensor([1, 1]),
            "pixel_values": torch.ones((1, 3, 224, 224)),
        },
    ]
    collated = collate_fn(batch)
    # On TPU, pad to max_seq_length=8, not dynamic max=3.
    with self.subTest(name="Static input_ids shape"):
      self.assertEqual(collated["input_ids"].shape, (2, 8))
    with self.subTest(name="Static labels shape"):
      self.assertEqual(collated["labels"].shape, (2, 8))
    with self.subTest(name="Static attention_mask shape"):
      self.assertEqual(collated["attention_mask"].shape, (2, 8))

  def test_collate_fn_pan_and_scan_variable_crops(self):
    """Tests collate with variable patch counts from Pan-and-Scan."""
    collate_fn = self.engine.get_collate_fn()
    # PnS produces variable num_patches per image:
    # Item 0: 3 patches (base + 2 crops), Item 1: 1 patch (square, no crops).
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "labels": torch.tensor([-100, -100, 1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "pixel_values": torch.ones((3, 3, 224, 224)),  # base + 2 crops
        },
        {
            "input_ids": torch.tensor([6, 7]),
            "labels": torch.tensor([-100, 6]),
            "attention_mask": torch.tensor([1, 1]),
            "pixel_values": torch.ones((1, 3, 224, 224)),  # no crops
        },
    ]
    collated = collate_fn(batch)
    with self.subTest(name="Test pixel_values shape with variable crops"):
      # torch.cat of [3, C, H, W] and [1, C, H, W] → [4, C, H, W]
      self.assertEqual(collated["pixel_values"].shape, (4, 3, 224, 224))
    with self.subTest(name="Test input_ids padded to max length"):
      self.assertEqual(collated["input_ids"].shape, (2, 5))
    with self.subTest(name="Test pixel_values dtype is float"):
      self.assertEqual(collated["pixel_values"].dtype, torch.float32)

  def test_collate_fn_with_image_position_ids(self):
    """Tests collate with image_position_ids (used by Gemma 4)."""
    collate_fn = self.engine.get_collate_fn()
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "pixel_values": torch.ones((1, 3, 224, 224)),
            "image_position_ids": torch.ones((5, 2)),
        },
        {
            "input_ids": torch.tensor([4, 5]),
            "labels": torch.tensor([4, 5]),
            "attention_mask": torch.tensor([1, 1]),
            "pixel_values": torch.ones((1, 3, 224, 224)),
            "image_position_ids": torch.ones((5, 2)) * 2,
        },
    ]
    collated = collate_fn(batch)
    with self.subTest(name="Test image_position_ids presence"):
      self.assertIn("image_position_ids", collated)
    with self.subTest(name="Test image_position_ids shape"):
      # Stacks shape [5, 2] * 2 -> [2, 5, 2]
      self.assertEqual(collated["image_position_ids"].shape, (2, 5, 2))
    with self.subTest(name="Test image_position_ids values"):
      self.assertTrue((collated["image_position_ids"][0] == 1).all())
      self.assertTrue((collated["image_position_ids"][1] == 2).all())

  @mock.patch.object(
      transformers.AutoProcessor, "from_pretrained", autospec=True
  )
  @mock.patch.object(
      transformers.AutoModelForImageTextToText, "from_pretrained", autospec=True
  )
  def test_load_model_disables_kv_cache(
      self, mock_model_cls, mock_processor_cls
  ):
    mock_config = mock.create_autospec(
        transformers.PretrainedConfig, instance=True
    )
    mock_config.use_cache = True
    mock_model = mock.create_autospec(
        transformers.PreTrainedModel, instance=True
    )
    mock_model.config = mock_config
    mock_model_cls.return_value = mock_model

    mock_processor = mock.create_autospec(
        transformers.ProcessorMixin, instance=True
    )
    mock_processor.tokenizer = mock.create_autospec(
        transformers.PreTrainedTokenizer, instance=True
    )
    mock_processor_cls.return_value = mock_processor

    model, _ = self.engine.load_model_and_processor(
        "fake-model-id", torch.device("cpu")
    )

    self.assertFalse(model.config.use_cache)

  @mock.patch.object(
      transformers.utils.quantization_config,
      "is_torchao_available",
      autospec=True,
  )
  @mock.patch.object(importlib.metadata, "version", autospec=True)
  @mock.patch.object(
      transformers.AutoProcessor, "from_pretrained", autospec=True
  )
  @mock.patch.object(
      transformers.AutoModelForImageTextToText, "from_pretrained", autospec=True
  )
  @mock.patch.object(torch.cuda, "is_available", return_value=True)
  def test_load_model_with_torchao(
      self,
      mock_is_available,
      mock_model_cls,
      mock_processor_cls,
      mock_version,
      mock_is_torchao_available,
  ):
    del mock_is_available  # Unused.
    mock_is_torchao_available.return_value = True
    mock_version.return_value = "1.0.0"

    mock_config = mock.create_autospec(
        transformers.PretrainedConfig, instance=True
    )
    mock_model = mock.create_autospec(
        transformers.PreTrainedModel, instance=True
    )
    mock_model.config = mock_config
    mock_model_cls.return_value = mock_model

    mock_processor = mock.create_autospec(
        transformers.ProcessorMixin, instance=True
    )
    mock_processor.tokenizer = mock.create_autospec(
        transformers.PreTrainedTokenizer, instance=True
    )
    mock_processor_cls.return_value = mock_processor

    self.engine.load_model_and_processor(
        "fake-model-id", torch.device("cpu"), use_torchao=True
    )

    mock_model_cls.assert_called_once_with(
        "fake-model-id",
        torch_dtype=mock.ANY,
        attn_implementation="eager",
        quantization_config=mock.ANY,
        device_map={"": 0},
    )

  @parameterized.named_parameters(
      # Row 1: No TorchAO, rank 0
      dict(
          testcase_name="no_torchao_rank0",
          use_torchao=False,
          local_rank="0",
          expected_device_map={"": 0},
          expect_quantization_config=False,
      ),
      # Row 2: No TorchAO, rank 1
      dict(
          testcase_name="no_torchao_rank1",
          use_torchao=False,
          local_rank="1",
          expected_device_map={"": 1},
          expect_quantization_config=False,
      ),
      # Row 3: TorchAO, rank 0
      dict(
          testcase_name="torchao_rank0",
          use_torchao=True,
          local_rank="0",
          expected_device_map={"": 0},
          expect_quantization_config=True,
      ),
      # Row 4: TorchAO, rank 1
      dict(
          testcase_name="torchao_rank1",
          use_torchao=True,
          local_rank="1",
          expected_device_map={"": 1},
          expect_quantization_config=True,
      ),
  )
  @mock.patch.object(
      transformers.utils.quantization_config,
      "is_torchao_available",
      autospec=True,
  )
  @mock.patch.object(importlib.metadata, "version", autospec=True)
  @mock.patch.object(
      transformers.AutoProcessor, "from_pretrained", autospec=True
  )
  @mock.patch.object(
      transformers.AutoModelForImageTextToText, "from_pretrained", autospec=True
  )
  @mock.patch.object(torch.cuda, "is_available", return_value=True)
  def test_device_map_matrix(
      self,
      mock_is_available,
      mock_model_cls,
      mock_processor_cls,
      mock_version,
      mock_is_torchao_available,
      use_torchao,
      local_rank,
      expected_device_map,
      expect_quantization_config,
  ):
    """Verifies device_map for all use_torchao x rank combos."""
    del mock_is_available  # Unused.
    mock_is_torchao_available.return_value = True
    mock_version.return_value = "1.0.0"

    mock_config = mock.create_autospec(
        transformers.PretrainedConfig, instance=True
    )
    mock_model = mock.create_autospec(
        transformers.PreTrainedModel, instance=True
    )
    mock_model.config = mock_config
    mock_model_cls.return_value = mock_model

    mock_processor = mock.create_autospec(
        transformers.ProcessorMixin, instance=True
    )
    mock_processor.tokenizer = mock.create_autospec(
        transformers.PreTrainedTokenizer, instance=True
    )
    mock_processor_cls.return_value = mock_processor

    with mock.patch.dict(os.environ, {"LOCAL_RANK": local_rank}):
      self.engine.load_model_and_processor(
          "fake-model-id",
          torch.device("cpu"),
          use_torchao=use_torchao,
      )

    _, call_kwargs = mock_model_cls.call_args

    with self.subTest(name="device_map"):
      self.assertEqual(call_kwargs.get("device_map"), expected_device_map)

    with self.subTest(name="quantization_config"):
      if expect_quantization_config:
        self.assertIn("quantization_config", call_kwargs)
      else:
        self.assertNotIn("quantization_config", call_kwargs)

    with self.subTest(name="loc_tokens_added"):
      mock_processor.tokenizer.add_tokens.assert_called_once()
      added_tokens = mock_processor.tokenizer.add_tokens.call_args[0][0]
      self.assertLen(added_tokens, 1024)
      self.assertEqual(added_tokens[0], "<loc0000>")

    with self.subTest(name="embeddings_resized"):
      mock_model.resize_token_embeddings.assert_called_once_with(
          len(mock_processor.tokenizer), mean_resizing=False
      )

  def test_freeze_vision_tower(self):
    """Verifies that freeze_vision_tower sets requires_grad=False."""
    mock_model = mock.create_autospec(
        transformers.PreTrainedModel, instance=True
    )
    # Delete base_model so hasattr(mock_model, "base_model") returns False.
    del mock_model.base_model
    mock_model.vision_tower = mock.create_autospec(
        torch.nn.Module, instance=True
    )
    param1 = torch.nn.Parameter(torch.randn(2, 2))
    param2 = torch.nn.Parameter(torch.randn(3, 3))
    mock_model.vision_tower.parameters.return_value = [param1, param2]

    self.assertTrue(param1.requires_grad)
    self.assertTrue(param2.requires_grad)

    self.engine.freeze_vision_tower(mock_model)

    self.assertFalse(param1.requires_grad)
    self.assertFalse(param2.requires_grad)

  @parameterized.named_parameters(
      dict(
          testcase_name="default_prompt",
          config_updates={},
          expected_prompt="detect cat ; dog",
      ),
      dict(
          testcase_name="custom_prompt",
          config_updates={"prompt": "find all animals"},
          expected_prompt="find all animals",
      ),
      dict(
          testcase_name="none_config",
          config_updates=None,
          expected_prompt="detect cat ; dog",
      ),
  )
  def test_get_transform_fn_prompt_text(
      self, config_updates, expected_prompt
  ):
    """Verifies prompt_text generation from class names or config override."""
    mock_processor = mock.create_autospec(
        transformers.ProcessorMixin, instance=True
    )
    mock_processor.tokenizer = mock.create_autospec(
        transformers.PreTrainedTokenizer, instance=True
    )
    if config_updates is None:
      cfg = None
    else:
      cfg = ml_collections.ConfigDict({
          "dataset": {"image_size": None},
          **config_updates,
      })

    mock_category = mock.create_autospec(datasets.ClassLabel, instance=True)
    mock_category.feature = mock.create_autospec(
        datasets.features.features.Sequence, instance=True
    )
    mock_category.feature.names = ["cat", "dog"]
    mock_features = {"objects": {"category": mock_category}}

    transform_fn = self.engine.get_transform_fn(
        mock_processor,
        cfg=cfg,
        dataset_features=mock_features,
    )

    self.assertIsNotNone(transform_fn)
    # Call with a minimal example to verify prompt_text is used.
    mock_image = mock.create_autospec(Image.Image, instance=True)
    mock_image.convert.return_value = mock_image
    mock_image.size = (100, 100)
    mock_processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        # Shape [1, C, H, W]: 1 patch (no Pan-and-Scan crops).
        "pixel_values": torch.ones((1, 3, 224, 224)),
    }
    transform_fn({
        "image": [mock_image],
        "objects": [{"category": [0], "bbox": [[10, 20, 30, 40]]}],
    })
    # Verify apply_chat_template received messages with the expected prompt.
    chat_call_args = mock_processor.apply_chat_template.call_args
    messages = chat_call_args[0][0]
    user_text = messages[0]["content"][1]["text"]
    self.assertEqual(user_text, expected_prompt)

  def test_transform_fn_collects_image_position_ids(self):
    """Verifies that transform_fn passes through image_position_ids."""
    mock_processor = mock.create_autospec(
        transformers.ProcessorMixin, instance=True
    )
    mock_processor.tokenizer = mock.create_autospec(
        transformers.PreTrainedTokenizer, instance=True
    )
    mock_category = mock.create_autospec(datasets.ClassLabel, instance=True)
    mock_category.feature = mock.create_autospec(
        datasets.features.features.Sequence, instance=True
    )
    mock_category.feature.names = ["cat", "dog"]
    mock_features = {"objects": {"category": mock_category}}

    transform_fn = self.engine.get_transform_fn(
        mock_processor,
        cfg=ml_collections.ConfigDict({"dataset": {"image_size": None}}),
        dataset_features=mock_features,
    )

    mock_image = mock.create_autospec(Image.Image, instance=True)
    mock_image.convert.return_value = mock_image
    mock_image.size = (100, 100)

    # Mock processor to return image_position_ids (simulating Gemma 4 processor)
    mock_processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "pixel_values": torch.ones((1, 3, 224, 224)),
        "image_position_ids": torch.ones((1, 5, 2)),
    }

    result = transform_fn({
        "image": [mock_image],
        "objects": [{"category": [0], "bbox": [[10, 20, 30, 40]]}],
    })

    self.assertIn("image_position_ids", result)
    # The output from transform_fn should be squeezed along batch dim: [5, 2]
    self.assertEqual(result["image_position_ids"][0].shape, (5, 2))

  @mock.patch("transformers.AutoModelForImageTextToText.from_pretrained")
  @mock.patch("transformers.AutoProcessor.from_pretrained")
  def test_load_model_and_processor_tpu(
      self, mock_processor_class, mock_model_class
  ):
    mock_model = mock.create_autospec(torch.nn.Module, instance=True)
    mock_model.config = mock.Mock()
    mock_model.to.return_value = mock_model
    mock_model_class.return_value = mock_model
    mock_processor = mock.create_autospec(
        transformers.ProcessorMixin, instance=True
    )
    mock_processor.tokenizer = mock.create_autospec(
        transformers.PreTrainedTokenizer, instance=True
    )
    mock_processor_class.return_value = mock_processor

    class MockTpuDevice:
      type = "tpu"

    tpu_device = MockTpuDevice()

    model, processor = self.engine.load_model_and_processor(
        model_id="dummy_model_id",
        device=tpu_device,
        cfg=ml_collections.ConfigDict({
            "training": {
                "precision": "bf16",
                "use_torchao": False,
                "use_lora": False,
            },
            "detection_format": "json",
        }),
    )
    self.assertEqual(model, mock_model)
    self.assertEqual(processor, mock_processor)
    mock_model.to.assert_called_once_with(tpu_device)

  def test_gemma_vision_transform_explicit_model_id(self):
    """Verifies explicit model_id is preferred over processor.name_or_path."""

    class DummyProcessor:
      name_or_path = "google/gemma-3-4b-it"

    dummy_processor = DummyProcessor()

    transform = gemma_vision.GemmaVisionTransform(
        processor=dummy_processor,
        class_names=["cat", "dog"],
        prompt_text="Detect objects",
        detection_format="json",
        do_pan_and_scan=False,
        cfg=None,
        model_id="/tmp/explicit_model_id",
    )

    self.assertEqual(transform._model_id, "/tmp/explicit_model_id")

  def test_gemma_vision_transform_pickling(self):
    import copy  # pylint: disable=g-import-not-at-top

    class DummyProcessor:
      name_or_path = "google/gemma-3-4b-it"

    dummy_processor = DummyProcessor()

    transform = gemma_vision.GemmaVisionTransform(
        processor=dummy_processor,
        class_names=["cat", "dog"],
        prompt_text="Detect objects",
        detection_format="json",
        do_pan_and_scan=False,
        cfg=None,
    )

    # Test __getstate__
    state = transform.__getstate__()
    self.assertIsNone(state["processor"])

    with mock.patch.object(
        transformers.AutoProcessor, "from_pretrained", autospec=True
    ) as mock_from_pretrained:
      class DummyTokenizer:
        padding_side = "left"

        def add_tokens(self, tokens):
          pass

      class DummyReloadedProcessor:
        def __init__(self):
          self.tokenizer = DummyTokenizer()

      reloaded_processor = DummyReloadedProcessor()
      mock_from_pretrained.return_value = reloaded_processor

      cloned = copy.deepcopy(transform)

      mock_from_pretrained.assert_called_once_with("google/gemma-3-4b-it")
      self.assertEqual(cloned.processor, reloaded_processor)

  def test_gemma_vision_transform_pickling_loc_format(self):
    import copy  # pylint: disable=g-import-not-at-top

    class DummyProcessor:
      name_or_path = "google/gemma-3-4b-it"

    dummy_processor = DummyProcessor()

    transform = gemma_vision.GemmaVisionTransform(
        processor=dummy_processor,
        class_names=["cat", "dog"],
        prompt_text="Detect objects",
        detection_format="loc",
        do_pan_and_scan=False,
        cfg=None,
    )

    with mock.patch.object(
        transformers.AutoProcessor, "from_pretrained", autospec=True
    ) as mock_from_pretrained:

      mock_tokenizer = mock.Mock()
      mock_tokenizer.padding_side = "left"

      class DummyReloadedProcessor:
        def __init__(self):
          self.tokenizer = mock_tokenizer

      reloaded_processor = DummyReloadedProcessor()
      mock_from_pretrained.return_value = reloaded_processor

      cloned = copy.deepcopy(transform)

      mock_from_pretrained.assert_called_once_with("google/gemma-3-4b-it")
      self.assertEqual(cloned.processor, reloaded_processor)

      # Verify loc tokens were added
      mock_tokenizer.add_tokens.assert_called_once()
      args, _ = mock_tokenizer.add_tokens.call_args
      self.assertLen(args[0], 1024)
      self.assertEqual(args[0][0], "<loc0000>")
      self.assertEqual(args[0][-1], "<loc1023>")

      # Verify padding side
      self.assertEqual(mock_tokenizer.padding_side, "right")

  def test_get_transform_fn_saves_processor_locally(self):
    """Verifies processor.save_pretrained is called for DataLoader worker pre-caching."""
    engine = gemma_vision.GemmaVisionPreprocessor()

    class DummyProcessor:
      name_or_path = "google/gemma-3-4b-it"

      def save_pretrained(self, path):
        self.saved_path = path

    processor = DummyProcessor()

    dataset_features = {
        "objects": {
            "category": mock.Mock(
                feature=mock.Mock(names=["cat", "dog"]),
            ),
        },
    }

    cfg = ml_collections.ConfigDict({"detection_format": "json"})

    transform = engine.get_transform_fn(
        processor, cfg=cfg, is_train=False, dataset_features=dataset_features,
    )

    self.assertIsNotNone(processor.saved_path)
    self.assertIsInstance(transform, gemma_vision.GemmaVisionTransform)


if __name__ == "__main__":
  absltest.main()
