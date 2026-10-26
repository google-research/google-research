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

from unittest import mock

from absl.testing import absltest
import ml_collections
import numpy as np
import torch

from Uboreshaji_Modeli.engines import gemma_audio


class GemmaAudioEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.engine = gemma_audio.GemmaAudioEngine()

  def test_get_sft_config_overrides(self):
    cfg = ml_collections.ConfigDict()
    overrides = self.engine.get_sft_config_overrides(cfg)
    self.assertEqual(
        overrides,
        {
            "dataset_kwargs": {"skip_prepare_dataset": True},
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        },
    )

  @mock.patch.object(gemma_audio, "transformers")
  def test_load_model_and_processor(self, mock_tf):
    model, processor = self.engine.load_model_and_processor("id", "cpu")
    load = mock_tf.AutoModelForMultimodalLM.from_pretrained
    load.assert_called_once_with("id", attn_implementation="eager")
    model.to.assert_called_once_with("cpu")
    self.assertEqual(processor.tokenizer.padding_side, "right")

  @mock.patch.object(gemma_audio, "peft")
  @mock.patch.object(gemma_audio, "transformers")
  def test_load_model_and_processor_with_lora(self, mock_tf, mock_peft):
    lora = {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ("q",)}
    cfg = ml_collections.ConfigDict(
        {"training": {"use_lora": True, "lora": lora}}
    )
    model, _ = self.engine.load_model_and_processor("id", "cpu", cfg=cfg)
    mock_peft.LoraConfig.assert_called_once_with(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q"],
    )
    mock_peft.get_peft_model.assert_called_once_with(
        mock_tf.AutoModelForMultimodalLM.from_pretrained.return_value,
        mock_peft.LoraConfig.return_value,
    )
    self.assertIs(model, mock_peft.get_peft_model.return_value)
    model.to.assert_called_once_with("cpu")

  def test_get_transform_fn(self):
    mock_processor = mock.Mock()

    cfg = ml_collections.ConfigDict()
    cfg.prompt = "Translate this:"

    transform_fn = self.engine.get_transform_fn(
        processor=mock_processor,
        text_inputs=[],
        dataset_id2label=[],
        model_label2id={},
        cfg=cfg,
    )

    batch = {
        "audio": {"array": np.zeros(16000), "sampling_rate": 16000},
        "text": "Hello world",
    }

    output = transform_fn(batch)

    # transform_fn now produces chat messages and decoded audio.
    self.assertIn("messages", output)
    self.assertIn("audio", output)
    self.assertLen(output["messages"], 1)
    self.assertLen(output["audio"], 1)

    messages = output["messages"][0]
    self.assertEqual(messages[0]["role"], "user")
    self.assertEqual(messages[1]["role"], "assistant")
    self.assertEqual(messages[1]["content"][0]["text"], "Hello world")

    self.assertIn("array", output["audio"][0])
    self.assertIn("sampling_rate", output["audio"][0])

    # Processor should NOT be called during transform.
    mock_processor.assert_not_called()

  def test_get_transform_fn_no_text(self):
    mock_processor = mock.Mock()
    cfg = ml_collections.ConfigDict()
    transform_fn = self.engine.get_transform_fn(
        processor=mock_processor,
        text_inputs=[],
        dataset_id2label=[],
        model_label2id={},
        cfg=cfg,
    )

    batch = {
        "audio": {"array": np.zeros(16000), "sampling_rate": 16000},
    }

    output = transform_fn(batch)

    self.assertIn("messages", output)
    self.assertIn("audio", output)
    self.assertLen(output["messages"], 1)
    self.assertLen(output["audio"], 1)

    # Verify message structure (assistant role with empty text).
    messages = output["messages"][0]
    self.assertLen(messages, 2)
    self.assertEqual(messages[0]["role"], "user")
    self.assertEqual(messages[1]["role"], "assistant")
    self.assertEqual(messages[1]["content"][0]["text"], "")

  def test_get_transform_fn_multiple_audio_no_text(self):
    mock_processor = mock.Mock()
    cfg = ml_collections.ConfigDict()
    transform_fn = self.engine.get_transform_fn(
        processor=mock_processor,
        text_inputs=[],
        dataset_id2label=[],
        model_label2id={},
        cfg=cfg,
    )

    # Test multiple audio inputs with no text.
    batch = {
        "audio": [
            {"array": np.zeros(16000), "sampling_rate": 16000},
            {"array": np.zeros(16000), "sampling_rate": 16000},
        ],
    }

    output = transform_fn(batch)

    self.assertIn("messages", output)
    self.assertIn("audio", output)
    self.assertLen(output["messages"], 2)
    self.assertLen(output["audio"], 2)

    # Verify message structure (assistant role with empty text).
    for messages in output["messages"]:
      self.assertLen(messages, 2)
      self.assertEqual(messages[0]["role"], "user")
      self.assertEqual(messages[1]["role"], "assistant")
      self.assertEqual(messages[1]["content"][0]["text"], "")

  def test_get_transform_fn_transcription(self):
    mock_processor = mock.Mock()
    cfg = ml_collections.ConfigDict()
    transform_fn = self.engine.get_transform_fn(
        processor=mock_processor,
        text_inputs=[],
        dataset_id2label=[],
        model_label2id={},
        cfg=cfg,
    )

    batch = {
        "audio": {"array": np.zeros(16000), "sampling_rate": 16000},
        "transcription": "Hello world transcription",
    }

    output = transform_fn(batch)
    self.assertIn("messages", output)
    messages = output["messages"][0]
    self.assertEqual(
        messages[1]["content"][0]["text"], "Hello world transcription"
    )

  def test_get_transform_fn_sample_rate_override(self):
    mock_processor = mock.Mock()
    cfg = ml_collections.ConfigDict()
    cfg.training = ml_collections.ConfigDict()
    cfg.training.audio_sample_rate = 8000

    transform_fn = self.engine.get_transform_fn(
        processor=mock_processor,
        text_inputs=[],
        dataset_id2label=[],
        model_label2id={},
        cfg=cfg,
    )

    batch = {
        "audio": {"array": np.zeros(8000), "sampling_rate": 8000},
        "text": "Hello world",
    }

    with mock.patch.object(
        gemma_audio.audio_utils, "get_audio_array"
    ) as mock_get_audio:
      mock_get_audio.return_value = np.zeros(8000)
      output = transform_fn(batch)
      mock_get_audio.assert_called_once_with(batch["audio"], target_sr=8000)
      self.assertEqual(output["audio"][0]["sampling_rate"], 8000)

  def test_get_transform_fn_bad_audio(self):
    mock_processor = mock.Mock()
    cfg = ml_collections.ConfigDict()
    transform_fn = self.engine.get_transform_fn(
        processor=mock_processor,
        text_inputs=[],
        dataset_id2label=[],
        model_label2id={},
        cfg=cfg,
    )

    batch = {
        "audio": {"array": np.zeros(16000), "sampling_rate": 16000},
        "text": "Hello world",
    }

    with mock.patch.object(
        gemma_audio.audio_utils, "get_audio_array"
    ) as mock_get_audio:
      mock_get_audio.return_value = None
      output = transform_fn(batch)
      self.assertEmpty(output["messages"])
      self.assertEmpty(output["audio"])

  def test_get_collate_fn(self):
    mock_processor = mock.Mock()
    mock_processor.apply_chat_template.return_value = "formatted text"

    mock_processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "input_features": torch.tensor([[[0.1]]]),
    }

    class DummyTokenizer:
      pad_token_id = 0
      audio_token_id = 1

    mock_processor.tokenizer = DummyTokenizer()

    cfg = ml_collections.ConfigDict()

    collate_fn = self.engine.get_collate_fn(cfg=cfg, processor=mock_processor)

    examples = [{
        "messages": [{"role": "user", "content": "hello"}],
        "audio": {"array": np.zeros(10), "sampling_rate": 16000},
    }]

    batch = collate_fn(examples)

    mock_processor.apply_chat_template.assert_called_once_with(
        examples[0]["messages"], tokenize=False, add_generation_prompt=False
    )
    mock_processor.assert_called_once_with(
        text=["formatted text"],
        audio=[examples[0]["audio"]["array"]],
        return_tensors="pt",
        padding=True,
    )

    self.assertIn("labels", batch)
    self.assertEqual(batch["labels"][0, 0].item(), -100)
    self.assertEqual(batch["labels"][0, 1].item(), 2)

  def test_get_collate_fn_tpu(self):
    mock_processor = mock.Mock()
    mock_processor.apply_chat_template.return_value = "formatted text"
    mock_processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
    }

    class DummyTokenizer:
      pass

    mock_processor.tokenizer = DummyTokenizer()

    cfg = ml_collections.ConfigDict()
    cfg.training = ml_collections.ConfigDict()
    cfg.training.device = "tpu"
    cfg.max_seq_length = 1024

    collate_fn = self.engine.get_collate_fn(cfg=cfg, processor=mock_processor)

    examples = [{
        "messages": [{"role": "user", "content": "hello"}],
        "audio": {"array": np.zeros(10), "sampling_rate": 16000},
    }]

    collate_fn(examples)

    mock_processor.assert_called_once_with(
        text=["formatted text"],
        audio=[examples[0]["audio"]["array"]],
        return_tensors="pt",
        padding="max_length",
        text_kwargs={"max_length": 1024},
    )


if __name__ == "__main__":
  absltest.main()
