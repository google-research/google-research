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
import peft
import torch
import transformers

from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.engines import gemma_text


class GemmaTextEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.engine = gemma_text.GemmaTextEngine()

  @mock.patch.object(
      transformers.AutoTokenizer, "from_pretrained", autospec=True
  )
  @mock.patch.object(
      transformers.AutoModelForCausalLM, "from_pretrained", autospec=True
  )
  def test_load_model_and_processor_no_lora(self, mock_model_cls, mock_tok_cls):
    mock_tok = mock.create_autospec(
        transformers.PreTrainedTokenizer, instance=True
    )
    mock_tok.pad_token = None
    mock_tok.eos_token = "<eos>"
    mock_tok_cls.return_value = mock_tok
    mock_model = mock.create_autospec(
        transformers.PreTrainedModel, instance=True
    )
    mock_model_cls.return_value = mock_model

    model, tokenizer = self.engine.load_model_and_processor(
        "mock-id", torch.device("cpu")
    )
    with self.subTest(name="Verify loaded model"):
      self.assertEqual(model, mock_model)
    with self.subTest(name="Verify loaded tokenizer"):
      self.assertEqual(tokenizer, mock_tok)
    with self.subTest(name="Verify path pad token initialized"):
      self.assertIsNotNone(tokenizer.pad_token)

  @mock.patch.object(peft, "get_peft_model", autospec=True)
  @mock.patch.object(
      transformers.AutoTokenizer, "from_pretrained", autospec=True
  )
  @mock.patch.object(
      transformers.AutoModelForCausalLM, "from_pretrained", autospec=True
  )
  def test_load_model_and_processor_with_lora(
      self, mock_model_cls, mock_tok_cls, mock_get_peft
  ):
    mock_tok = mock.create_autospec(
        transformers.PreTrainedTokenizer, instance=True
    )
    mock_tok.pad_token = "<pad>"
    mock_tok_cls.return_value = mock_tok
    mock_model = mock.create_autospec(
        transformers.PreTrainedModel, instance=True
    )
    mock_model_cls.return_value = mock_model
    mock_peft_model = mock.create_autospec(peft.PeftModel, instance=True)
    mock_get_peft.return_value = mock_peft_model

    cfg = ml_collections.ConfigDict({
        "model_flavor": config.ModelFlavor.GEMMA_3_TEXT,
        "training": {
            "use_lora": True,
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": ["q_proj"],
            },
        }
    })

    model, _ = self.engine.load_model_and_processor(
        "mock-id", torch.device("cpu"), cfg=cfg
    )
    with self.subTest(name="Verify wrapped PEFT model"):
      self.assertEqual(model, mock_peft_model)
    with self.subTest(name="Verify PEFT wrapper creation called"):
      mock_get_peft.assert_called_once()

  def test_collate_fn_padding(self):
    # Composed preprocessor expects pad_token_id in kwargs
    collate_fn = self.engine.get_collate_fn(pad_token_id=0)
    batch = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 2, 3],
        },
        {
            "input_ids": [4, 5],
            "attention_mask": [1, 1],
            "labels": [-100, 5],
        },
    ]
    collated = collate_fn(batch)
    with self.subTest(name="Verify collated input_ids shape"):
      self.assertEqual(collated["input_ids"].shape, (2, 3))
    with self.subTest(name="Verify collated labels shape"):
      self.assertEqual(collated["labels"].shape, (2, 3))
    with self.subTest(name="Verify padding token"):
      self.assertEqual(collated["input_ids"][1, 2].item(), 0)
    with self.subTest(name="Verify padding mask labels"):
      self.assertEqual(collated["labels"][1, 2].item(), -100)
    with self.subTest(name="Verify collated input_ids content"):
      self.assertEqual(collated["input_ids"][0, 0].item(), 1)
      self.assertEqual(collated["input_ids"][0, 1].item(), 2)
      self.assertEqual(collated["input_ids"][0, 2].item(), 3)
      self.assertEqual(collated["input_ids"][1, 0].item(), 4)
      self.assertEqual(collated["input_ids"][1, 1].item(), 5)

  def test_get_transform_fn_masking(self):
    mock_tok = mock.create_autospec(
        transformers.PreTrainedTokenizer, instance=True
    )
    mock_tok.apply_chat_template.return_value = "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\nhello<end_of_turn>\n"
    mock_tok.return_value = {
        "input_ids": [101, 102, 103, 201, 202, 203, 204],
        "attention_mask": [1, 1, 1, 1, 1, 1, 1],
    }
    mock_tok.encode.return_value = [201, 202]

    cfg = ml_collections.ConfigDict(
        {"model_flavor": config.ModelFlavor.GEMMA_3_TEXT, "max_seq_length": 128}
    )

    # Align call to ModelEngine base signature (dummy positional arguments)
    transform_fn = self.engine.get_transform_fn(mock_tok, [], [], {}, cfg=cfg)

    batch = {
        "messages": [[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]]
    }
    transformed = transform_fn(batch)

    with self.subTest(name="Verify prompt and assistant SFT labels masking"):
      self.assertEqual(
          transformed["labels"][0], [-100, -100, -100, -100, -100, 203, 204]
      )

  def test_gemma_text_transform_explicit_model_id(self):
    """Verifies explicit model_id is preferred over processor.name_or_path."""

    class DummyProcessor:
      name_or_path = "google/gemma-3-4b-it"
      chat_template = "{{ messages }}"

    dummy_processor = DummyProcessor()

    transform = gemma_text.GemmaTextTransform(
        processor=dummy_processor,
        max_length=128,
        cfg=None,
        model_id="/tmp/explicit_model_id",
    )

    self.assertEqual(transform._model_id, "/tmp/explicit_model_id")

  def test_gemma_text_transform_pickling(self):
    """Verifies __getstate__/__setstate__ round-trip via deepcopy."""
    import copy  # pylint: disable=g-import-not-at-top

    class DummyProcessor:
      name_or_path = "google/gemma-3-4b-it"
      chat_template = "{{ messages }}"

    dummy_processor = DummyProcessor()

    transform = gemma_text.GemmaTextTransform(
        processor=dummy_processor,
        max_length=128,
        cfg=None,
    )

    # Verify _model_id is set from processor.name_or_path
    self.assertEqual(transform._model_id, "google/gemma-3-4b-it")

    # Test __getstate__ strips processor
    state = transform.__getstate__()
    self.assertIsNone(state["processor"])

    # Test __setstate__ via deepcopy
    with mock.patch.object(
        transformers.AutoTokenizer, "from_pretrained", autospec=True
    ) as mock_from_pretrained:
      mock_tok = mock.create_autospec(
          transformers.PreTrainedTokenizer, instance=True
      )
      mock_tok.pad_token = None
      mock_tok.eos_token = "<eos>"
      mock_tok.chat_template = None
      mock_from_pretrained.return_value = mock_tok

      cloned = copy.deepcopy(transform)

      mock_from_pretrained.assert_called_once_with("google/gemma-3-4b-it")
      self.assertEqual(cloned.processor, mock_tok)
      # Verify chat_template was restored from the saved copy
      self.assertEqual(cloned.processor.chat_template, "{{ messages }}")
      # Verify pad_token was set to eos_token
      self.assertEqual(cloned.processor.pad_token, "<eos>")

  def test__post_load_processor_pad_token(self):
    """Verifies pad_token logic in _post_load_processor."""

    # Case 1: pad_token is None
    mock_tok_none = mock.Mock()
    mock_tok_none.pad_token = None
    mock_tok_none.eos_token = "<eos>"
    mock_tok_none.chat_template = None

    transform_none = gemma_text.GemmaTextTransform(
        processor=mock_tok_none, max_length=128, cfg=None
    )
    transform_none._post_load_processor()
    self.assertEqual(mock_tok_none.pad_token, "<eos>")

    # Case 2: pad_token is already set
    mock_tok_set = mock.Mock()
    mock_tok_set.pad_token = "<pad>"
    mock_tok_set.eos_token = "<eos>"
    mock_tok_set.chat_template = None

    transform_set = gemma_text.GemmaTextTransform(
        processor=mock_tok_set, max_length=128, cfg=None
    )
    transform_set._post_load_processor()
    self.assertEqual(mock_tok_set.pad_token, "<pad>")

  def test_get_transform_fn_saves_processor_locally(self):
    """Verifies processor.save_pretrained is called for DataLoader worker pre-caching."""
    mock_tok = mock.create_autospec(
        transformers.PreTrainedTokenizer, instance=True
    )
    mock_tok.name_or_path = "mock-id"

    cfg = ml_collections.ConfigDict(
        {"model_flavor": config.ModelFlavor.GEMMA_3_TEXT, "max_seq_length": 64}
    )

    transform = self.engine.get_transform_fn(
        mock_tok, [], [], {}, cfg=cfg,
    )

    mock_tok.save_pretrained.assert_called_once()
    self.assertIsInstance(transform, gemma_text.GemmaTextTransform)


if __name__ == "__main__":
  absltest.main()
