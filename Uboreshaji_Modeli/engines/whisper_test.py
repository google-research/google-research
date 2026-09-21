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
import transformers

from Uboreshaji_Modeli.engines import whisper


class WhisperEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.engine = whisper.WhisperEngine()

  def test_get_sft_config_overrides(self):
    cfg = ml_collections.ConfigDict()
    overrides = self.engine.get_sft_config_overrides(cfg)
    self.assertEqual(
        overrides,
        {"dataset_kwargs": {"skip_prepare_dataset": True}},
    )

  def test_get_collate_fn(self):
    mock_processor = mock.create_autospec(
        transformers.WhisperProcessor, instance=True
    )
    collate_fn = self.engine.get_collate_fn(processor=mock_processor)
    self.assertIsNotNone(collate_fn)

  def test_get_transform_fn(self):
    mock_processor = mock.create_autospec(
        transformers.WhisperProcessor, instance=True
    )
    transform_fn = self.engine.get_transform_fn(
        processor=mock_processor,
        text_inputs=[],
        dataset_id2label=[],
        model_label2id={},
    )
    self.assertIsInstance(transform_fn, whisper.WhisperTransform)
    mock_processor.save_pretrained.assert_called_once()
    args, _ = mock_processor.save_pretrained.call_args
    self.assertIn("local_whisper_processor_", args[0])

  def test_whisper_transform_explicit_model_id(self):
    """Verifies explicit model_id is preferred over processor.name_or_path."""

    class DummyProcessor:
      name_or_path = "openai/whisper-tiny"

    dummy_processor = DummyProcessor()

    transform = whisper.WhisperTransform(
        processor=dummy_processor,
        cfg=None,
        model_id="/tmp/explicit_model_id",
    )

    self.assertEqual(transform._model_id, "/tmp/explicit_model_id")

  def test_whisper_transform_pickling(self):
    import copy  # pylint: disable=g-import-not-at-top

    class DummyProcessor:
      name_or_path = "openai/whisper-tiny"

    dummy_processor = DummyProcessor()

    transform = whisper.WhisperTransform(
        processor=dummy_processor,
        cfg=None,
    )

    # Test __getstate__
    state = transform.__getstate__()
    self.assertIsNone(state["processor"])

    with mock.patch.object(
        transformers.WhisperProcessor, "from_pretrained", autospec=True
    ) as mock_from_pretrained, mock.patch.object(
        whisper.WhisperTransform, "_post_load_processor", autospec=True
    ) as mock_post_load:
      class DummyReloadedProcessor:
        pass

      reloaded_processor = DummyReloadedProcessor()
      mock_from_pretrained.return_value = reloaded_processor

      cloned = copy.deepcopy(transform)

      mock_from_pretrained.assert_called_once_with("openai/whisper-tiny")
      self.assertEqual(cloned.processor, reloaded_processor)
      mock_post_load.assert_called_once()


if __name__ == "__main__":
  absltest.main()
