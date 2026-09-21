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

from Uboreshaji_Modeli.engines import mms


class MmsEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.engine = mms.MmsEngine()

  def test_get_sft_config_overrides(self):
    cfg = ml_collections.ConfigDict()
    overrides = self.engine.get_sft_config_overrides(cfg)
    self.assertEqual(
        overrides,
        {"dataset_kwargs": {"skip_prepare_dataset": True}},
    )

  def test_get_collate_fn(self):
    mock_processor = mock.create_autospec(
        transformers.Wav2Vec2Processor, instance=True
    )
    collate_fn = self.engine.get_collate_fn(processor=mock_processor)
    self.assertIsNotNone(collate_fn)

  def test_get_transform_fn(self):
    class DummyProcessor:

      def __init__(self):
        self.saved_path = None

      def save_pretrained(self, path):
        self.saved_path = path

    dummy_processor = DummyProcessor()

    transform_fn = self.engine.get_transform_fn(
        processor=dummy_processor,
        text_inputs=[],
        dataset_id2label=[],
        model_label2id={},
    )
    self.assertIsInstance(transform_fn, mms.MmsTransform)
    self.assertIsNotNone(dummy_processor.saved_path)

  def test_mms_transform_explicit_model_id(self):
    """Verifies explicit model_id is preferred over processor.name_or_path."""

    class DummyProcessor:
      name_or_path = "facebook/mms-300m"

    dummy_processor = DummyProcessor()

    transform = mms.MmsTransform(
        processor=dummy_processor,
        model_id="/tmp/explicit_model_id",
    )

    self.assertEqual(transform._model_id, "/tmp/explicit_model_id")

  def test_mms_transform_pickling(self):
    import copy  # pylint: disable=g-import-not-at-top

    class DummyProcessor:
      name_or_path = "facebook/mms-300m"

    dummy_processor = DummyProcessor()

    transform = mms.MmsTransform(
        processor=dummy_processor,
    )

    # Test __getstate__
    state = transform.__getstate__()
    self.assertIsNone(state["processor"])

    with mock.patch.object(
        transformers.AutoProcessor, "from_pretrained", autospec=True
    ) as mock_from_pretrained:
      class DummyReloadedProcessor:
        pass

      reloaded_processor = DummyReloadedProcessor()
      mock_from_pretrained.return_value = reloaded_processor

      cloned = copy.deepcopy(transform)

      mock_from_pretrained.assert_called_once_with("facebook/mms-300m")
      self.assertEqual(cloned.processor, reloaded_processor)


if __name__ == "__main__":
  absltest.main()
