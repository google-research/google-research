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
from absl.testing import parameterized
import ml_collections

from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.engines import base
from Uboreshaji_Modeli.engines import factory
from Uboreshaji_Modeli.engines import owl


class EngineFactoryTest(parameterized.TestCase):

  def test_get_owl_v2_engine_composition(self):
    engine = factory.get_engine(config.ModelFlavor.OWL_V2_TORCH)

    with self.subTest("EngineInstance"):
      self.assertIsInstance(engine, base.ModelEngine)
      self.assertIsInstance(engine, owl.Owlv2Engine)

    with self.subTest("PreprocessorInstance"):
      self.assertIsInstance(engine.preprocessor, base.DataPreprocessor)
      self.assertIsInstance(engine.preprocessor, owl.Owlv2Preprocessor)

    with self.subTest("LossHandlerInstance"):
      self.assertIsInstance(engine.loss_handler, base.LossHandler)
      self.assertIsInstance(engine.loss_handler, owl.Owlv2LossHandler)

    with self.subTest("DecoderInstance"):
      # Owlv2 doesn't have a separate composed decoder yet.
      self.assertIsNone(engine.decoder)

  def test_get_transform_fn_delegation_with_cfg(self):
    engine = factory.get_engine(config.ModelFlavor.OWL_V2_TORCH)
    mock_preprocessor = mock.create_autospec(
        base.DataPreprocessor, instance=True
    )
    engine.preprocessor = mock_preprocessor

    cfg = ml_collections.ConfigDict()
    engine.get_transform_fn(
        processor="dummy_processor",
        text_inputs=["a"],
        dataset_id2label=["b"],
        model_label2id={"c": 1},
        cfg=cfg,
        is_train=True,
        extra_kwarg="val",
    )

    mock_preprocessor.get_transform_fn.assert_called_once_with(
        "dummy_processor",
        cfg,
        is_train=True,
        text_inputs=["a"],
        dataset_id2label=["b"],
        model_label2id={"c": 1},
        extra_kwarg="val",
    )


if __name__ == "__main__":
  absltest.main()
