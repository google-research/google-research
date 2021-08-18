# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Input pipeline tests."""

from absl.testing import parameterized
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

from spin_spherical_cnns import input_pipeline
from spin_spherical_cnns.configs import default


class InputPipelineTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters("spherical_mnist/rotated",
                            "spherical_mnist/canonical")
  def test_create_datasets_spherical_mnist(self, dataset):
    rng = jax.random.PRNGKey(42)
    config = default.get_config()
    config.dataset = dataset
    config.per_device_batch_size = 8
    config.eval_pad_last_batch = False
    dataset_loaded = False
    if not dataset_loaded:
      splits = input_pipeline.create_datasets(config, rng)
    self.assertEqual(splits.info.features["label"].num_classes, 10)
    self.assertEqual(splits.train.element_spec["input"].shape,
                     (1, 8, 64, 64, 1, 1))
    self.assertEqual(splits.train.element_spec["label"].shape, (1, 8))
    self.assertEqual(splits.validation.element_spec["input"].shape,
                     (1, 8, 64, 64, 1, 1))
    self.assertEqual(splits.validation.element_spec["label"].shape, (1, 8))
    self.assertEqual(splits.test.element_spec["input"].shape,
                     (1, 8, 64, 64, 1, 1))
    self.assertEqual(splits.test.element_spec["label"].shape, (1, 8))


if __name__ == "__main__":
  tf.test.main()
