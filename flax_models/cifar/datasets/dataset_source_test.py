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

# Lint as: python3
"""Tests for flax_cifar.datasets.dataset_source."""


from absl.testing import absltest
from absl.testing import parameterized

from flax_models.cifar.datasets import dataset_source


class DatasetSourceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('none', 'none'),
      ('cutout', 'cutout'))
  def test_LoadCifar10(self, batch_level_augmentation):
    cifar_10_source = dataset_source.Cifar10(
        2,
        image_level_augmentations='autoaugment',
        batch_level_augmentations=batch_level_augmentation)
    for batch in cifar_10_source.get_train(use_augmentations=True):
      self.assertEqual(batch['image'].shape, [2, 32, 32, 3])
      self.assertEqual(batch['label'].shape, [2, 10])
      break
    for batch in cifar_10_source.get_test():
      self.assertEqual(batch['image'].shape, [2, 32, 32, 3])
      self.assertEqual(batch['label'].shape, [2, 10])
      break

  @parameterized.named_parameters(
      ('none', 'none'),
      ('cutout', 'cutout'))
  def test_LoadCifar100(self, batch_level_augmentation):
    cifar_100_source = dataset_source.Cifar100(
        2,
        image_level_augmentations='autoaugment',
        batch_level_augmentations=batch_level_augmentation)
    for batch in cifar_100_source.get_train(use_augmentations=True):
      self.assertEqual(batch['image'].shape, [2, 32, 32, 3])
      self.assertEqual(batch['label'].shape, [2, 100])
      break
    for batch in cifar_100_source.get_test():
      self.assertEqual(batch['image'].shape, [2, 32, 32, 3])
      self.assertEqual(batch['label'].shape, [2, 100])
      break

  @parameterized.named_parameters(
      ('none', 'none'),
      ('cutout', 'cutout'))
  def test_LoadFashionMnist(self, batch_level_augmentation):
    fashion_mnist_source = dataset_source.FashionMnist(
        2,
        image_level_augmentations='basic',
        batch_level_augmentations=batch_level_augmentation)
    for batch in fashion_mnist_source.get_train(use_augmentations=True):
      self.assertEqual(batch['image'].shape, [2, 28, 28, 1])
      self.assertEqual(batch['label'].shape, [2, 10])
      break
    for batch in fashion_mnist_source.get_test():
      self.assertEqual(batch['image'].shape, [2, 28, 28, 1])
      self.assertEqual(batch['label'].shape, [2, 10])
      break


if __name__ == '__main__':
  absltest.main()
