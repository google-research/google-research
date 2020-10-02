# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""VisionDataset with potentially corrupted labels."""

import numpy as np
import torchvision


class VisionDatasetWithCorruption(torchvision.datasets.vision.VisionDataset):
  """VisionDataset with potentially corrupted labels."""

  def __init__(self, dataset, n_classes, corrupt_fraction, rng):
    """Constructs new dataset with potentially corrupted labels.

    Args:
      dataset (VisionDataset): dataset to corrupt
      n_classes (int): number of classes
      corrupt_fraction (float): number from [0, 1] indicating what fraction
                                of the data should be corrupted
      rng (numpy.random.RandomState): seeded random number generator
    """
    super(VisionDatasetWithCorruption, self).__init__(None)
    size = len(dataset)

    should_corrupt = rng.uniform(0, 1, size=size) < corrupt_fraction
    new_targets = rng.randint(0, n_classes, (size,))
    self.original_targets = dataset.targets
    self.targets = np.where(should_corrupt, new_targets, self.original_targets)
    self.dataset = dataset

  def __getitem__(self, index):
    data, _ = self.dataset.__getitem__(index)
    return data, self.targets[index]

  def __len__(self):
    return len(self.dataset)

  def is_corrupt(self, index):
    return self.original_targets[index] != self.targets[index]
