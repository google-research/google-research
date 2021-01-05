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
"""VisionDataset defined as a subsect of a different dataset."""

import torchvision


class VisionDatasetWithSelection(torchvision.datasets.vision.VisionDataset):
  """VisionDataset defined as a subsect of a different dataset."""

  def __init__(self, dataset, indices):
    """Constructs a new dataset.

    Args:
        dataset (VisionDataset): dataset to choose subset from
        indices (list): list of indices from the original dataset
    """
    super(VisionDatasetWithSelection, self).__init__(None)
    self.dataset = dataset
    self.indices = indices

  def __getitem__(self, index):
    real_index = self.indices[index]
    return self.dataset.__getitem__(real_index)

  def __len__(self):
    return len(self.indices)
