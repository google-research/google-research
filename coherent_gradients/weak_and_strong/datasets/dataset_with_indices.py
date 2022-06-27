# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""VisionDataset that allows user to get example indices."""

import torchvision


class VisionDatasetWithIndices(torchvision.datasets.vision.VisionDataset):
  """VisionDataset that allows user to get example indices.

  Dataset that returns a triple (data, targets, indices)
  instead of just (data, targets). Indices of training examples can be
  used to track model performance on individual examples, for instance to find
  training examples that are learned faster than others.
  """

  def __init__(self, dataset):
    super(VisionDatasetWithIndices, self).__init__(None)
    self.dataset = dataset

  def __getitem__(self, index):
    data, target = self.dataset.__getitem__(index)
    return data, target, index

  def __len__(self):
    return len(self.dataset)
