# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

# -*- coding: utf-8 -*-
"""Class definition for LHQ loader."""
from data.lhq_folder import LHQTestDataset
import torch.utils.data


class LHQGANDataLoader():
  """LHQ GAN Data Loader."""

  def __init__(self, args, batch_size, data_list, phase, num_threads):
    dataset = LHQTestDataset(opt=args, data_list=data_list, phase=phase)
    self.train_sampler = None
    self.data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_threads),
        pin_memory=True,
        drop_last=True,
        sampler=self.train_sampler)
    self.dataset = dataset

  def load_data(self):
    return self.data_loader

  def get_sampler(self):
    return self.train_sampler

  def name(self):
    return 'LHQDataLoader'

  def __len__(self):
    return len(self.dataset)
