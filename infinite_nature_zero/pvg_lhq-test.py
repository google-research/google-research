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
"""Inference code for running InfiniteNature-Zero."""
import os

from data.nature_data_loader import LHQGANDataLoader
from models.render_model import RenderModel
from options.test_options import TestOptions
import torch

args = TestOptions().parse()

test_path_list = [
    './test-data/images/0000003.png', './test-data/images/0000190.png',
    './test-data/images/0001337.png'
]

lerp = 0.05  # interpolation factor
sky_fraction = 0.1  # desired fraction of sky content
near_fraction = 0.3  # desired fraction of near content

batch_size = 1
train_num_threads = 1
inference_data_loader = LHQGANDataLoader(args,
                                         batch_size,
                                         test_path_list,
                                         'test',
                                         train_num_threads)

inference_dataset = inference_data_loader.load_data()

model = RenderModel(args, is_train=False)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

for i, data in enumerate(inference_dataset):
  index = int(data['index'][0].item())
  sky_ratio = data['sky_ratio'][0].item()

  with torch.no_grad():

    save_dir = os.path.join('release-test-outputs', '%07d' % index)
    os.makedirs(save_dir, exist_ok=True)
    return_value = model.view_generation(
        data,
        save_dir,
        sky_fraction=sky_fraction,
        near_fraction=near_fraction,
        lerp=lerp,
        num_steps=100)

    if not return_value:
      print('Auto-pilot crashed into an obstacle. Please rerun the code.')

