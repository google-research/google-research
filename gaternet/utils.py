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

"""Utils for building models."""

import torch.nn as nn


def init_weights(module):
  if isinstance(module, nn.Linear):
    module.bias.data.zero_()
  elif isinstance(module, nn.Conv2d):
    nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
  elif isinstance(module, nn.BatchNorm2d):
    module.bias.data.zero_()
    module.weight.data.fill_(1)
