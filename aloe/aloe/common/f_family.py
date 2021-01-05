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

# pylint: skip-file
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from aloe.common.pytorch_util import MLP, NONLINEARITIES


class MLPScore(nn.Module):
    def __init__(self, input_dim, hidden_dims, scale=1.0, nonlinearity='elu', act_last=None, bn=False, dropout=-1, bound=-1):
        super(MLPScore, self).__init__()
        self.scale = scale
        self.bound = bound
        self.mlp = MLP(input_dim, hidden_dims, nonlinearity, act_last, bn, dropout)

    def forward(self, z):
        raw_score = self.mlp(z.float() / self.scale)
        if self.bound > 0:
            raw_score = torch.clamp(raw_score, min=-self.bound, max=self.bound)
        return raw_score
