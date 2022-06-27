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

# pylint: skip-file
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationPruner(nn.Module):
    def __init__(self, dim, num_relation):
        super(RelationPruner, self).__init__()
        self.dim = dim
        self.num_relation = num_relation
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)
        self.layer3 = nn.Linear(self.dim, self.num_relation)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)

    def forward(self, embedding):
        x = F.relu(self.layer1(embedding))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x

class BranchPruner(nn.Module):
    def __init__(self, dim, aggr=torch.max):
        super(BranchPruner, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)
        self.layer3 = nn.Linear(self.dim, self.dim)
        self.layer4 = nn.Linear(self.dim, 1)
        self.aggr = aggr

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)

    def forward(self, embeddings):
        x = F.relu(self.layer1(embeddings))
        x = F.relu(self.layer2(x))
        if self.aggr in [torch.max, torch.min]:
            x = self.aggr(x, dim=0)[0]
        elif self.aggr in [torch.mean, torch.sum]:
            x = self.aggr(x, dim=0)
        x = F.relu(self.layer3(x))
        x = self.layer4(x)

        return x


