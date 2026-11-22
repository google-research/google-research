# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""The Attention based head module to attend each video representations to all cluster representations.
"""

import torch
from torch import nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
  """Attention head.

  The detailed attention operation follows the one used in "Neural Machine
  Translation by Jointly Learning to Align and Translate"
  (https://arxiv.org/abs/1409.0473) and "Pointer Networks"
  (https://arxiv.org/abs/1506.03134)
  """

  def __init__(self, num_hidden):
    super().__init__()
    self.num_hidden = num_hidden
    self.w1 = nn.Linear(num_hidden, num_hidden, bias=False)
    self.w2 = nn.Linear(num_hidden, num_hidden, bias=False)
    self.v = nn.Linear(num_hidden, 1, bias=False)  # working as dot product

  def forward(self, x_query, x_key):
    """Forward pass.

    Args:
      x_query: Query vectors (video representations).
      x_key: Key vectors (cluster representations).
    Shape:
      x_query: (B, N_q, num_hidden)
      x_key: (B, N_k, num_hidden)

    Returns:
      The normalized attention scores (log_softmax) with shape (B, N_q, N_k).
    """
    # (B, N_q, N_k, C) <- (B, N_k, C)
    key_transform = self.w1(x_key).unsqueeze(1).expand(-1, x_query.shape[1], -1,
                                                       -1)
    # (B, N_q, 1, C) <- (B, N_q, C)
    query_transform = self.w2(x_query).unsqueeze(2)
    # (B, N_q, N_k) <- (B, N_q, N_k, C), (B, N_q, 1, C)
    prod = self.v(torch.tanh(key_transform + query_transform)).squeeze(-1)
    return F.log_softmax(prod, dim=-1)
