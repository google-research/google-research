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

"""Favorita embedder."""

import pickle

from data_formatting import base_data_formatter as base
import numpy as np
import torch
from torch import nn


DataTypes = base.DataTypes
InputTypes = base.InputTypes


class FavoritaEmbedder(nn.Module):
  """Favorita embedder.

  Embeds categorical variables, and scale numerical variables
  """

  def __init__(self, embedding_dim=50, device=torch.device('cpu')):
    super().__init__()
    formatter = pickle.load(open('../data/favorita/data_formatter.pkl', 'rb'))
    self.colnames = np.load('../data/favorita/favorita_full_pivoted_vars.npy',
                            allow_pickle=True)

    self.column_definition = formatter.get_column_definition()
    self.embedding_dim = embedding_dim

    self.categorical_vars = []
    self.numerical_vars = []
    self.vars = []
    self.category_counts = pickle.load(
        open('../data/favorita/favorita_full_category_counts.pkl', 'rb')
    )
    self.embedders = nn.ModuleList()
    for col in self.column_definition:
      if col[2] in [InputTypes.ID, InputTypes.TIME]:
        continue
      elif col[1] == DataTypes.CATEGORICAL:  # categorical
        self.categorical_vars.append(col[0])
        self.vars.append(col[0])
        num_cats = self.category_counts[col[0]] + 1
        self.embedders.append(
            nn.Embedding(num_cats, embedding_dim).float().to(device)
        )
      elif col[1] == DataTypes.REAL_VALUED:  # numerical
        self.numerical_vars.append(col[0])
        self.vars.append(col[0])
        self.embedders.append(nn.Linear(1, embedding_dim).float().to(device))
      else:
        print('unexpected datatype: ', col)

    for j in range(len(self.embedders)):
      assert self.colnames[j] == self.vars[j]
      assert self.column_definition[j + 2][0] == self.colnames[j]

  def forward(self, x):  # assumes second dim is covariate
    assert len(x.shape) == 3  # N, T, D
    embeddings = []
    for j, embedder in enumerate(self.embedders):
      x_input = x[:, :, j].unsqueeze(-1)
      if 'embedding' in str(embedder).lower():
        x_input = torch.clamp(
            x_input, max=self.category_counts[self.vars[j]] - 1
        ).int()
        embedding = embedder(x_input)
        embedding = embedding.squeeze(
            -2
        )  # note: assumes embedding size is not 1
      else:
        embedding = embedder(x_input)
      embeddings.append(embedding)
    embeddings = torch.stack(embeddings, axis=1)
    return embeddings
