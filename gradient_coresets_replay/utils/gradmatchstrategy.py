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

"""Gradmatch strategy for subset selection."""

import time
import numpy as np
import torch
from torch.utils import data
from gradient_coresets_replay.utils.dataselectionstrategy import DataSelectionStrategy
from gradient_coresets_replay.utils.helpers.omp_solvers import orthogonalmp_reg
from gradient_coresets_replay.utils.helpers.omp_solvers import orthogonalmp_reg_parallel


class OMPGradMatchStrategy(DataSelectionStrategy):
  """Gradmatch strategy for subset selection."""

  name = 'gradmatch'

  def __init__(
      self,
      model,
      trainloader,
      valloader,
      num_classes,
      inner_loss,
      outer_loss,
      device,
  ):
    super().__init__(
        model,
        trainloader,
        valloader,
        num_classes,
        inner_loss,
        outer_loss,
        device,
    )
    self.eta = 0.03
    self.init_out = list()
    self.init_l1 = list()
    self.selection_type = 'PerClass'
    self.valid = False
    self.lam = 0.5
    self.eps = 1e-100
    self.linear_layer = True

  def ompwrapper(self, x, y, bud):
    if self.device == 'cpu':
      reg = orthogonalmp_reg(
          x.cpu().numpy(), y.cpu().numpy(), nnz=bud, positive=True, lam=0
      )
      ind = np.nonzero(reg)[0]
    else:
      reg = orthogonalmp_reg_parallel(
          x,
          y,
          nnz=bud,
          positive=True,
          lam=self.lam,
          tol=self.eps,
          device=self.device,
      )
      ind = torch.nonzero(reg).view(-1)
    return ind.tolist(), reg[ind].tolist()

  def select(self, budget, model_params):
    omp_start_time = time.time()
    self.update_model(model_params)
    idxs = []
    gammas = []
    if self.selection_type == 'PerClass':
      self.get_labels(valid=self.valid)
      self.eff_classes = torch.unique(self.trn_lbls)
      self.eff_num_classes = self.eff_classes.shape[0]
      # budget_per_class = math.ceil(budget/self.eff_num_classes)
      budget_per_class = self.get_budget_per_class(self.trn_lbls, budget)
      print(f'num_classes={self.eff_num_classes}')
      for i in self.eff_classes:
        self.update_model(model_params)
        trn_subset_idx = torch.where(self.trn_lbls == i)[0].tolist()
        trn_data_sub = data.Subset(self.trainloader.dataset, trn_subset_idx)
        self.pctrainloader = data.DataLoader(
            trn_data_sub, batch_size=self.trainloader.batch_size, shuffle=False
        )
        if self.valid:
          val_subset_idx = torch.where(self.val_lbls == i)[0].tolist()
          val_data_sub = data.Subset(self.valloader.dataset, val_subset_idx)
          self.pcvalloader = data.DataLoader(
              val_data_sub,
              batch_size=self.trainloader.batch_size,
              shuffle=False,
          )

        self.compute_gradients(self.valid, batch=False, perClass=True)
        trn_gradients = self.grads_per_elem
        if self.valid:
          sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
        else:
          sum_val_grad = torch.sum(trn_gradients, dim=0)
        idxs_temp, gammas_temp = self.ompwrapper(
            torch.transpose(trn_gradients, 0, 1),
            sum_val_grad,
            budget_per_class[i.item()],
        )
        if len(idxs_temp) < budget_per_class[i.item()]:
          remain_list = set(np.arange(len(trn_subset_idx))).difference(
              set(idxs_temp)
          )
          new_idxs_temp = np.random.choice(
              list(remain_list),
              size=budget_per_class[i.item()] - len(idxs_temp),
              replace=False,
          )
          gammas_temp.extend(
              [1 for _ in range(budget_per_class[i.item()] - len(idxs_temp))]
          )
          idxs_temp.extend(new_idxs_temp)
        idxs.extend(list(np.array(trn_subset_idx)[idxs_temp]))
        gammas.extend(gammas_temp)

    omp_end_time = time.time()
    diff = budget - len(idxs)
    print(f'difference={diff}')

    if diff > 0:
      remain_list = set(np.arange(self.N_trn)).difference(set(idxs))
      new_idxs = np.random.choice(list(remain_list), size=diff, replace=False)
      idxs.extend(new_idxs)
      gammas.extend([1 for _ in range(diff)])
      idxs = np.array(idxs)
      gammas = np.array(gammas)

    rand_indices = np.random.permutation(len(idxs))
    idxs = list(np.array(idxs)[rand_indices])
    gammas = list(np.array(gammas)[rand_indices])

    if diff < 0:
      indices = np.argsort(gammas)[:budget]
      indices = indices[np.random.permutation(len(indices))]
      idxs = list(np.array(idxs)[indices])
      gammas = list(np.array(gammas)[indices])

    print(
        'OMP algorithm Subset Selection time is: ',
        omp_end_time - omp_start_time,
    )
    return idxs, gammas
