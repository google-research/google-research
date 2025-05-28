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

"""Evaluation of Koopman Neural Operator."""
import os

from modules.eval_metrics import RMSE
from modules.eval_metrics import SMAPE
from modules.eval_metrics import WRMSE
import numpy as np
import torch

# Calculate the sMAPE of ensembled predictions on M4
direc_m4 = "M4_results/"
files_m4 = os.listdir(direc_m4)
ems_preds_m4 = []
for f in files_m4:
  if "test_" in f and "Weekly" in f:
    results_m4 = torch.load(direc_m4 + f)
    ems_preds_m4.append(results_m4["test_preds"])
ems_preds_m4 = np.stack(ems_preds_m4).mean(0)
m4_smape = SMAPE(ems_preds_m4, results_m4["test_tgts"])
print("M4-Weekly SMAPE: %5.3f" % m4_smape[-1])

# Calculate the mean and std of RMSE of five runs on Traj dataset
direc_traj = "Traj_results/"
files_traj = os.listdir(direc_traj)
lst_rmse_traj = []
for f in files_traj:
  if "test_" in f:
    results = torch.load(direc_traj + f)
    lst_rmse_traj.append(RMSE(results["test_preds"], results["test_tgts"]))
traj_rmse = np.concatenate(
    [
        np.mean(lst_rmse_traj, axis=0, keepdims=True),
        np.std(lst_rmse_traj, axis=0, keepdims=True),
    ],
    axis=0,
)
traj_rmse = tuple(traj_rmse.T.reshape(-1))
print("Traj RMSE: %2.2f ± %0.2f" % traj_rmse[-2:])

# Calculate the mean and std of Weighted RMSE of five runs on Cryptos dataset
direc_cryptos = "Cryptos_results/"
files_cryptos = os.listdir(direc_cryptos)
lst_wrmse_cryptos = []
for f in files_cryptos:
  if "test_" in f:
    results = torch.load(direc_cryptos + f)
    lst_wrmse_cryptos.append(WRMSE(results["test_preds"], results["test_tgts"]))
cryptos_wrmse = np.concatenate(
    [
        np.mean(lst_wrmse_cryptos, axis=0, keepdims=True),
        np.std(lst_wrmse_cryptos, axis=0, keepdims=True),
    ],
    axis=0,
)
cryptos_wrmse = tuple(cryptos_wrmse.T.reshape(-1))
print("Cryptos Weighted RMSE: %2.5f ± %0.5f" % cryptos_wrmse[-2:])
