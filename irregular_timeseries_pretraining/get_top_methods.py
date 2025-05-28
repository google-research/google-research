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

"""Checks validation results from trained models and orders them by validation performance."""

import os
import sys
import pandas as pd
import strategy_config

dset = sys.argv[1]

downsampled_frac = strategy_config.static_args_dict["downsampled_frac"]
if downsampled_frac:
  data_name_save = dset + "-" + str(downsampled_frac)
else:
  data_name_save = dset

savedir = os.path.join(
    strategy_config.static_args_dict["results_dir"], data_name_save
)
val_res_path = os.path.join(
    savedir, "finetuning-1val", "SUPERVISED_LOGS_EPOCHS"
)


# Loop through all validation results and get max validation AUROC and AUPRC
val_pr_aucs = []
val_roc_aucs = []
finished_methods = []

for i, elt in enumerate(os.listdir(val_res_path)):
  sup_file = os.path.join(val_res_path, elt)
  if ".csv" in sup_file:
    val_res = pd.read_csv(
        sup_file,
        skiprows=1,
        names=["tr_pr_auc", "tr_roc_auc", "val_pr_auc", "val_roc_auc"],
    )
    val_pr_aucs.append(val_res["val_pr_auc"].max())
    val_roc_aucs.append(val_res["val_roc_auc"].max())

    finished_methods.append(elt[:-4])


# Sort strategies by validation performance -> save to results folder
strategies_df = pd.DataFrame(finished_methods, columns=["strategy"])
strategies_df["val_pr_auc"] = val_pr_aucs
strategies_df["val_roc_auc"] = val_roc_aucs
strategies_df["val_pr_roc_sum"] = (
    strategies_df["val_pr_auc"] + strategies_df["val_roc_auc"]
)

strategies_df.to_csv(os.path.join(savedir, "val_sorted_methods.csv"))

print("saved sorted strategies by validation performance to:")
print(os.path.join(savedir, "val_sorted_methods.csv"))
