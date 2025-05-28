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

"""From config file, generates a random ordering of pretraining/finetuning strategies."""

import os
import numpy as np
from sklearn.model_selection import ParameterGrid
import strategy_config


pt_strategies_list = []
strategies_list = []

for args_dict in list(ParameterGrid(strategy_config.sweep_dict_list)):
  # Fixing args_dict + getting save strings

  if args_dict["PT_task"] == (0, 0):
    for key in [
        "PT_recon_decoder",
        "aug_jitter_std",
        "aug_masksampling",
        "aug_maskrate",
        "aug_maskpart",
        "aug_maskval",
    ]:
      args_dict[key] = "NA"
    if args_dict[key] == "same":
      args_dict[key] = "none"

  elif args_dict["PT_task"][1] == 0:
    args_dict["PT_recon_decoder"] = "NA"

  if args_dict["aug_maskrate"] == 0:
    args_dict["aug_masksampling"] = "NA"
    args_dict["aug_maskpart"] = "NA"
    args_dict["aug_maskval"] = "NA"

  PT_save_string_list = [
      v + "-" + str(args_dict[v]) for v in strategy_config.PT_args_to_save
  ]
  pt_save_string = "~".join(PT_save_string_list)

  FT_save_string_list = [
      v + "-" + str(args_dict[v]) for v in strategy_config.FT_args_to_save
  ]
  ft_save_string = "~".join(FT_save_string_list)

  strategies_list.append(ft_save_string)
  pt_strategies_list.append(pt_save_string)


# Get random ordering

unique_strategies, idx_for_pt = np.unique(strategies_list, return_index=True)

random_order = np.random.choice(
    len(unique_strategies), len(unique_strategies), replace=False
)

savedir = strategy_config.static_args_dict["results_dir"]
if not os.path.isdir(savedir):
  os.makedirs(savedir)
np.savetxt(
    os.path.join(savedir, "STRATEGIES_LIST.txt"),
    unique_strategies[random_order],
    fmt="%s",
    delimiter="\t",
)

print("SAVED RANDOM ORDERING TO:")
print(os.path.join(savedir, "STRATEGIES_LIST.txt"))
