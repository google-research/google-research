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

import os, sys
import subprocess
import flashtool


def replace(script, key, value):
  key = str(key)
  value = str(value)
  startid = script.find("\n" + key + "=")
  startid += 1
  assert script[startid - 1] == "\n"
  endid = script.find("\n", startid)
  print("Replacing|{}|==>|{}|".format(script[startid:endid], key + "=" + value))
  newscript = script[:startid] + key + "=" + value + script[endid:]
  return newscript


def run_new_script(basescript, replace_dict):
  print("==================BEGIN===================")
  try:
    _script = replace(basescript, "PROJECT_ROOT", os.path.abspath("."))
    for k, v in replace_dict.items():
      _script = replace(_script, k, v)
    subprocess.run(_script, shell=True, cwd=os.path.abspath("."))
  except:
    print("==================FAILED==================")
    print(replace_dict)
  print("====================END===================")
  print("\n\n")


def main():
  basescript = "script/syn_air_ope.sh"
  with open(basescript, mode="r") as f:
    basescript = f.read()

  flashtool.trackpid(10067, 5)
  basic_config = {
      "FIX_BERT": "true",
      "SHARE_BERT": "true",
      "FREEZE_BERT": "true",
      "alphaR": "0",
      "alphaQ": "0",
      "alphaC": "1",
      "alphaL": "0",
      "regfunQ": "square_cut5",
      "actC": "square",
      "actQ": "no",
      "OPT": "sgd",
      "LR": "2e-2",
      "SGD_MOM": "0.5",
      "WEIGHT_DECAY": "1e-4",
      "LR_BERT": "1",
      "LR_LAMB": "10",
      "LR_Q": "5",
      "EPOCH": 500,
      "LR_SCHEDULE": "linear",
      "LOG_STEP": "7",
      "TRAIN_BATCH": 48,
      "MAXNORM": "1",
      "lambinit": "0",
      "TAG": "\"\"",
      "GPUID": "0,1,2,3,4,5,6,7",
  }

  #basic_config["DATA"] = "syn_ope_data_500/tgt_L5"
  #run_new_script(basescript, basic_config)
  basic_config["DATA"] = "syn_ope_data_500/tgt_L4"
  run_new_script(basescript, basic_config)
  #basic_config["DATA"] = "syn_ope_data_500/tgt_L3"
  #run_new_script(basescript, basic_config)
  basic_config["DATA"] = "syn_ope_data_500/tgt_L2"
  run_new_script(basescript, basic_config)
  basic_config["DATA"] = "syn_ope_data_500/tgt_L1"
  run_new_script(basescript, basic_config)
  #basic_config["DATA"] = "syn_ope_data_500/tgt_L0"
  #run_new_script(basescript, basic_config)


if __name__ == "__main__":
  main()
