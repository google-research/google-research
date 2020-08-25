# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
  basescript = "script/selfplay_air_ope.sh"
  with open(basescript, mode="r") as f:
    basescript = f.read()

  #flashtool.trackpid(10067,5)
  basic_config = {
      "alphaAux": 0,
  }
  ALL_SETTINGS = [
      "20K",
      "50K",
      "5K",
      "10K",
      "30K",
      "40K",
      "75K",
      "100K",
      "150K",
      "200K",
      "250K",
      "full",
      "20K_w",
      "50K_w",
      "5K_w",
      "10K_w",
      "30K_w",
      "40K_w",
      "75K_w",
      "100K_w",
      "150K_w",
      "200K_w",
      "250K_w",
      "full_w",
  ]
  for model in ALL_SETTINGS:
    basic_config["DATA"] = model
    run_new_script(basescript, basic_config)


if __name__ == "__main__":
  main()
