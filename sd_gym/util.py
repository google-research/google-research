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

"""Handy utilities."""

from typing import Union
from BPTK_Py.sdcompiler.parsers.xmile.xmile import parse_xmile
import pandas as pd

from sd_gym import env


def sd_info(model_identifier):
  """Generates a dataframe of info about the sd model.

  Parses the file using BPTK.

  Args:
    model_identifier: Can be a model file location or an SDEnv

  Returns:
    a parsed model dict
    pd.DataFrame of information about the model or env.
  """
  def case(var):
    return "".join(["_"+i.lower() if i.isupper() else i
                    for i in var]).lstrip("_")

  if isinstance(model_identifier, env.SDEnv):
    file = model_identifier.params.sd_file
  elif isinstance(model_identifier, str):
    file = model_identifier
  else:
    raise ValueError("`model_identifier` argument must be `SDEnv` or `str`")

  # Parse file to access equations and variable types
  parsed_info = dict()
  parsed_file = parse_xmile(file)
  parsed_model = list(parsed_file["models"].values())[0]
  parsed_model_entities = parsed_model["entities"]
  for var_type in parsed_model_entities.keys():
    for var_dict in parsed_model_entities[var_type]:
      var = var_dict["name"]
      if not var_dict["equation"]:
        continue
      if var_type == "stock":
        equ = "d/dt = " + " + ".join(var_dict["inflow"])
        if var_dict["outflow"]:
          equ += " - " + " - ".join(var_dict["outflow"])
      else:
        equ = var_dict["equation"][0]
      parsed_info[var] = [var_type, equ]

  # Make a data frame with the info
  info_dict = {
      "Variable Name": [],
      "Observable": [],
      "Actionable": [],
      "Type": [],
      "Equation": [],
  }
  all_model_vars = list(parsed_info.keys())
  for var in all_model_vars:
    var_type, equ = parsed_info[var]
    # Save the variable name, type and equation
    info_dict["Variable Name"].append(var)
    info_dict["Type"].append(var_type)
    info_dict["Equation"].append(equ)

    if isinstance(model_identifier, env.SDEnv):
      # Check if it is observable
      if (var in model_identifier.observables or
          case(var) in model_identifier.observables):
        info_dict["Observable"].append("Yes")
      else:
        info_dict["Observable"].append("No")
      # Check if it is actionable
      if (var in model_identifier.actionables or
          case(var) in model_identifier.actionables):
        info_dict["Actionable"].append("Yes")
      else:
        info_dict["Actionable"].append("No")
    else:
      info_dict["Observable"].append("NA")
      info_dict["Actionable"].append("NA")

  return parsed_model, pd.DataFrame(info_dict, index=all_model_vars)
