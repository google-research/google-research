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

"""Utils for the JAX model.Model class."""

from typing import Dict, Any


def match_params(params_dict,
                 struct_to_match):
  """Matches the params in params_dict to the tree structure of struct_to_match.

  This function maps between the JAX model.Model param representations and a
  more conventional Flax model. The main challenge in substituting one set of
  parameters for another is that the Flax model often includes nested modules,
  which results in the parameters following a tree structure. In contrast, our
  model class executes the computation graph in a single module, so that the
  parameters are just a simple list.

  Thus we adopt a convention in our parameter names: to a first approximation,
  the parameter with key "a/b/c/d" is equivalent to the nested dictionary
    {a: {b: {c: {d: param}}}}

  To ensure that this operation does not map two different parameters to the
  same flat key, we will assume that the keys in any (sub)dictionary are
  prefix-free, e.g., if A is a key, then AB is not a key, where A and B are any
  two strings and AB is their concatenation.

  Then it holds that we can map any nested dictionary to a flattened
  representation, e.g.,
    {a: {b: {c: {d: param}}}} => {a/b/c/d: param}

  Capitalization does not matter.

  Args:
    params_dict: the dictionary of params
    struct_to_match: the new dictionary structure to match

  Returns:
    a dictionary of params organized according to struct_to_match
  """

  def flatten_params(params):
    flat_dict = {}
    for k, v in params.items():
      if isinstance(v, dict):
        for subk, subv in flatten_params(v).items():
          flat_dict[f"{k}/{subk}".lower()] = subv
      else:
        flat_dict[k.lower()] = v
    return flat_dict

  flat_params = flatten_params(params_dict)

  new_params_dict = {}

  def build_tree(struct, prefix, new_params, flat_params):
    for k, v in struct.items():
      if isinstance(v, dict):
        new_params_v = new_params.pop(k, {})
        build_tree(v, f"{prefix}/{k}", new_params_v, flat_params)
        new_params[k] = new_params_v
      else:
        flat_k = f"{prefix}/{k}".lower()[1:]  # prefix starts with "/"
        flat_v = flat_params.pop(flat_k)
        if flat_v.shape != v.shape:
          raise ValueError(f"{flat_k}: new param shape {flat_v.shape} does not "
                           f"match old param shape {v.shape}")
        new_params[k] = flat_v

  build_tree(struct_to_match, "", new_params_dict, flat_params)

  if flat_params:
    raise ValueError(f"params_dict not consumed: {flat_params}")

  return new_params_dict


def split_div_mul(v):
  """Returns the base, div, and mul factor from a symbolic shape constant."""
  if "*" in v:
    arr = v.split("*")
    if len(arr) != 2:
      raise ValueError(f"Too many mults in features {v}.")
    v, mul = arr[0], int(arr[1])
  else:
    mul = 1
  if "%" in v:
    arr = v.split("%")
    if len(arr) != 2:
      raise ValueError(f"Too many divs in features {v}.")
    v, div = arr[0], int(arr[1])
  else:
    div = 1
  return v, div, mul
