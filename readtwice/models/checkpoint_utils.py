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

"""Utils for handling Tensorflow checkpoints."""
import collections
import os
import re
from typing import Dict, List, Optional, Sequence, Text, Tuple

import tensorflow.compat.v1 as tf


def get_assignment_map_from_checkpoint(
    variables,
    ckpt_path,
    variable_scope = "",
    ckpt_variable_scope = "",
    require_all_variables_initialized = False
):
  """Gets the mapping from checkpoint variable names to `variable` names.

  Computes the *intersection* of `variables` (under `variable_scope`) and
  checkpoint variables (under `ckpt_variable_scope`) and gets the name
  mapping from the latter to the former.

  Args:
    variables: The list of Tensorflow variables one aims to initialize.
    ckpt_path: Path to the checkpoint to load `variables` from.
    variable_scope: The scope of `variables` to initialize. `Variables` outside
      this scope will be ignored. If "", use all `variables`; otherwise it
      should end with '/'.
    ckpt_variable_scope: The scope of checkpoint variables to initialize from.
      Checkpoint variables outside this scope will be ignored. If "", use all
      `variables`; otherwise it should end with '/'.
    require_all_variables_initialized: If True, a ValueError will be raised if
      not all `variables` in the `variable_scope` can be mapped to the
      corresponding checkpoint variables in the `ckpt_variable_scope`.

  Returns:
    assignment_map: Mapping from checkpoint variable names to `variable`.
      Keys and values are matching variables under the `ckpt_variable_scope`
      and `variable_scope` (sub-)trees.
    initialized_variable_names: Names of `variables` that get matched to
      checkpoint variables.

  Raises:
    ValueError if
      (a) input scope name is not empty and doesn't end with "/"; or
      (b) names of `variables` doesn't end with ':0' (unlikely to happen); or
      (c) not all variables in variable_scope are initialized
          (if `require_all_variables_initialized` is True).

  Example
    Input:
      variables: ["a/aa/aaa:0", "a/c/cc/ccc:0", "d/dd:0"]
      ckpt_variables: ["b/aa/aaa", "b/f"]
      variable_scope: "a/"
      ckpt_variable_scope: "b/"
    Output:
      assignment_map: {"b/aa/aaa": <tf.Variable "a/aa/aaa:0">}
      initialized_variable_names: ["a/aa/aaa:0"]
  """
  if variable_scope and not variable_scope.endswith("/"):
    raise ValueError("{} should end with '/'.".format(variable_scope))

  if ckpt_variable_scope and not ckpt_variable_scope.endswith("/"):
    raise ValueError("{} should end with '/'.".format(ckpt_variable_scope))

  variable_names_stripped = set()
  for var in variables:
    var_name = var.name

    # Ignores `variables` outside scope.
    # Note that all strings start with "".
    if not var_name.startswith(variable_scope):
      continue

    # Names of variables from Tensorflow API all have the suffix of ":0"
    # while those from checkpoint don't. Here we strip the suffix out.
    m = re.match("^(.*):\\d+$", var_name)
    if m is not None:
      var_name = m.group(1)
    else:
      raise ValueError(
          "Variable name doesn't end with ':0': {}".format(var_name))

    # Strips the `variable_scope` prefix out.
    var_name_stripped = var_name[len(variable_scope):]
    if var_name_stripped:
      variable_names_stripped.add(var_name_stripped)

  var_name_to_variable = {var.name: var for var in variables}
  assignment_map = collections.OrderedDict()
  initialized_variable_names = []

  for ckpt_var_name, _ in tf.train.list_variables(ckpt_path):
    # Ignores checkpoint variables outside scope.
    # Note that all strings start with "".
    if not ckpt_var_name.startswith(ckpt_variable_scope):
      continue

    ckpt_var_name_stripped = ckpt_var_name[len(ckpt_variable_scope):]
    if ckpt_var_name_stripped not in variable_names_stripped:
      continue
    variable_names_stripped.remove(ckpt_var_name_stripped)
    var_name = variable_scope + ckpt_var_name_stripped + ":0"

    assignment_map[ckpt_var_name] = var_name_to_variable[var_name]
    initialized_variable_names.append(var_name)
  if variable_names_stripped and require_all_variables_initialized:
    raise ValueError(
        f"The following variables in variable_scope cannot be mapped to any "
        f"checkpoint variable in ckpt_variable_scope: "
        f"{variable_names_stripped}.")
  return (assignment_map, initialized_variable_names)


def _log_customized_initialization(
    init_checkpoint,
    variables,
    global_variables,
    initialized_variable_names = ()):
  """Logs customized initialization."""
  if init_checkpoint:
    tf.logging.info("Initialize from the ckpt %s", init_checkpoint)
  else:
    tf.logging.info("Random initialized.")

  if global_variables:
    tf.logging.info("**** Global Variables ****")
  else:
    tf.logging.info("**** Trainable Variables ****")

  for var in variables:
    init_string = ""
    if var.name in initialized_variable_names:
      init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)


def get_scaffold_fn(
    init_checkpoint,
    global_vars = False,
    variable_scope_pairs = (("", ""))
):  # pytype: disable=annotation-type-mismatch
  """Gets `scaffold_fn` for initializing model variables from checkpoint.

  If the checkpoint ends with "latest", then load the latest checkpoint in the
  directory of the `init_checkpoint`.

  Args:
    init_checkpoint: Text, the initial checkpoint.
    global_vars: bool, whether or not initialize global variables.
    variable_scope_pairs: Sequence of (variable_scope, ckpt_variable_name)
      pairs, where `variable_scope` is the scope of variables to initialize, and
      `ckpt_variable_name` is the scope of checkpoint variables to initialize
      from. The initializations from later pairs will overwrite those from
      earlier pairs.

  Returns:
    The `scaffold_fn` for initializing model variables from checkpoint. If
    `init_checkpoint` is None, return None.

  """
  tvars = tf.global_variables() if global_vars else tf.trainable_variables()

  if init_checkpoint is None:
    _log_customized_initialization(init_checkpoint, tvars, global_vars)
    return None
  if init_checkpoint.endswith("latest"):
    ckpt_dir = os.path.dirname(init_checkpoint)
    init_checkpoint = tf.train.latest_checkpoint(ckpt_dir)

  def scaffold_fn():
    """The TPU scaffold function."""
    for variable_scope, ckpt_variable_scope in variable_scope_pairs:
      if variable_scope and not variable_scope.endswith("/"):
        variable_scope += "/"
      if ckpt_variable_scope and not ckpt_variable_scope.endswith("/"):
        ckpt_variable_scope += "/"
      assignment_map, initialized_variable_names = (
          get_assignment_map_from_checkpoint(tvars, init_checkpoint,
                                             variable_scope,
                                             ckpt_variable_scope))
      _log_customized_initialization(init_checkpoint, tvars, global_vars,
                                     initialized_variable_names)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    return tf.train.Scaffold()

  return scaffold_fn
