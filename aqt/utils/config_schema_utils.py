# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Utilities to generate ConfigDict instances.

This is the shared part of config schema without model-specific components.
Most users will just create a ConfigDict instance with 'get_config' and then
override its parameters to specialize the configuration.
"""

from typing import List, Optional, Union

import ml_collections
from ml_collections import config_dict


def make_reference(
    config,
    field,
):
  """Returns a reference to a field for wiring up a config dict.

  The returned reference is "one-way": a change to the original field value
  propagates to the reference, but a change to the reference does not propagate
  back up.

  This works recursively. If config['field'] is itself a ConfigDict (or a tree
  of ConfigDicts) then an identical tree is constructed with all the leaves
  referring to original leaves. Internal nodes are *not* references.

  See config_schema_test.MakeReferenceRecursiveTest for example usage.

  Args:
    config: A ConfigDict instance.
    field: The name of a field contained in 'config'.

  Returns:
    If config['field'] is a scalar, returns a one-way FieldReference to
      config['field']. If config['field'] is a ConfigDict, returns a new
      ConfigDict whose fields are references to the fields in config['field'].
  """
  field_value = config[field]
  if isinstance(field_value, ml_collections.ConfigDict):
    new_dict = ml_collections.ConfigDict()
    for subfield in field_value:
      new_dict[subfield] = make_reference(field_value, subfield)
    return new_dict
  else:
    return ml_collections.FieldReference(config.get_ref(field))


def set_default_reference(child_config,
                          parent_config,
                          field,
                          *,
                          parent_field = None):
  """Sets fields in a child config to be references to fields in a parent.

  When all parameters are provided and field is a single string, this
  is equivalent to
  child_config[field] =
    make_reference(parent_config[parent_field])

  See documentation on make_reference for the full implications.

  If parent_field is not specified, it is taken to be the same as 'field'.
  This is useful for the common case where the name of a field is the same
  in the parent and child.

  If 'field' is a list of strings, that is equivalent to calling this
  function separately with each specific string. Useful for creating many
  references at once.

  See config_schema_test.SetDefaultReferenceTest for example usage.

  Args:
    child_config: The child ConfigDict.
    parent_config: The parent ConfigDict.
    field: Either the name of a field or a list of field names.
    parent_field: The name of the parent field in 'parent_config'.
      If None (the default), the parent field name is assumed to be the same
      as the child field name.
  """
  if isinstance(field, list):
    for subfield in field:
      set_default_reference(
          child_config=child_config,
          field=subfield,
          parent_config=parent_config,
          parent_field=parent_field)
  else:
    if parent_field is None:
      parent_field = field
    child_config[field] = make_reference(parent_config, parent_field)


# Functions returning placeholders are marked with _ph suffix are a device
# to increase code reability in this file. Their intent is to reduce large
# amount of repetition and getting the type closer to the colon.
float_ph = lambda: config_dict.placeholder(float)
int_ph = lambda: config_dict.placeholder(int)
str_ph = lambda: config_dict.placeholder(str)
bool_ph = lambda: config_dict.placeholder(bool)


def get_dense_config(
    parent_config):
  """Creates a ConfigDict corresponding to aqt.flax_layers.DenseAqt.HParams."""
  config = ml_collections.ConfigDict()
  set_default_reference(config, parent_config, [
      "weight_prec", "weight_quant_granularity", "quant_type", "quant_act",
      "weight_half_shift"
  ])
  config.lock()
  return config


def get_conv_config(
    parent_config):
  """Creates a ConfigDict corresponding to aqt.flax_layers.ConvAqt.HParams."""
  config = ml_collections.ConfigDict()
  set_default_reference(config, parent_config, [
      "weight_prec", "weight_quant_granularity", "quant_type", "quant_act",
      "weight_half_shift"
  ])
  config.lock()
  return config


def get_fp_quant_config():
  config = ml_collections.ConfigDict({
      "fp_spec": get_fp_config(),
      "is_scaled": bool_ph(),
  })
  config.lock()
  return config


def get_fp_config():
  config = ml_collections.ConfigDict({
      "exp_min": int_ph(),
      "exp_max": int_ph(),
      "sig_bits": int_ph()
  })
  config.lock()
  return config


# TODO(shivaniagrawal): base config should be more generic and only model
# specific configs should be updated.
def get_base_config(use_auto_acts,
                    fp_quant):
  """Return a base ConfigDict for AQT; does not have model specific fields."""
  if use_auto_acts:
    bounds = ml_collections.ConfigDict({
        "initial_bound": float_ph(),
        "stddev_coeff": float_ph(),
        "absdev_coeff": float_ph(),
        "mix_coeff": float_ph(),
        "reset_stats": bool_ph(),
        "ema_coeff": float_ph(),
        "use_cams": bool_ph(),
        "exclude_zeros": bool_ph(),
        "use_mean_of_max": bool_ph(),
        "granularity": str_ph()
    })
  else:
    bounds = float_ph()
  if fp_quant:
    prec = get_fp_quant_config()
  else:
    prec = int_ph()
  base_config = ml_collections.ConfigDict({
      "metadata": {
          "description": "Base configuration",
          "hyper_str": str_ph(),
      },
      "weight_decay": float_ph(),
      "activation_bound_update_freq": int_ph(),
      "activation_bound_start_step": int_ph(),
      "prec": prec,
      "half_shift": bool_ph(),
      "quant_type": str_ph(),
      "quant_act": {
          "bounds": bounds,
          # TODO(shivaniagrawal): The input distribution is really an intrinsic
          # model property and shouldn't be part of the model configuration.
          # Update the hparam dataclasses to eliminate the input_distribution
          # field and then delete this.
          "input_distribution": "symmetric"
      },
      "weight_quant_granularity": str_ph()
  })

  set_default_reference(
      base_config, base_config, "weight_prec", parent_field="prec")
  set_default_reference(base_config.quant_act, base_config, "prec")
  set_default_reference(
      base_config, base_config, "weight_half_shift", parent_field="half_shift")
  set_default_reference(base_config.quant_act, base_config, "half_shift")

  return base_config
