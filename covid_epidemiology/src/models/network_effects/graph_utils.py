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

"""Utility functions to support graph-theoretic features.

Utlity functions for graph-theoretic features over locations that capture
neighborhood effects. Include both static and temporal features.

Specifically, this includes the ability to programmatically create and insert
new features (as feature key-value combinations and new FeatureSpecs into the
namespace of already loaded Python modules.
"""

import enum
import logging
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from covid_epidemiology.src import constants
from covid_epidemiology.src.models import generic_seir_specs_county  # pylint: disable=unused-import
from covid_epidemiology.src.models import generic_seir_specs_japan_prefecture  # pylint: disable=unused-import
from covid_epidemiology.src.models import generic_seir_specs_state  # pylint: disable=unused-import
from covid_epidemiology.src.models.shared import model_spec

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(levelname)s:%(message)s")


class FeatureType(enum.Enum):
  """Type of feature."""

  TIMESERIES = enum.auto()
  STATIC = enum.auto()


def is_imported(module):
  """Checks if module already imported into current namespace or any module."""
  return module in sys.modules and module in dir()


def get_variables_in_module(module):
  """Gets all variable names from a given module."""
  return [
      item for item in dir(sys.modules[module]) if not item.startswith("__")
  ]


def get_variable_maps(
    module,
    include = None,
    exclude = None):
  """Gets a maps of variable name->value and value->name from a given module."""

  var_list = get_variables_in_module(module)
  var_val_map = {}
  val_var_map = {}
  for v in var_list:
    if include and not v.startswith(include):
      continue
    if exclude and v.startswith(exclude):
      continue
    var_value = getattr(sys.modules[module], v)
    var_val_map[v] = var_value
    if isinstance(var_value, list):  # lists are not hashable so map to tuple
      var_value = tuple(var_value)
    if not isinstance(var_value, dict):  # dicts are not hashable so ignore
      try:
        val_var_map[var_value] = v
      except TypeError:
        logging.debug(
            "Value type %s is not hashable, unable to save. Continuing.",
            type(var_value))
  return var_val_map, val_var_map


def create_feature_names(module,
                         base_list,
                         ops_dict,
                         location_granularity,
                         prefix = ""):
  """Creates new graph feature names.

  Creates new graph feature constants in the appropriate modules' namespaces.
  This function will create new features with the specified base name+ops combos
  and insert them into the specified module. Downstream to the calling function,
  other functions retrieve these features for further processing.

  Args:
    module: string. Name of module into which feature name/value pairs are to be
      created.
    base_list: list of strings. List of base features from which to generate new
      features.
    ops_dict: dict of name->aggregation functions to apply while generating
      graph features.
    location_granularity: string. Location granularity, one of "STATE",
      "COUNTY", "JAPAN_PREFECTURE"
    prefix: string. String to prepend to new feature names.

  Returns:
    None.

  Raises:
    KeyError: if the base feature is not found as a value in the module.
  """

  # get correct module references based on location granularity
  if location_granularity == constants.LOCATION_GRANULARITY_STATE:
    spec_module_name = "generic_seir_specs_state"
    inclusions = None
    exclusions = (constants.LOCATION_GRANULARITY_COUNTY,
                  constants.LOCATION_GRANULARITY_JAPAN_PREFECTURE)
  elif location_granularity == constants.LOCATION_GRANULARITY_COUNTY:
    spec_module_name = "generic_seir_specs_county"
    inclusions = None
    exclusions = tuple(constants.LOCATION_GRANULARITY_JAPAN_PREFECTURE,)
  elif location_granularity == constants.LOCATION_GRANULARITY_JAPAN_PREFECTURE:
    spec_module_name = "generic_seir_specs_japan_prefecture"
    inclusions = tuple(constants.LOCATION_GRANULARITY_JAPAN_PREFECTURE,)
    exclusions = (constants.LOCATION_GRANULARITY_COUNTY,
                  constants.POPULATION_DENSITY)
  else:
    raise ValueError(("Graph features not enabled for",
                      "location granularity {}".format(location_granularity)))

  # get all the constants from the 'constants' module
  _, value_constant_map = get_variable_maps(
      module, include=inclusions, exclude=exclusions)

  # Generate graph feature name-value pairs from the list+dict provided above.
  # syntax consistency with the module is maintained.
  # Full Cartesian product of features will be generated.
  for base in base_list:
    if base not in value_constant_map:
      raise KeyError((f"Expected '{base}' key not found in value->constant map "
                      f"for module '{module}'"))
    else:
      feature_key = value_constant_map[base]
    for name in ops_dict:
      # define new feature constants, with special handling for
      # 'population_density', since this is a key/value combination where
      # multiple keys map to the same value
      if feature_key.lower() == "population_density":
        base_local = (
            prefix.upper() + location_granularity.upper() + "_" +
            feature_key.upper())
      else:
        base_local = prefix.upper() + feature_key.upper()
      new_feature_name = (
          base_local + "_" + name.upper()
      )  # of form 'PREFIX_XXX_YY' (all uppercase) in 'constants'
      new_feature_value = prefix.lower() + base.lower() + "_" + name.lower()
      # set the new feature constant.
      # e.g.: constants.GRAPH_MOBILITY_MEAN = "graph_m50_mean"
      setattr(sys.modules[module], new_feature_name, new_feature_value)
      logging.info("Inserting %s with value %s into %s", new_feature_name,
                   new_feature_value, module)


def create_feature_specs(module,
                         base_list,
                         ops_dict,
                         location_granularity,
                         apply_lasso = False,
                         prefix = "",
                         suffix = "spec"):
  """Creates new graph FeatureSpecs.

  Create new graph FeatureSpecs in the appropriate modules' namespaces.

  Args:
    module: Name of module from which base feature name/value pairs are to be
      read.
    base_list: List of base features from which to generate new features.
    ops_dict: dict of name->aggregation functions to apply while generating
      graph features.
    location_granularity: Location granularity, one of "STATE", "COUNTY"
    apply_lasso: Flag to turn on lasso (L1) regularization.
    prefix: String to prepend to new feature specs.
    suffix: String to append to new feature specs.

  Raises:
    KeyError: If the base feature is not found as a value in the module.
  """

  # get correct module references based on location granularity
  if location_granularity == constants.LOCATION_GRANULARITY_STATE:
    spec_module_name = "models.generic_seir_specs_state"
    inclusions = None
    exclusions = (constants.LOCATION_GRANULARITY_COUNTY,
                  constants.LOCATION_GRANULARITY_JAPAN_PREFECTURE)
  elif location_granularity == constants.LOCATION_GRANULARITY_COUNTY:
    spec_module_name = "models.generic_seir_specs_county"
    inclusions = None
    exclusions = tuple(constants.LOCATION_GRANULARITY_JAPAN_PREFECTURE,)
  elif location_granularity == constants.LOCATION_GRANULARITY_JAPAN_PREFECTURE:
    spec_module_name = "models.generic_seir_specs_japan_prefecture"
    inclusions = tuple(constants.LOCATION_GRANULARITY_JAPAN_PREFECTURE,)
    exclusions = tuple(constants.LOCATION_GRANULARITY_COUNTY,)
  else:
    raise ValueError(("Graph features not enabled for",
                      "location granularity {}".format(location_granularity)))

  # get all the constants from the 'constants' module
  _, value_constant_map = get_variable_maps(
      module, include=inclusions, exclude=exclusions)

  # Get graph feature name-value pairs from the list+dict provided above.
  # syntax consistency with the module is maintained.
  # Full Cartesian product of FeatureSpecs will be generated.
  for base in base_list:
    if base not in value_constant_map:
      raise KeyError((f"Expected '{base}' key not found in value->constant map "
                      f"for module '{module}'"))
    else:
      feature_key = value_constant_map[base]
    for name in ops_dict:
      # define new feature constants, with special handling for
      # 'population_density', since this is a key/value combination where
      # multiple keys map to the same value
      if feature_key.lower() == "population_density":
        base_local = (f"{prefix.upper()}{location_granularity.upper()}_"
                      f"{feature_key.upper()}")
      else:
        base_local = prefix.upper() + feature_key.upper()
      # of form 'PREFIX_XXX_YY' (all uppercase) in module (e.g. 'constants')
      new_feature_name = f"{base_local}_{name.upper()}"
      new_spec_name = new_feature_name.lower() + "_" + suffix.lower()
      # set the new FeatureSpec dataclasses.
      # e.g. graph_mobility_mean = model_spec.FeatureSpec(
      #        name=constants.GRAPH_MOBILITY_MEAN, initializer=None,
      #        apply_lasso=True)
      new_feature_key = getattr(sys.modules[module], new_feature_name)
      setattr(
          sys.modules[spec_module_name], new_spec_name,
          model_spec.FeatureSpec(
              name=new_feature_key, initializer=None, apply_lasso=apply_lasso))
      logging.info(
          "Inserting FeatureSpec %s for feature key %s and value %s into %s",
          new_spec_name, new_feature_name, new_feature_key, spec_module_name)


def get_feature_spec_by_feature_name(
    module,
    feature,
    include=None,
    exclude=None):
  """Return FeatureSpec corresponding to feature name from specified module."""

  constant_value_map, _ = get_variable_maps(
      module, include=include, exclude=exclude)
  for var, value in constant_value_map.items():
    if isinstance(value, model_spec.FeatureSpec) and value.name == feature:
      return var, value
  return None


def get_feature_name_to_feature_spec_map(
    module,
    include=None,
    exclude=None):
  """Return the feature name to FeatureSpec map from the specified module."""

  constant_value_map, _ = get_variable_maps(
      module, include=include, exclude=exclude)
  name_spec_map = {}
  for value in constant_value_map.values():
    if isinstance(value, model_spec.FeatureSpec):
      name_spec_map[value.name] = value
  return name_spec_map


def insert_featurespecs_in_encoderspec(
    modelspec, rate, feature_type,
    feature_specs):
  """Insert a list of FeatureSpec into a EncoderSpec in a given ModelSpec.

  Insert a list of FeatureSpecs, all of the same feature type, into a
  EncoderSpec specified by its name. The EncoderSpec comes from a specified
  ModelSpec.

  Args:
    modelspec: model_spec.ModelSpec. Model specification that contains
      EncoderSpecs.
    rate: string. Name of EncoderSpec into which to insert FeatureSpecs
    feature_type: FeatureType. Type of feature in the FeatureSpecs, one of
      FeatureType.TIMESERIES or FeatureType.STATIC.
    feature_specs: List of model_spec.FeatureSpec. List of FeatureSpecs to
      insert.

  Raises:
    ValueError: if ModelSpec / List of FeatureSpecs is not provided.
    ValueError: if feature_type is unknown
  """

  if not modelspec:
    raise ValueError(f"Must provide a ModelSpec, received {modelspec}.")
  if not feature_specs:
    raise ValueError(("Must provide a list of FeatureSpec, received "
                      f"{feature_specs}."))
  if not rate:
    raise ValueError("Must provide an encoder name, received {rate}.")
  # TODO(sinharaj): the loop below can be more efficient, change this.
  num_enc = len(modelspec.encoder_specs)
  for enc in modelspec.encoder_specs:
    if enc.encoder_name == rate:  # found correct encoder
      if feature_type == FeatureType.TIMESERIES:
        if enc.covariate_feature_specs:
          enc.covariate_feature_specs.extend(feature_specs)
        else:
          enc.covariate_feature_specs = list(feature_specs)
      elif feature_type == FeatureType.STATIC:
        if enc.static_feature_specs:
          enc.static_feature_specs.extend(feature_specs)
        else:
          enc.static_feature_specs = list(feature_specs)
      break
  else:
    raise ValueError((f"An unknown FeatureSpec  from {feature_specs} of type "
                      f"{feature_type} in encoder {rate}."))


def verify_graph_features(features_list, base_features,
                          ops_dict, prefix):
  """Verify that all graph features are present in a given list.

  Check that the full Cartesian product of base features and ops is present in
  the provided features list. Return any absent features.

  Args:
    features_list: List of strings. List of features to check for completeness.
    base_features: List of strings. List of base features.
    ops_dict: Dict of graph ops. Dictionary of op names to op functions.
    prefix: string. String prefix to add before each feature.

  Returns:
    List of missing features.
  """

  # remove all non-graph features from the features_list
  graph_features = [f for f in features_list if f.startswith(prefix)]
  # construct cartesian product
  cartesian_product = []
  for base in base_features:
    for op_name in list(ops_dict.keys()):
      for t in prefix:
        cartesian_product.append(f"{t}_{base}_{op_name}")
  # calculate common elements, and asymmetric differences between lists
  common = list(set(graph_features).intersection(set(cartesian_product)))
  if set(common) == set(cartesian_product):  # all expected features are present
    logging.info(f"All features with prefix '{prefix}' are present")  # pylint: disable=logging-format-interpolation
    return None
  feature_list_leftover = set(graph_features) - set(cartesian_product)
  cartesian_product_leftover = set(cartesian_product) - set(graph_features)
  if feature_list_leftover:
    num = len(feature_list_leftover)
    logging.info(f"Found {num} extra features {feature_list_leftover}")  # pylint: disable=logging-format-interpolation
  if cartesian_product_leftover:
    num = len(cartesian_product_leftover)
    logging.info(  # pylint: disable=logging-format-interpolation
        f"Expected but did not find {num} features {cartesian_product_leftover}"
    )
  return list(feature_list_leftover.union(cartesian_product_leftover))
