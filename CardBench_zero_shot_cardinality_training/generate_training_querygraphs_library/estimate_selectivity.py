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

"""Estimates the selectivity of single table predicates."""

from typing import Any

from CardBench_zero_shot_cardinality_training.generate_training_querygraphs_library import generate_training_querygraphs_helpers

predicate_operator_dict_operators_in_text = (
    generate_training_querygraphs_helpers.predicate_operator_dict_operators_in_text
)


def get_info_from_histogram(
    colnode, node
):
  """Gets the information from the histogram.

  Args:
    colnode: The column node.
    node: The node.

  Returns:
    A tuple of the information from the histogram.
  """
  minv = -1
  maxv = -1
  first_percentile = -1
  last_percentile = -1
  no_precentiles = -1
  constant = -1
  low = -1
  high = -1
  calculated_pos_from_from_histogram = False

  if colnode["column_type"] in [
      "INT64",
      "INT32",
      "NUMERIC",
      "BIGNUMERIC",
      "UINT64",
      "UINT32",
  ]:
    constant = int(node["constant"])
    low, high, minv, maxv, first_percentile, last_percentile, no_precentiles = (
        get_pos_int(colnode["percentiles_100"], constant)
    )
    calculated_pos_from_from_histogram = True
  elif colnode["column_type"] in ["FLOAT64", "DOUBLE", "DECIMAL", "BIGDECIMAL"]:
    constant = float(node["constant"])
    low, high, minv, maxv, first_percentile, last_percentile, no_precentiles = (
        get_pos_float(colnode["percentiles_100"], constant)
    )
    calculated_pos_from_from_histogram = True

  # the no percentiles is a check from when we used to have not fixed size
  # histograms
  if calculated_pos_from_from_histogram and no_precentiles >= 2:
    return (
        calculated_pos_from_from_histogram,
        low,
        high,
        minv,
        maxv,
        first_percentile,
        last_percentile,
        no_precentiles,
        constant,
    )
  else:
    return (
        False,
        low,
        high,
        minv,
        maxv,
        first_percentile,
        last_percentile,
        no_precentiles,
        constant,
    )


def estimate_selectivity_equality(
    colnode, node
):
  """Estimates the selectivity of an equality predicate.

  Args:
    colnode: The column node.
    node: The node.

  Returns:
    A tuple of the estimated selectivity and the offset.
  """
  if node["constant"] == "NULL":
    return colnode["null_frac"], [0.0, 1.0, -1.0, -1.0, -1.0, -1.0]
  (
      calculated_pos_from_from_histogram,
      low,
      high,
      minv,
      maxv,
      first_percentile,
      last_percentile,
      no_precentiles,
      constant,
  ) = get_info_from_histogram(colnode, node)

  if calculated_pos_from_from_histogram:
    # all the values in the table are the same
    if low == high:
      if low == constant:
        return 1.0, [0.0, 1.0, -1.0, -1.0, -1.0, -1.0]
      else:
        return 0.0, [0.0, 0.0, -1.0, -1.0, -1.0, -1.0]

    worth_of_percentile = 1 / (no_precentiles - 1)
    if maxv == minv and maxv == constant:
      fraction_of_own_percentiles = 1.0
    else:
      fraction_of_own_percentiles = 1 / colnode["num_unique"]

    size_of_constant_range_percentiles = last_percentile - first_percentile + 1

    estimated_sel = worth_of_percentile * (
        size_of_constant_range_percentiles + worth_of_percentile
    )

    remainder_from_fraction_of_own_percentile = (
        1 - fraction_of_own_percentiles
    ) / 2
    left_offset = (
        first_percentile + remainder_from_fraction_of_own_percentile
    ) * worth_of_percentile
    right_offset = (
        (no_precentiles - last_percentile)
        + remainder_from_fraction_of_own_percentile
    ) * worth_of_percentile

    return estimated_sel, [left_offset, right_offset, -1.0, -1.0, -1.0, -1.0]
  else:
    estimated_selectivity = 1 / colnode["num_unique"]
    remainder_from_estimated_selectivity = (1 - estimated_selectivity) / 2
    return estimated_selectivity, [
        remainder_from_estimated_selectivity,
        1 - remainder_from_estimated_selectivity,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    ]


# first is min, last is max
def get_pos_int(
    percentiles, value
):
  """Gets the position of the value in the percentiles.

  Args:
    percentiles: The percentiles.
    value: The value.

  Returns:
    A tuple of the position of the value in the percentiles.
  """
  no_precentiles = len(percentiles)
  last_percentile = 0

  for i in range(1, len(percentiles)):
    last_percentile = i
    if value <= int(percentiles[i]):
      break

  first_percentile = last_percentile - 1
  while first_percentile > 1 and int(percentiles[first_percentile]) == value:
    first_percentile -= 1

  minv = int(percentiles[first_percentile])
  maxv = int(percentiles[last_percentile])
  low = int(percentiles[0])
  high = int(percentiles[-1])

  return (
      low,
      high,
      minv,
      maxv,
      first_percentile,
      last_percentile,
      no_precentiles,
  )


def get_pos_float(
    percentiles, value
):
  """Gets the position of the value in the percentiles.

  Args:
    percentiles: The percentiles.
    value: The value.

  Returns:
    A tuple of the position of the value in the percentiles.
  """
  no_precentiles = len(percentiles)
  last_percentile = 0

  for i in range(1, len(percentiles)):
    last_percentile = i
    if value <= float(percentiles[i]):
      break

  first_percentile = last_percentile
  while last_percentile > 1 and float(percentiles[last_percentile]) == value:
    last_percentile -= 1

  minv = float(percentiles[last_percentile])
  maxv = float(percentiles[first_percentile])
  low = float(percentiles[0])
  high = float(percentiles[-1])

  return (
      low,
      high,
      minv,
      maxv,
      first_percentile,
      last_percentile,
      no_precentiles,
  )


def estimate_selectivity_gt_gte_no_percentiles(
    colnode, node
):
  """Estimates the selectivity of a greater than or equal to predicate.

  Args:
    colnode: The column node.
    node: The node.

  Returns:
    A tuple of the estimated selectivity and the offset.
  """
  minv = colnode["min_val"]
  maxv = colnode["max_val"]
  constant = node["constant"]
  if colnode["column_type"] in [
      "INT64",
      "INT32",
      "NUMERIC",
      "BIGNUMERIC",
      "UINT64",
      "UINT32",
  ]:
    minv = int(minv)
    maxv = int(maxv)
    constant = int(constant)
    if maxv - minv > 0:
      return (maxv - constant) / (maxv - minv)
    else:
      return 1 / 3
  elif colnode["column_type"] in ["FLOAT64", "DOUBLE", "DECIMAL", "BIGDECIMAL"]:
    minv = float(minv)
    maxv = float(maxv)
    constant = float(constant)
    if maxv - minv > 0:
      return (maxv - constant) / (maxv - minv)
    else:
      return 1 / 3
  else:
    return 1 / 3


def estimate_selectivity_lt_lte_no_percentiles(
    colnode, node
):
  """Estimates the selectivity of a less than or equal to predicate.

  Args:
    colnode: The column node.
    node: The node.

  Returns:
    A tuple of the estimated selectivity and the offset.
  """
  minv = colnode["min_val"]
  maxv = colnode["max_val"]
  constant = node["constant"]
  if colnode["column_type"] in [
      "INT64",
      "INT32",
      "NUMERIC",
      "BIGNUMERIC",
      "UINT64",
      "UINT32",
  ]:
    minv = int(minv)
    maxv = int(maxv)
    constant = int(constant)
    if maxv - minv > 0:
      return (constant - minv) / (maxv - minv)
    else:
      if constant == minv:
        return 1
      else:
        return 0

  elif colnode["column_type"] in ["FLOAT64", "DOUBLE", "DECIMAL", "BIGDECIMAL"]:
    minv = float(minv)
    maxv = float(maxv)
    constant = float(constant)
    if maxv - minv > 0:
      return (constant - minv) / (maxv - minv)
    else:
      if constant == minv:
        return 1
      else:
        return 0
  else:
    return 1 / 33


def estimate_selectivity_between(
    colnode, node
):
  """Estimates the selectivity of a between predicate.

  Args:
    colnode: The column node.
    node: The node.

  Returns:
    A tuple of the estimated selectivity and the offset.
  """
  minv = colnode["min_val"]
  maxv = colnode["max_val"]
  constant = node["constant"]
  if colnode["column_type"] in [
      "INT64",
      "INT32",
      "NUMERIC",
      "BIGNUMERIC",
      "UINT64",
      "UINT32",
  ]:
    minv = int(minv)
    maxv = int(maxv)
    constant1 = constant.split(", ")[0]
    constant2 = constant.split(", ")[1]
    if maxv - minv > 0:
      return abs(int(constant1) - int(constant2)) / (maxv - minv)
    else:
      if constant1 == minv:
        return 1
      else:
        return 0
  else:
    return 1 / 3


def estimate_selectivity_lt_lte(
    colnode, node
):
  """Estimates the selectivity of a less than or equal to predicate.

  Args:
    colnode: The column node.
    node: The node.

  Returns:
    A tuple of the estimated selectivity and the offset.
  """
  (
      calculated_pos_from_from_histogram,
      low,
      high,
      minv,
      maxv,
      first_percentile,
      last_percentile,
      no_precentiles,
      constant,
  ) = get_info_from_histogram(colnode, node)

  # the no percentiles is a check from when we used to have not fixed size
  # histograms
  if calculated_pos_from_from_histogram:
    # all the values in the table are the same
    if low == high:
      if low == constant:
        return 1.0, [0.0, 1.0, -1.0, -1.0, -1.0, -1.0]
      else:
        return 0.0, [0.0, 0.0, -1.0, -1.0, -1.0, -1.0]

    worth_of_percentile = 1 / (no_precentiles - 1)
    if maxv == minv:
      fraction_of_own_percentiles = 0.5
    else:
      fraction_of_own_percentiles = (constant - minv) / (maxv - minv)

    size_of_constant_range_percentiles = last_percentile - first_percentile + 1

    estimated_sel = (
        worth_of_percentile * first_percentile
        + worth_of_percentile
        * (fraction_of_own_percentiles * size_of_constant_range_percentiles)
    )

    left_offset = 0.0
    right_offset = (
        first_percentile + fraction_of_own_percentiles
    ) * worth_of_percentile

    return estimated_sel, [left_offset, right_offset, -1.0, -1.0, -1.0, -1.0]

  # fallback method
  estimated_sel = estimate_selectivity_lt_lte_no_percentiles(colnode, node)
  return estimated_sel, [0.0, estimated_sel, -1.0, -1.0, -1.0, -1.0]


def estimate_selectivity_gt_gte(
    colnode, node
):
  """Estimates the selectivity of a greater than or equal to predicate.

  Args:
    colnode: The column node.
    node: The node.

  Returns:
    A tuple of the estimated selectivity and the offset.
  """
  (
      calculated_pos_from_from_histogram,
      low,
      high,
      minv,
      maxv,
      first_percentile,
      last_percentile,
      no_precentiles,
      constant,
  ) = get_info_from_histogram(colnode, node)

  if calculated_pos_from_from_histogram:
    # all the values in the table are the same
    if low == high:
      if low == constant:
        return 1.0, [0.0, 1.0, -1.0, -1.0, -1.0, -1.0]
      else:
        return 0.0, [0.0, 0.0, -1.0, -1.0, -1.0, -1.0]

    worth_of_percentile = 1 / (no_precentiles - 1)
    if maxv == minv:
      fraction_of_own_percentiles = 0.5
    else:
      fraction_of_own_percentiles = (maxv - constant) / (maxv - minv)

    size_of_constant_range_percentiles = last_percentile - first_percentile + 1

    estimated_sel = (
        worth_of_percentile * (no_precentiles - last_percentile)
        + worth_of_percentile
        * (1 - fraction_of_own_percentiles)
        * size_of_constant_range_percentiles
    )

    left_offset = (
        first_percentile + fraction_of_own_percentiles
    ) * worth_of_percentile
    right_offset = 1.0

    return estimated_sel, [left_offset, right_offset, -1.0, -1.0, -1.0, -1.0]

  # fallback method
  estimated_sel = estimate_selectivity_gt_gte_no_percentiles(colnode, node)
  return estimated_sel, [(1 - estimated_sel), 1.0, -1.0, -1.0, -1.0, -1.0]


def estimate_selectivity(queryplan):
  """Estimates the selectivity of the predicates.

  Args:
    queryplan: The query plan.
  """
  for node in queryplan["nodes"]:
    if node["nodetype"] == "predicate_operator":
      node["offset"] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
      # find input col node
      prednodeid = node["id"]
      colnode_id = -1
      count_matches = 0
      for e in queryplan["edges"]:
        if e["to"] == prednodeid:
          colnode_id = e["from"]
          count_matches += 1
      if count_matches > 1:  # join predicate
        node["estimated_selectivity"] = -1
        continue
      colnode = {}
      for n in queryplan["nodes"]:
        if n["id"] == colnode_id:
          colnode = n

      estimated_selectivity = -1
      offset = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
      if node["operator"] == predicate_operator_dict_operators_in_text["equal"]:
        estimated_selectivity, offset = estimate_selectivity_equality(
            colnode, node
        )

      elif (
          node["operator"]
          == predicate_operator_dict_operators_in_text["not_equal"]
      ):
        estimated_selectivity, offset = estimate_selectivity_equality(
            colnode, node
        )
        estimated_selectivity = 1 - estimated_selectivity

      elif (
          node["operator"]
          == predicate_operator_dict_operators_in_text["greater_or_equal"]
          or node["operator"]
          == predicate_operator_dict_operators_in_text["greater"]
      ):
        if not colnode["percentiles_100"]:
          estimated_selectivity, offset = (
              estimate_selectivity_gt_gte_no_percentiles(colnode, node)
          )
        else:
          estimated_selectivity, offset = estimate_selectivity_gt_gte(
              colnode, node
          )

      elif (
          node["operator"]
          == predicate_operator_dict_operators_in_text["less_or_equal"]
          or node["operator"]
          == predicate_operator_dict_operators_in_text["less"]
      ):
        if not colnode["percentiles_100"]:
          estimated_selectivity, offset = (
              estimate_selectivity_lt_lte_no_percentiles(colnode, node)
          )
        else:
          estimated_selectivity, offset = estimate_selectivity_lt_lte(
              colnode, node
          )

      elif (
          node["operator"]
          == predicate_operator_dict_operators_in_text["between"]
      ):
        estimated_selectivity, offset = estimate_selectivity_between(
            colnode, node
        )

      node["estimated_selectivity"] = estimated_selectivity
      node["offset"] = offset

      if node["estimated_selectivity"] > 1:
        node["estimated_selectivity"] = 1.0
      if node["estimated_selectivity"] < 0:
        node["estimated_selectivity"] = 0.0
