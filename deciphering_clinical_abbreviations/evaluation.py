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

"""Helper functions for evaluating of a medical abbreviation expansion model."""

import collections
from collections.abc import Collection, Mapping, Sequence
import numpy as np
import pandas as pd
from deciphering_clinical_abbreviations import expansion_attribution as expansion_attribution_lib

# Type aliases.
AbbrevExpansionDict = Mapping[str, Sequence[str]]
FilePath = str


def load_abbreviation_expansion_dictionary(
    abbrev_dict_path,
    abbreviation_as_key):
  """Loads a table of abbreviation-expansion pairs and converts to a mapping.

  Args:
    abbrev_dict_path: The path to the abbreviation expansion dictionary csv. The
      csv should have two self-explanatory columns: 'abbreviation' and
      'expansion'.
    abbreviation_as_key: Whether to make the abbreviation the key and the list
      of expansions the value. If False, the expansion is the key and the
      abbreviations is the value.
  Returns:
    A dictionary mapping each abbreviation to a list of all valid expansions.
  """
  abbreviation_expansions_dict_df = pd.read_csv(
      abbrev_dict_path, na_filter=False)
  abbreviation_expansions_dict = collections.defaultdict(list)
  for row_tuple in abbreviation_expansions_dict_df.itertuples(index=False):
    key, value = row_tuple.abbreviation, row_tuple.expansion
    if not abbreviation_as_key:
      key, value = value, key
    abbreviation_expansions_dict[key].append(value)
  return dict(abbreviation_expansions_dict)


def load_expansion_equivalencies_set(
    expansion_equivalencies_path):
  """Loads a table of abbreviation-equivalency pairs and converts to a set.

  Args:
    expansion_equivalencies_path: The path to the expansion equivalencies csv.
      The csv should have the following 2 columns:
        - abbreviation: self explanatory
        - equivalent_expansions: a pipe-separated list of expansion phrases
          which are considered clinically equivalent for the given abbreviation.

  Returns:
    A set of tuples, each of which contains 3 elements. The first element is the
      abbreviation and the second and third elements are two clinically
      equivalent expansion phrases, in lexicographic order. This set can be used
      to check if a pair of expansion phrases are considered equivalent for a
      given abbreviation by ordering the expansions, constructing the tuple, and
      checking for set membership.
  """
  abbrev_to_equivalencies = pd.read_csv(
      expansion_equivalencies_path, na_filter=False)
  expansion_equivalencies_set = set()
  for row_tuple in abbrev_to_equivalencies.itertuples(index=False):
    abbreviation = row_tuple.abbreviation
    equivalent_expansions = sorted(row_tuple.equivalent_expansions.split("|"))
    while equivalent_expansions:
      expansion = equivalent_expansions.pop()
      for other_expansion in equivalent_expansions:
        expansion_equivalencies_set.add(
            (abbreviation, other_expansion, expansion))
  return expansion_equivalencies_set


def _generate_abbreviation_expansions_from_label(
    label):
  """Generates a ground truth abbreviation-expansion mapping from the label.

  Args:
    label: The string containing the labels. This string follows the following
      label format: each abbreviation-expansion pair is separated by a comma,
      and within a pair, the abbreviation and expansion are separated by a
      space. This implies that abbreviations that contain spaces are not
      currently supported; nor are expansion phrases containing commas. For
      abbreviations which exist in multiple places and have different
      expansions, every expansion must be listed alongside the duplicated
      abbreviation in the order they appear. If an abbreviation exists in
      multiple places but has the same expansion, a single entry will suffice.
  Returns:
    A mapping from abbreviation to list of expansions in the order they occur.
  """
  abbrev_expansion_elements = [x.strip() for x in label.split(",")]
  gt_dict = collections.defaultdict(list)
  for elem in abbrev_expansion_elements:
    elem = elem.strip()
    abbrev = elem.split()[0]
    expansion = " ".join(elem.split()[1:])
    gt_dict[abbrev].append(expansion)
  return dict(gt_dict)


def _generate_label_model_comparisons(
    label_abbreviation_expansions,
    model_expansion_mapping,
    ):
  """Generates expansion-level comparisons between labels and model outputs.

  Args:
    label_abbreviation_expansions: The labeled abbreviation-expansion pairs in
      the form of a mapping from abbreviation to a list of expansions in the
      order they occur. If an abbreviation occurs multiple times but has the
      same expansion in every occurrence, a single expansion entry suffices.
    model_expansion_mapping: An expansion mapping for the model output,
      connecting each token in the input text to the resulting token or phrase
      in the model's output text.

  Returns:
    A list of tuples containing 5 elements:
      - A unique id for the abbreviation being compared
      - The abbreviation string
      - A unique id for the expansion being compared
      - The labeled expansion
      - The model's output expansion
  """
  comparisons = []
  abbrev_id = 0
  for abbrev, model_expansions in sorted(model_expansion_mapping.items()):
    if abbrev in label_abbreviation_expansions:
      label_expansions = label_abbreviation_expansions[abbrev]
      if len(label_expansions) == 1 and len(model_expansions) > 1:
        label_expansions = list(label_expansions) * len(model_expansions)
    else:
      label_expansions = [abbrev] * len(model_expansions)
    for expansion_id, (label_expansion, model_expansion) in enumerate(
        zip(label_expansions, model_expansions)):
      if abbrev == label_expansion == model_expansion: continue
      comparisons.append(
          (abbrev_id, abbrev, expansion_id, label_expansion, model_expansion))
    abbrev_id += 1
  return comparisons


def _compute_true_positive_detection(row):
  return 1 if ((row["abbreviation"] != row["label_expansion"])
               and (row["model_expansion"] != row["abbreviation"])) else 0


def _compute_false_positive_detection(row):
  return 1 if ((row["abbreviation"] == row["label_expansion"])
               and (row["model_expansion"] != row["abbreviation"])) else 0


def _compute_false_negative_detection(row):
  return 1 if ((row["model_expansion"] != row["label_expansion"])
               and (row["model_expansion"] == row["abbreviation"])) else 0


def _compute_correct_expansion(row):
  return 1 if (_compute_true_positive_detection(row)
               and (row["model_expansion"] == row["label_expansion"])) else 0


def _compute_incorrect_expansion(row):
  return 1 if (_compute_true_positive_detection(row)
               and (row["model_expansion"] != row["label_expansion"])) else 0


def _compute_row_indicators(row):
  return (
      _compute_false_negative_detection(row),
      _compute_false_positive_detection(row),
      _compute_true_positive_detection(row),
      _compute_incorrect_expansion(row),
      _compute_correct_expansion(row)
  )


def _add_label_model_comparison_cols(
    df, abbreviation_expansions
    ):
  """Adds columns for the expansion-level comparison bt labels and model."""
  df.loc[:, "model_expansion_mapping"] = df.apply(
      lambda row: expansion_attribution_lib.create_expansion_mapping(  # pylint: disable=g-long-lambda
          orig_text=row["raw_input"],
          expanded_text=row["model_output"],
          label_abbreviation_expansions=row["label_abbreviation_expansions"],
          abbreviation_expansions_dict=abbreviation_expansions),
      axis=1)
  df.loc[:, "label_model_comparisons"] = df.apply(
      lambda row: _generate_label_model_comparisons(  # pylint: disable=g-long-lambda
          row["label_abbreviation_expansions"],
          row["model_expansion_mapping"]),
      axis=1)
  df = df.explode("label_model_comparisons")
  comparison_cols = [
      "abbreviation_id", "abbreviation", "expansion_id", "label_expansion",
      "model_expansion"]
  df[comparison_cols] = pd.DataFrame(
      df["label_model_comparisons"].tolist(), index=df.index)
  df = df.drop(columns=["label_model_comparisons"])
  return df


def _add_indicator_cols(df):
  df.loc[:, "indicators"] = df.apply(_compute_row_indicators, axis=1)
  indicator_cols = [
      "false_negative_detection", "false_positive_detection",
      "true_positive_detection", "incorrect_expansion", "correct_expansion"]
  df[indicator_cols] = pd.DataFrame(df["indicators"].tolist(), index=df.index)
  df = df.drop(columns=["indicators"])
  return df


def _add_equivalency_indicator_cols(
    df,
    abbrev_expansion_pair_equivalencies
    ):
  """Adds metrics which treat clinically equivalent expansions as correct."""
  def _get_equivalency_tuple(row):
    return ((row["abbreviation"],) +
            tuple(sorted([row["label_expansion"], row["model_expansion"]])))
  df.loc[:, "is_equivalency"] = df.apply(
      lambda row: (  # pylint: disable=g-long-lambda
          _get_equivalency_tuple(row) in abbrev_expansion_pair_equivalencies),
      axis=1)
  df.loc[:, "correct_expansion_equiv"] = df["correct_expansion"]
  df.loc[:, "incorrect_expansion_equiv"] = df["incorrect_expansion"]
  df.loc[df["is_equivalency"], "correct_expansion_equiv"] = 1
  df.loc[df["is_equivalency"], "incorrect_expansion_equiv"] = 0
  return df


def _compute_detection_recall(df):
  true_positives = np.sum(df["true_positive_detection"])
  false_negatives = np.sum(df["false_negative_detection"])
  return true_positives / (true_positives + false_negatives)


def _compute_detection_precision(df):
  true_positives = np.sum(df["true_positive_detection"])
  false_positives = np.sum(df["false_positive_detection"])
  return true_positives / (true_positives + false_positives)


def _compute_expansion_accuracy(df):
  correct_expansions = np.sum(df["correct_expansion"])
  incorrect_expansions = np.sum(df["incorrect_expansion"])
  return correct_expansions / (correct_expansions + incorrect_expansions)


def _compute_expansion_accuracy_equiv(df):
  correct_expansions = np.sum(df["correct_expansion_equiv"])
  incorrect_expansions = np.sum(df["incorrect_expansion_equiv"])
  return correct_expansions / (correct_expansions + incorrect_expansions)


def compute_df_metrics(df):
  """Computes aggregate performance metrics on a dataframe of expansions."""
  metrics = {
      "detection_recall": _compute_detection_recall(df),
      "detection_precision": _compute_detection_precision(df),
      "expansion_accuracy": _compute_expansion_accuracy(df),
      "expansion_accuracy_equiv": _compute_expansion_accuracy_equiv(df),
  }
  metrics["total_accuracy"] = (
      metrics["detection_recall"] * metrics["expansion_accuracy"])
  metrics["total_accuracy_equiv"] = (
      metrics["detection_recall"] * metrics["expansion_accuracy_equiv"])
  return metrics


def generate_model_expansion_indicators(
    abbreviation_expansions,
    input_data_df,
    model_output_df,
    expansion_equivalencies
    ):
  """Adds FP/TP/FN detections and correct/incorrect expansions."""
  input_data_df.loc[:, "label_abbreviation_expansions"] = (
      input_data_df["label"].apply(
          _generate_abbreviation_expansions_from_label))
  input_output_df = input_data_df.merge(
      model_output_df, how="inner", on=["input_id"])
  input_output_df = _add_label_model_comparison_cols(
      input_output_df, abbreviation_expansions)
  input_output_df = _add_indicator_cols(input_output_df)
  input_output_df = _add_equivalency_indicator_cols(
      input_output_df, expansion_equivalencies)
  return input_output_df
