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

"""Code for running evaluation of an abbreviation expansion model."""
from absl import app
from absl import flags
import pandas as pd
from deciphering_clinical_abbreviations import evaluation as evaluation_lib


_ABBREV_DICT_PATH = flags.DEFINE_string(
    name="abbreviation_dictionary_path", default=None,
    help=("The path to the abbreviation dictionary csv, including the "
          "following columns: "
          "1) abbreviation - the abbreviation string; "
          "2) expansion - the associated expansion string;"), required=True)
_INPUT_DATA_PATH = flags.DEFINE_string(
    name="input_data_path", default=None,
    help=("The path to the input data, including the following columns: "
          "1) input_id - a unique string id for each raw input; "
          "2) raw_input - the raw input string; "
          "3) label - the abbreviation expansions label "
          "string;"), required=True)
_MODEL_OUTPUTS_PATH = flags.DEFINE_string(
    name="model_outputs_path", default=None,
    help=("The path to the model outputs, including the following columns: "
          "1) input_id - a unique string for each raw input, which matches"
          "the values in the input_data_path; "
          "2) model_output - the output of the model for the string "
          "corresponding to the input id"), required=True)
_EXPANSION_EQUIVALENCIES_PATH = flags.DEFINE_string(
    name="expansion_equivalences_path", default=None,
    help=("The path to the expansion equivalencies, including the following "
          "columns: "
          "1) abbreviation - the abbreviation for which the expansion are "
          "associated; "
          "2) equivalent_expansions - the pipe-separated list of expansions "
          "considered clinically equivalent for that abbreviation"
          ), required=True)


FLAGS = flags.FLAGS


def main(_):
  abbrev_dict_path = _ABBREV_DICT_PATH.value
  input_data_path = _INPUT_DATA_PATH.value
  model_outputs_path = _MODEL_OUTPUTS_PATH.value
  expansion_equivalencies_path = _EXPANSION_EQUIVALENCIES_PATH.value
  abbreviation_expansions = (
      evaluation_lib.load_abbreviation_expansion_dictionary(
          abbrev_dict_path,
          abbreviation_as_key=True))
  ambiguous_abbreviations = [
      abbrev for abbrev, expansions in abbreviation_expansions.items()
      if len(expansions) > 1]

  # Load data and outputs.
  input_data_df = pd.read_csv(input_data_path, na_filter=False)
  model_output_df = pd.read_csv(model_outputs_path, na_filter=False)
  expansion_equivalencies = evaluation_lib.load_expansion_equivalencies_set(
      expansion_equivalencies_path)

  # Create indicators (TP/FP/FN, etc.).
  model_expansion_indicators = (
      evaluation_lib.generate_model_expansion_indicators(
          abbreviation_expansions=abbreviation_expansions,
          input_data_df=input_data_df,
          model_output_df=model_output_df,
          expansion_equivalencies=expansion_equivalencies))
  model_expansion_indicators_ambig = model_expansion_indicators[
      model_expansion_indicators["abbreviation"].isin(ambiguous_abbreviations)]
  model_expansion_indicators_unambig = model_expansion_indicators[
      ~model_expansion_indicators["abbreviation"].isin(ambiguous_abbreviations)]

  # Compute metrics.
  all_metrics = evaluation_lib.compute_df_metrics(
      model_expansion_indicators)
  ambiguous_metrics = evaluation_lib.compute_df_metrics(
      model_expansion_indicators_ambig)
  unambiguous_metrics = evaluation_lib.compute_df_metrics(
      model_expansion_indicators_unambig)
  metric_names = [
      "detection_recall", "detection_precision", "expansion_accuracy",
      "expansion_accuracy_equiv", "total_accuracy", "total_accuracy_equiv"]

  # Print results.
  print("\nAmbiguous abbreviations:")
  num_unique_abbrevs = len(
      model_expansion_indicators_ambig["abbreviation"].drop_duplicates())
  num_unique_abbrev_expansions = len(
      model_expansion_indicators_ambig[["abbreviation", "label_expansion"]]
      .drop_duplicates())
  print(f"{num_unique_abbrevs} unique abbreviations")
  print(f"{num_unique_abbrev_expansions} unique abbreviation-expansion pairs")
  print(f"{model_expansion_indicators_ambig.shape[0]} instances")
  for metric_name in metric_names:
    print(f"{metric_name}: {ambiguous_metrics[metric_name]*100:0.1f}%")
  print("\nUnambiguous abbreviations:")
  num_unique_abbrevs = len(
      model_expansion_indicators_unambig["abbreviation"].drop_duplicates())
  num_unique_abbrev_expansions = len(
      model_expansion_indicators_unambig[["abbreviation", "label_expansion"]]
      .drop_duplicates())
  print(f"{num_unique_abbrevs} unique abbreviations")
  print(f"{num_unique_abbrev_expansions} unique abbreviation-expansion pairs")
  print(f"{model_expansion_indicators_unambig.shape[0]} instances")
  for metric_name in metric_names:
    print(f"{metric_name}: {unambiguous_metrics[metric_name]*100:0.1f}%")
  print("\nAll abbreviations:")
  num_unique_abbrevs = len(
      model_expansion_indicators["abbreviation"].drop_duplicates())
  num_unique_abbrev_expansions = len(
      model_expansion_indicators[["abbreviation", "label_expansion"]]
      .drop_duplicates())
  print(f"{num_unique_abbrevs} unique abbreviations")
  print(f"{num_unique_abbrev_expansions} unique abbreviation-expansion pairs")
  print(f"{model_expansion_indicators.shape[0]} instances")
  for metric_name in metric_names:
    print(f"{metric_name}: {all_metrics[metric_name]*100:0.1f}%")
  print()

if __name__ == "__main__":
  app.run(main)
