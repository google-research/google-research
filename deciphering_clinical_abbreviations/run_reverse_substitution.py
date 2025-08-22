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

"""Code to carry out reverse substitution on a document dataset."""

import collections
import itertools
import os
from absl import app
from absl import flags
import pandas as pd
from deciphering_clinical_abbreviations import evaluation as evaluation_lib
from deciphering_clinical_abbreviations import text_processing
from deciphering_clinical_abbreviations import text_sampling
from deciphering_clinical_abbreviations import tokenizer as tokenizer_lib


_DOCUMENT_DATASET_PATH = flags.DEFINE_string(
    name="document_dataset_path", default=None,
    help=("The path to the document dataset csv, including a column "
          "called 'document_text' which contains the text of each document."),
    required=True)
_ABBREV_DICT_PATH = flags.DEFINE_string(
    name="abbreviation_dictionary_path", default=None,
    help=("The path to the abbreviation dictionary csv, including the "
          "following columns: "
          "1) abbreviation - the abbreviation string; "
          "2) expansion - the associated expansion string."), required=True)
_SAVE_FILEPATH = flags.DEFINE_string(
    name="save_filepath", default=None,
    help="The filepath to save the output files to.",
    required=True)
_EXPECTED_REPLACEMENTS_PER_EXPANSION = flags.DEFINE_integer(
    name="expected_replacements_per_expansion", default=None,
    help=("The number of expected replacements per expansion. This is used to "
          "calculate a unique replacement rate for each expansion, for which "
          "the formula is rate=(expected_replacements_per_expansion/N) where N "
          "is the total number of instances of that expansion in the dataset."),
    required=True)
_MIN_SNIPPETS_PER_SUBSTITUTION_PAIR = flags.DEFINE_integer(
    name="min_snippets_per_substitution_pair", default=None,
    help=("The minimum number of snippets containing a substitution pair that "
          "must be sampled for the final dataset. If a snippet contains any "
          "pair for which the number of previous snippets sampled containing "
          "that pair is less than this number, the snippet is sampled."),
    required=True)
_EXCLUSION_STRINGS = flags.DEFINE_list(
    name="exclusion_strings",
    default=[],
    help=("Strings whose presence in a snippet should lead to the exclusion "
          "of that snippet from the dataset"),
    required=False)
_RANDOM_SEED = flags.DEFINE_integer(
    name="random_seed", default=1,
    help=("An optional random seed for determinism. Default value is 1. "
          "Setting this flag to -1 will result in non-determinism."),
    required=False)


def main(_):
  random_seed = _RANDOM_SEED.value if _RANDOM_SEED.value != -1 else None
  abbrev_dict_path = _ABBREV_DICT_PATH.value
  document_dataset_path = _DOCUMENT_DATASET_PATH.value

  print("\nLoading datasets.")
  abbreviations_by_expansion = (
      evaluation_lib.load_abbreviation_expansion_dictionary(
          abbrev_dict_path, abbreviation_as_key=False))
  expansion_re = tokenizer_lib.create_word_finder_regex(
      sorted(abbreviations_by_expansion.keys(), key=len, reverse=True))
  document_dataset = pd.read_csv(document_dataset_path, na_filter=False)
  print("\t- Number of expansions in dictionary: "
        f"{len(abbreviations_by_expansion)}")
  print(f"\t- Document count: {document_dataset.shape[0]}")

  print("\nProcessing snippets.")
  snippets = text_processing.generate_snippets_from_notes(
      document_dataset["document_text"], min_char_len=40, max_char_len=200)
  snippets_dataset = pd.DataFrame({"snippet_text": snippets})
  if _EXCLUSION_STRINGS.value:
    snippets_dataset = snippets_dataset[
        ~(snippets_dataset["snippet_text"].str.contains(
            "|".join(_EXCLUSION_STRINGS.value)))]
  snippets_dataset = snippets_dataset.sort_values("snippet_text")
  snippets_dataset["expansion_spans"] = (
      snippets_dataset["snippet_text"].apply(
          lambda x: text_processing.find_query_spans_in_text(x, expansion_re)))
  num_containing_expansions = snippets_dataset[
      snippets_dataset["expansion_spans"].apply(bool)].shape[0]
  print(f"\t- Total number of snippets: {snippets_dataset.shape[0]}")
  print("\t- Number of snippets containing expansions: "
        f"{num_containing_expansions} "
        f"({(num_containing_expansions / snippets_dataset.shape[0]) * 100:0.1f}"
        "%)")
  unique_expansions = set(itertools.chain(*snippets_dataset["expansion_spans"]))
  print("\t- Number of unique expansions found in snippets: "
        f"{len(unique_expansions)}")

  print("\nApplying reverse substitution.")
  expansion_counts = collections.Counter()
  for row_tuple in snippets_dataset.itertuples(index=False):
    for expansion, spans in row_tuple.expansion_spans.items():
      expansion_counts[expansion] += len(spans)
  substitute_probs_for_expansion = {
      expansion: min(1, _EXPECTED_REPLACEMENTS_PER_EXPANSION.value / count)
      for expansion, count in expansion_counts.items()}
  snippets_dataset[["abbreviated_snippet_text", "label_span_expansions"]] = (
      snippets_dataset.apply(lambda x: text_processing.reverse_substitute(  # pylint: disable=g-long-lambda
          target_string=x["snippet_text"],
          expansions=x["expansion_spans"],
          abbreviations_by_expansion=abbreviations_by_expansion,
          substitute_probs_for_expansion=substitute_probs_for_expansion,
          seed=random_seed), axis=1, result_type="expand"))
  snippets_dataset["label_abbreviation_expansions"] = (
      snippets_dataset.apply(
          lambda x: text_processing.generate_abbreviation_expansion_pair_labels(  # pylint: disable=g-long-lambda
              abbreviated_snippet_text=x["abbreviated_snippet_text"],
              label_span_expansions=x["label_span_expansions"]), axis=1))

  print("\nDownsampling snippets.")
  snippets_dataset["pairs"] = snippets_dataset.apply(
      lambda x: set(  # pylint: disable=g-long-lambda
          [(x["abbreviated_snippet_text"][start:end], expansion)
           for (start, end), expansion in x["label_span_expansions"].items()]),
      axis=1)
  sampled_snippets, substitution_counts = (
      text_sampling.sample_n_instances_per_contained_value(
          examples=snippets_dataset,
          contained_values_col_name="pairs",
          n_per_value=_MIN_SNIPPETS_PER_SUBSTITUTION_PAIR.value,
          seed=random_seed))
  substitution_counts_df = pd.DataFrame({
      "abbreviation": [k[0] for k in substitution_counts.keys()],
      "expansion": [k[1] for k in substitution_counts.keys()],
      "count": substitution_counts.values(),
  })
  print(f"\t- Final dataset size: {sampled_snippets.shape[0]}")
  print("\t- Number of unique abbreviations: "
        f"{substitution_counts_df['abbreviation'].drop_duplicates().shape[0]}")
  print("\t- Number of unique abbreviation-expansion pairs: "
        f"{len(substitution_counts)}")
  print("\t- Total number of substitutions: "
        f"{substitution_counts_df['count'].sum()}")

  print("\nSaving outputs.")
  sampled_snippets_path = os.path.join(_SAVE_FILEPATH.value, "dataset.csv")
  substitution_counts_path = os.path.join(
      _SAVE_FILEPATH.value, "substitution_counts.csv")
  if not os.path.exists(_SAVE_FILEPATH.value):
    os.makedirs(_SAVE_FILEPATH.value)
  sampled_snippets.to_csv(sampled_snippets_path, index=True)
  substitution_counts_df.to_csv(substitution_counts_path, index=False)

  print("\nComplete.")


if __name__ == "__main__":
  app.run(main)
