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

"""Reproduces the results from "Knowledge Based Machine Translation Evaluation".

https://research.google/pubs/pub49521/
"""

import collections
import itertools
import json
import math
import os
import sys

import pandas as pd

KOBE = 'KoBE'
KOBE_REFERENCE_BASED = 'KoBE reference based'
ANNOTATED_SENTENCE = 'annotated_sentence'
ENTITIES = 'entities'
ID = 'id'
SOURCE_COUNT = 'source_count'
CANDIDATE_COUNT = 'candidate_count'
MATCH_COUNT = 'match_count'
SOURCE = 'src'
REFERENCE = 'ref'
LANGUAGE_PAIR = 'lp'
SYSTEM = 'system'

TO_ENGLISH = 'to_english'
FROM_ENGLISH = 'from_english'
NO_ENGLISH = 'no_english'
ALL_LANGUAGE_PAIRS = 'all'

IBM1_MORPHEME = 'ibm1-morpheme'
IBM1_POS4GRAM = 'ibm1-pos4gram'
BLEU = 'BLEU'


def init_language_pairs():
  """Initializes a dictionary containing all language pairs.

  THe language pairs are from the WMT19 metrics track.

  Returns:
     A dictionary mapping 4 categories to lists of corresponding language pairs:
     "to_english", "from_english", "no_english" and "all".
  """

  language_pairs = {
      TO_ENGLISH: [
          'de-en', 'fi-en', 'gu-en', 'kk-en', 'lt-en', 'ru-en', 'zh-en'
      ],
      FROM_ENGLISH: [
          'en-cs', 'en-de', 'en-fi', 'en-gu', 'en-kk', 'en-lt', 'en-ru', 'en-zh'
      ],
      NO_ENGLISH: ['de-cs', 'de-fr', 'fr-de']
  }

  language_pairs[ALL_LANGUAGE_PAIRS] = list(
      itertools.chain.from_iterable(language_pairs.values()))

  return language_pairs


def get_system_name(annotations_filename, language_pair):
  """Extracts the MT system name from the annotations file name.

  For consistency, the annotations file naming are similr to the naming of the
  files in WMT19. The format of the submitted systems is -
  'newstest2019.'{system_name}.{source_language}-{target_language}'. The format
  of the source is -
  'newstest2019.'{source_language}{target_language}-src.{source_language}'. The
  format of the referemce is -
  'newstest2019.'{source_language}{target_language}-ref.{target_language}'
  Args:
    annotations_filename: Name of the file containing the annotations.
    language_pair: The language pair of the relevant file.

  Returns:
    The underlying machine translation system name.
  """
  if '-ref.' in annotations_filename:
    return REFERENCE
  elif '-src.' in annotations_filename:
    return SOURCE
  else:
    # Extract the system name assuming the format described above. For example
    # for 'online-Y-0' system in 'fr-de' the filename will be
    # 'newstest2019.online-Y.0.fr-de'.
    prefix_to_strip = 'newstest2019.'
    suffix_to_strip = f'.{language_pair}'
    return annotations_filename[len(prefix_to_strip):-len(suffix_to_strip)]


def read_all_annotations(data_path, language_pairs):
  """Loads all the annotated data.

  Args:
    data_path: Path to directory containing the data.
    language_pairs: All the language pairs.

  Returns:
    Dictionary keyed by language pair and system name containing the
    annotations.
    Annotations are in JSON format and contain all the entities that were
    detected in each sentence. Each entity has an identifier (from the knowledge
    base), start and end positions in the sentence and the text of the mention.
  """
  annotated_data_path = os.path.join(
      data_path, 'annotations/wmt19-submitted-data/newstest2019')
  print(f'reading annotated data from - {annotated_data_path}\n')

  lp_and_sys_name_to_annotations = collections.defaultdict(
      collections.defaultdict)
  for language_pair in language_pairs:
    print(f'{language_pair} read annotations\n')
    for annotations_filename in os.listdir(
        os.path.join(annotated_data_path, language_pair)):
      with open(
          os.path.join(annotated_data_path, language_pair,
                       annotations_filename)) as reader:
        annotations = json.load(reader)
      system_name = get_system_name(annotations_filename, language_pair)
      lp_and_sys_name_to_annotations[language_pair][system_name] = annotations
  return lp_and_sys_name_to_annotations


def count_matches(source_annotations, candidate_annotations):
  """Computes entity matches between the sources and the candidates.

  For details about the annotation format consult the "Data" section in the
  'README.md' or the 'supplementary material' section in the paper.

  Args:
    source_annotations: Source entity annotations. When calculating the
      reference-less version of KoBE metric - expected to be called with
      entities from the source sentences (the one that we translate from). When
      calculating the reference-based version of KoBE metric - expected to be
      called with entities from the reference translation of the source
      sentences (the correct translation of the source sentence).
    candidate_annotations: Entity annotations for the candidate translation.

  Returns:
    Counter that contains the matches counts together with the total entities
    count in the source and the candidate.
  """
  counters = collections.Counter()
  assert len(source_annotations) == len(candidate_annotations)
  for source_annotation, candidate_annotation in zip(source_annotations,
                                                     candidate_annotations):
    # Calculate candidate_ids_counter and update the global candidate entities
    # counter. candidate_ids_counter will be used to make sure we match each
    # candidate entity only once.
    candidate_ids_counter = collections.Counter()
    for candidate_entity in candidate_annotation[ENTITIES]:
      counters.update([CANDIDATE_COUNT])
      candidate_ids_counter.update([candidate_entity[ID]])

    # For each source entity - count matches with the candidate and update the
    # global match counter. In addition - update the global source entities
    # counter.
    for src_entity in source_annotation[ENTITIES]:
      counters.update([SOURCE_COUNT])
      if src_entity[ID] in candidate_ids_counter:
        counters.update([MATCH_COUNT])
        candidate_ids_counter.subtract([src_entity[ID]])
        if candidate_ids_counter[src_entity[ID]] == 0:
          candidate_ids_counter.pop(src_entity[ID])

  return counters


def calc_percentage(total, matched):
  return round(float(matched * 100) / total, 2)


def entity_count_penalty(source_entities_count, candidate_entities_count):
  """Calculates the entity count penalty (ECP) introduced in the paper.

  ECP is inspired by BLEU’s brevity penalty. ECP penalizes systems producing c
  entities if c is more than twice the number of entities in the source.

  Args:
    source_entities_count: Count of source entities.
    candidate_entities_count: Count of candidate entities.

  Returns:
    ECP value.
  """
  s, c = float(source_entities_count), float(candidate_entities_count)
  return 1.0 if c < 2 * s else math.exp(1 - c / (2 * s))


def get_scores_dictionary(all_annotations):
  """Calculates scores and arranges them in a dictionary.

  Args:
    all_annotations: Dictionary keyed by language pair and system name
      containing the annotations.

  Returns:
    Dictionary with KoBE scores for each system in each language pair. It
    contains 2 keys - 'KoBE' and 'KoBE reference based'. Each containing the
    scores keyed by by language pair and system name.
  """
  scores = {
      KOBE: collections.defaultdict(collections.defaultdict),
      KOBE_REFERENCE_BASED: collections.defaultdict(collections.defaultdict)
  }
  for language_pair, annotations in all_annotations.items():
    print(f'{language_pair} calculate scores\n')
    all_system_names = [
        key for key in annotations if key not in {SOURCE, REFERENCE}
    ]
    for system_name in all_system_names:
      source_based_matches = count_matches(
          annotations[SOURCE][ANNOTATED_SENTENCE],
          annotations[system_name][ANNOTATED_SENTENCE])
      reference_based_matches = count_matches(
          annotations[REFERENCE][ANNOTATED_SENTENCE],
          annotations[system_name][ANNOTATED_SENTENCE])

      scores[KOBE][language_pair][system_name] = calc_percentage(
          source_based_matches[SOURCE_COUNT],
          source_based_matches[MATCH_COUNT]) * entity_count_penalty(
              source_based_matches[SOURCE_COUNT],
              source_based_matches[CANDIDATE_COUNT])
      scores[KOBE_REFERENCE_BASED][
          language_pair][system_name] = calc_percentage(
              reference_based_matches[SOURCE_COUNT],
              reference_based_matches[MATCH_COUNT]) * entity_count_penalty(
                  reference_based_matches[SOURCE_COUNT],
                  reference_based_matches[CANDIDATE_COUNT])

  return scores


def arrange_scores_in_a_data_frame(scores):
  """Arranges the score dictionary in a data frame.

  Args:
    scores: Dictionary with KoBE scores for each system in each language pair.
      It contains 2 keys - 'KoBE' and 'KoBE reference based'. Each containing
      the scores keyed by by language pair and system name.

  Returns:
    Dataframe with KoBE scores for each system in each language pair.
  """
  data_frame_rows = {
      LANGUAGE_PAIR: [],
      SYSTEM: [],
      KOBE: [],
      KOBE_REFERENCE_BASED: []
  }

  assert set([KOBE, KOBE_REFERENCE_BASED]) <= set(scores.keys())

  for language_pair in scores[KOBE].keys():
    for system in scores[KOBE][language_pair].keys():
      data_frame_rows[LANGUAGE_PAIR].append(language_pair)
      data_frame_rows[SYSTEM].append(system)
      data_frame_rows[KOBE].append(scores[KOBE][language_pair][system])
      data_frame_rows[KOBE_REFERENCE_BASED].append(
          scores[KOBE_REFERENCE_BASED][language_pair][system])
  kobe_scores = pd.DataFrame(
      data_frame_rows,
      columns=[LANGUAGE_PAIR, SYSTEM, KOBE, KOBE_REFERENCE_BASED])
  return kobe_scores


def calculate_kobe_scores(all_annotations):
  """Calculates scores and arranges them in a data frame.

  Args:
    all_annotations: Dictionary keyed by language pair and system name
      containing the annotations.

  Returns:
    Dataframe with KoBE scores for each system in each language pair.
  """
  scores = get_scores_dictionary(all_annotations)
  kobe_scores = arrange_scores_in_a_data_frame(scores)
  return kobe_scores


def get_wmt19_results(data_path, language_pairs):
  """Loads the submitted metric scores from WMT19.

  Args:
    data_path: Path to directory containing the data.
    language_pairs: All the language pairs.

  Returns:
    A dataframe with all the relevant submitted scores.
  """
  submitted_scores_path = os.path.join(
      data_path, 'wmt19_metric_task_results/sys-level_scores_metrics.csv')
  all_submitted_scores = pd.read_csv(open(submitted_scores_path), sep=',')
  del all_submitted_scores['Unnamed: 0']

  # Extract only the relevant scores. We extract all the scores of the metrics
  # that were submitted to the Quality Estimation (QA) as a metric track
  # (Task 3 in the QA shared tasks). More details in
  # https://www.aclweb.org/anthology/W19-5401v2.pdf.
  # In addition we also extract the BLEU scores.
  relevant_submitted_scores = all_submitted_scores[[
      LANGUAGE_PAIR, 'DA', SYSTEM, BLEU, IBM1_MORPHEME, IBM1_POS4GRAM, 'LASIM',
      'LP', 'UNI', 'UNI+', 'USFD', 'USFD-TL', 'YiSi-2', 'YiSi-2_srl'
  ]].loc[all_submitted_scores[LANGUAGE_PAIR].isin(set(language_pairs))]
  return relevant_submitted_scores


def merge_scores_with_wmt19_scores(kobe_scores, submitted_wmt19_scores):
  """Loads the submitted metric scores from WMT19.

  Args:
    kobe_scores: A dataframe with all KoBE scores.
    submitted_wmt19_scores: A dataframe with all the relevant submitted scores.

  Returns:
    Dataframe with all scores (KoBE + WMT19 metrics).
  """
  merged_scores = pd.merge(
      submitted_wmt19_scores,
      kobe_scores,
      how='outer',
      left_on=[LANGUAGE_PAIR, SYSTEM],
      right_on=[LANGUAGE_PAIR, SYSTEM])

  # Filter out 'online-B.0' in 'gu-en' (scores are missing).
  merged_scores = merged_scores[(merged_scores[SYSTEM] != 'online-B.0') |
                                (merged_scores[LANGUAGE_PAIR] != 'gu-en')]
  return merged_scores


def get_correlation(all_scores, metric_mame, language_pair):
  """Computes correlations with human direct assessment (DA) for a given metric and language pair.

  For particular metric in particular language pair.

  Args:
    all_scores: Dataframe with all the scores (KoBE + WMT19 metrics).
    metric_mame: Name of the metric
    language_pair: The language pair.

  Returns:
    Correlation with human DA.
  """
  langauge_pair_scores = all_scores.loc[all_scores[LANGUAGE_PAIR] ==
                                        language_pair]
  return round(
      langauge_pair_scores['DA'].corr(langauge_pair_scores[metric_mame]), 3)


def get_correlations(all_scores, language_pairs):
  """Computes correlations with human direct assessment (DA) for all metrics and language pairs.

  For all metrics in all language pairs.

  Args:
    all_scores: Dataframe with all the scores (KoBE + WMT19 metrics).
    language_pairs: All the language pairs.

  Returns:
    A Dataframe containing correlations with human DA.
  """

  dataframe_rows = {lp: [] for lp in language_pairs}
  dataframe_rows['metric'] = []
  for metric in all_scores.columns:
    if metric not in {LANGUAGE_PAIR, 'DA', SYSTEM}:
      dataframe_rows['metric'].append(metric)
      for language_pair in language_pairs:
        dataframe_rows[language_pair].append(
            get_correlation(all_scores, metric, language_pair))
  correlations = pd.DataFrame(
      dataframe_rows, columns=['metric'] + language_pairs).set_index('metric')
  correlations.index.name = None
  return correlations


def generate_results_table(correlations, language_pairs):
  """Generates results table in the same format as in the paper.

  Relevant for the main reference-less metric.

  Args:
    correlations: Dataframe containing the correlations.
    language_pairs: All the language pairs.

  Returns:
    A Dataframe containing the correlations in the same format as in the paper.
  """
  results_table = correlations[language_pairs].drop([KOBE_REFERENCE_BASED
                                                    ]).dropna(how='all')
  results_table.fillna(value='--', inplace=True)
  return results_table


def print_results(correlations, language_pairs):
  """Prints results in the same format as in the paper.

  Args:
    correlations: A Dataframe containing the correlations.
    language_pairs: A dictionary of language pairs divided into 3 categories:
      "to_english", "from_english" and "no_english".

  Returns:
    A Dataframe containing the correlations in the same format as in the paper.
  """
  to_english_results = generate_results_table(correlations,
                                              language_pairs[TO_ENGLISH])
  from_english_results = generate_results_table(correlations,
                                                language_pairs[FROM_ENGLISH])
  no_english_results = generate_results_table(correlations,
                                              language_pairs[NO_ENGLISH])

  # Drop ibm1 results from 'to-english' and 'from-english'.
  # No reported result for 'to-english' and 'from-english' in WMT19.
  ibm1_metrics_names = [IBM1_MORPHEME, IBM1_POS4GRAM]
  to_english_results = to_english_results.drop(ibm1_metrics_names)
  from_english_results = from_english_results.drop(ibm1_metrics_names)

  # Print all the reference-less results.
  print(f'{to_english_results}\n')
  print(f'{from_english_results}\n')
  print(f'{no_english_results}\n')

  # Print the 'to-english' reference-based comparison to BLEU.
  to_english_metric_results = correlations.loc[[BLEU, KOBE_REFERENCE_BASED
                                               ]][language_pairs[TO_ENGLISH]]
  print(f'\n{to_english_metric_results}\n')


def main(data_path):
  language_pairs = init_language_pairs()

  # Read all annotated WMT19 data.
  all_annotations = read_all_annotations(data_path,
                                         language_pairs[ALL_LANGUAGE_PAIRS])

  # Calculate the KoBE scores.
  kobe_scores = calculate_kobe_scores(all_annotations)

  # Read other metrics results from WMT19.
  submitted_wmt19_scores = get_wmt19_results(data_path,
                                             language_pairs[ALL_LANGUAGE_PAIRS])

  # Merge KoBE results to a single table.
  all_scores = merge_scores_with_wmt19_scores(kobe_scores,
                                              submitted_wmt19_scores)

  # Calculate all metrics correlations with human direct assessment (DA).
  correlations = get_correlations(all_scores,
                                  language_pairs[ALL_LANGUAGE_PAIRS])

  # Print the results as presented in the paper.
  print_results(correlations, language_pairs)


if __name__ == '__main__':
  main(sys.argv[1])
