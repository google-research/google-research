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

"""The class for benchmark saving the dataset & labels, and evaluation functions."""

import collections
import json
import os
from typing import Any, Optional, Union

from vrdu.match_utils import AddressMatch  # pylint: disable=unused-import
from vrdu.match_utils import DateMatch  # pylint: disable=unused-import
from vrdu.match_utils import DefaultMatch  # pylint: disable=unused-import
from vrdu.match_utils import Entity
from vrdu.match_utils import GeneralStringMatch  # pylint: disable=unused-import
from vrdu.match_utils import Match
from vrdu.match_utils import NameMatch  # pylint: disable=unused-import
from vrdu.match_utils import NumericalStringMatch  # pylint: disable=unused-import
from vrdu.match_utils import PriceMatch  # pylint: disable=unused-import

# The {Type}Match is called implicitly by globals().get('{Type}Match'), and
# pylint can't detect this implicit call, and that's why we disable the
# unused-import warnings above.


class DataUtils:
  """The class `DataUtils` is to load the dataset.

  This class loads the labeled dataset including:
    1. the OCR results of the documents in the dataset;
    2. the annotated entities in the documents;
    3. the train/test/valid splits of the experiment.
  """

  def __init__(self,
               base_dirpath,
               split_filepath = None):
    """The initialization loads the dataset.

    Loads:
    1. meta.json: A json-format file saving the name of the dataset and more
    detailed features of each entity name, including the corresponding match
    function and appearance pattern.
    2. dataset.jsonl: Each line of this file is in json format, with attributes
    corresponding to the filename, doc_path (relative to dataset_path), ocr, and
    annotations.

    Args:
      base_dirpath: The path to a directory containing files/folders describing
        an annotated dataset.
      split_filepath: The path to a json-format file involving three filename
        lists for train/valid/test splits.
    """
    meta_filepath = os.path.join(base_dirpath, 'meta.json')
    self.entity_name_to_match_func, self.appearance_pattern_to_entity_names = (
        self.load_entity_name_info(meta_filepath)
    )

    jsonl_filepath = os.path.join(base_dirpath, 'dataset.jsonl')
    self.doc_path, self.ocr, self.annotations = self.load_dataset(
        jsonl_filepath, base_dirpath)

    if split_filepath is None:
      self.train_filenames, self.valid_filenames, self.test_filenames = (
          None,
          None,
          None,
      )
    else:
      self.update_splits(split_filepath)

  def update_splits(self, split_filepath):
    """Updates the train/valid/test splits from the json file.

    This is used when we want to change the train/test/valid splits for new
    experiment settings or when we initialize the DataUtils object.

    Args:
      split_filepath: The path to a json-format file involing three filename
        lists for train/valid/test splits, in the format as, { 'train':
        ['001.pdf', '002.pdf', ...], 'valid': ['101.pdf', '102.pdf', ...],
        'test': ['201.pdf', '202.pdf', ...] }
    """
    with open(split_filepath, 'r') as f:
      splits = json.load(f)
    self.train_filenames = list(splits['train'])
    self.valid_filenames = list(splits['valid'])
    self.test_filenames = list(splits['test'])

  def load_entity_name_info(
      self, meta_filepath):
    """Loads match function and other schema information for each entity name.

    The entity_name_to_match_func maps the entity_name to the corresponding
    match function; the appearance_pattern_to_entity_names denotes the lists of
    entity names for different appearance pattern, including 'repeated',
    'unrepeated', and other nested entity names.

    Args:
      meta_filepath: A json-format file saving the name of the dataset, the
        mapping between entity_name to the name of the match func, and the
        mapping between entity_name to string indicating the appearance pattern,
        e.g., { 'entity_name_to_match_func':  {'file_date': DateMatch,}
        'entity_name_appearance_pattern': { 'file_date': 'unrepeated',
        'staff_name': 'repeated', 'program_name': 'line-item', 'program-time':
        'line-item' } }. where entity names of same types will be grouped into
        the `appearance_pattern_to_entity_names` as output.

    Returns:
      entity_name_to_match_func: A dictionary mapping the entity name to the
        corresponding {Type}Match, e.g., { 'file_date': DateMatch },
      appearance_pattern_to_entity_names: A dictionary mapping the
        appearance_pattern to a list of entity names, e.g., { 'unrepeated':
        ['file_date', ...], 'repeated': ['product_name', ...], 'line_item':
        [...] }
    """
    with open(meta_filepath, 'r') as f:
      meta_info = json.load(f)
    entity_name_to_match_func = {
        entity_name: globals().get(func_name, DefaultMatch) for entity_name,
        func_name in meta_info['entity_name_to_match_func'].items()
    }
    entity_appearance_pattern = meta_info.get('entity_appearance_pattern', {
        entity_name: 'unrepeated' for entity_name in entity_name_to_match_func
    })

    # Group the entity names with the same appearance patterns.
    appearance_pattern_to_entity_names = collections.defaultdict(list)
    for entity_name, appearance_pattern in entity_appearance_pattern.items():
      appearance_pattern_to_entity_names[appearance_pattern].append(entity_name)
    if 'repeated' not in appearance_pattern_to_entity_names:
      appearance_pattern_to_entity_names['repeated'] = []
    if 'unrepeated' not in appearance_pattern_to_entity_names:
      appearance_pattern_to_entity_names['unrepeated'] = []

    return entity_name_to_match_func, appearance_pattern_to_entity_names

  def load_dataset(
      self, jsonl_filepath, base_dirpath
  ):
    """Loads the file_path, ocr, annotations from the jsonl file.

    Args:
      jsonl_filepath: The path to the json_line file. Each line is in
        json-format with attributes corresponding to the filename, file_path
        (relative to base_dirpath), ocr, and annotations, e.g., { 'filename':
        'just-a-file.pdf', 'file_path' './pdfs/just-a-file.pdf'  # based on the
        dataset_path. 'ocr': { a dictionary containing OCR information },
        'annotations': [ a list of annotated entities ] }  where the OCR info
        format is a dictionary of the ocr results, a. ocr['text']: a string of
        text contents in reading order. b. ocr['pages']: a list of information
        of each page, where more detailed ocr results are provided, including
        the tokens, lines, paragraphs, and blocks. Each region
        (token/line/paragraph/block) records the text contents, the start/end
        indices in the reading order sequence, the bounding box, and the
        orientation.  and the annotated entity format is a list of entity items,
        involving the entity name and several groundtruth candidates. The
        extracted entity only needs to match one of the candidates. (See the
        definition of generic type `Entity`.) e.g., [('file_date', [('July 1,
        2022', (0, 1.1, 2.2, 3.3, 4.4), [(0, 12)]), ('07/01/2022', (1, 11.1,
        12.2, 13.3, 14.4), [(10, 19)]),...] ) ]
      base_dirpath: The path to the base directory.

    Returns:
      doc_path: A dictionary mapping the filename to the path to the document.
      ocr: A dictionary mapping the filename to the OCR results.
      annotations: A dictionary mapping the filename to the list of annotated
        entity items.
    """
    doc_path = {}
    ocr = {}
    annotations = {}

    with open(jsonl_filepath, 'r') as f:
      for line in f:
        line = line.strip()
        json_doc = json.loads(line)
        filename = json_doc['filename']

        doc_path[filename] = os.path.join(base_dirpath, json_doc['file_path'])
        ocr[filename] = json_doc['ocr']

        annotations[filename] = [
            convert_list_to_tuple_for_entity_items(json_entity_item)
            for json_entity_item in json_doc['annotations']
        ]

    return doc_path, ocr, annotations


def is_nested_entity(
    entity_name):
  """"Nested entity name should be a tuple or list, instead of a string."""
  if isinstance(entity_name, (tuple, list)):
    return True
  elif isinstance(entity_name, str):
    return False
  raise ValueError('Unknown data format.')


def get_nested_entity_name(
    entity_name_tuple,
    nested_entity_name_dict):
  """Gets the nested entity name according to the entity name tuple.

  A nested entity includes several entities as components. This function
  checks the given entity names in the tuple and figures out which nested entity
  they belong to.

  Args:
    entity_name_tuple: A tuple of strings. Each string is an entity name.
    nested_entity_name_dict: A dictionary mapping the entity name to the
      included entity names.

  Returns:
    The corresponding nested entity name.

  Raises:
    <Any>: The nested entity name cannot match with any entity short name.
  """
  entity_name_tuple = set(entity_name_tuple)
  for nested_entity_name, sub_entity_names in nested_entity_name_dict.items():
    if entity_name_tuple.issubset(set(sub_entity_names)):
      return nested_entity_name
  raise ValueError(
      'The entity name tuple cannot match with any nested entity name.')


def convert_list_to_tuple_for_entity_items(
    json_entity_item
):
  """Converts the entity_item from json format to required format.

  The json format cannot save `tuple` but uses `list` instead. This function
  is to convert the entity items in json format back to the required format,
  aka, convert the `list` into `tuple`.

  The entity items support simple entities and nested entities.

  Args:
    json_entity_item: The entity item loaded from json files, e.g., ['date',
      [['07/01/2022', [0,1.1,2.2,3.3,4.4], [[0,10]]]]]

  Returns:
    The fields of the entity, the bounding box, the segments are converted
    into tuples, which is defined by the generic type `Entity`, e.g.,
    ('date', [
        ('07/01/2022', (0,1.1,2.2,3.3,4.4), [(0,10)])
        ]
    )
  """

  def convert_list_to_tuple_for_entity(json_entity):
    """Converts the entity from json format to the generic type `Entity`.

    Args:
      json_entity: For example, ['07/01/2022', [0,1.1,2.2,3.3,4.4], [[0,10]]]

    Returns:
      For example, ('07/01/2022', (0,1.1,2.2,3.3,4.4), [(0,10)])
    """
    entity_text = json_entity[0]
    entity_bbox = tuple(json_entity[1])
    entity_segments = [tuple(seg) for seg in json_entity[2]]
    entity = (entity_text, entity_bbox, entity_segments)
    assert Match.is_entity(entity), entity
    return entity

  entity_name = json_entity_item[0]
  entity_list = []

  # Branch for nested entities; nested entity only has one groundtruth
  # candidate, so there is only one element in the entity_list.
  if is_nested_entity(entity_name):
    entity_name = tuple(entity_name)
    json_entity = json_entity_item[1][0]
    nested_entity = tuple(
        convert_list_to_tuple_for_entity(json_sub_entity)
        for json_sub_entity in json_entity)
    entity_list.append(nested_entity)
  # Branch for simple entities
  else:
    for json_entity in json_entity_item[1]:
      entity_list.append(convert_list_to_tuple_for_entity(json_entity))

  entity_item = (entity_name, entity_list)
  return entity_item


def remove_redundant_unrepeated_entities(
    doc_extractions,
    unrepeated_entity_names):
  """Only keeps the first extracted entity for each unrepeated entity name.

  The entity names include repeated and unrepeated ones. For unrepeated entity
  names, the model should only keep one extraction item as the most confident
  one. When the model does not have such a strategy in the algorithm and
  extracts multiple entities for an unrepeated entity name, we only keep the
  first one as the extraction result. In the future, the models can propose
  customized algorithm to choose the most confident entity.

  Args:
    doc_extractions: A list of entity items from extraction results.
    unrepeated_entity_names: A list of unrepeated entity names and the result
      only has one entity for each entity name in this list.

  Returns:
    There is only one entity available for each unrepeated entity names in the
    output list. The repeated entities remain the same.
  """

  selected_extractions = []
  selected_entity_names = set()

  for ex_entity_item in doc_extractions:
    entity_name = ex_entity_item[0]
    if entity_name not in unrepeated_entity_names:
      selected_extractions.append(ex_entity_item)
    else:
      if entity_name not in selected_entity_names:
        selected_entity_names.add(entity_name)
        selected_extractions.append(ex_entity_item)
  return selected_extractions


def group_repeated_entities_into_nested_entities(
    doc_extractions, repeated_entity_names):
  """Divides and groups the repeated entity list into line items.

  Given a list of entities, we first select the entities with the entity names
  to be nested and use the first repeated nested entity name to divide the list
  into several spans and each span will be converted into a nested entity.

  For example, [A, B, C, B, C, A, C, C] -> [(A, B, C), (B, C, A), (C), (C)].
  When meeting an entity name that already appears in the previous nested
  entity, we start to construct a new nested entity.

  Args:
    doc_extractions: A list of entity items from extraction results.
    repeated_entity_names: A list of entity names that needs to be nested.

  Returns:
    The repeated entities with the nested entity names will be grouped into
    several nested entities and other entities will remain unchanged.
  """
  unchanged_entity_items = []
  repeated_entity_items = []

  for entity_item in doc_extractions:
    entity_name = entity_item[0]
    if entity_name in repeated_entity_names:
      repeated_entity_items.append(entity_item)
    else:
      unchanged_entity_items.append(entity_item)

  if not repeated_entity_items:
    all_entity_items = unchanged_entity_items
  else:
    repeated_entity_groups = [[]]
    grouped_repeated_entity_names = set()

    for entity_item in repeated_entity_items:
      entity_name = entity_item[0]
      if entity_name not in grouped_repeated_entity_names:
        grouped_repeated_entity_names.add(entity_name)
        repeated_entity_groups[-1].append(entity_item)
      else:
        grouped_repeated_entity_names = {entity_name}
        repeated_entity_groups.append([entity_item])

    nested_entity_items = []
    for entity_group in repeated_entity_groups:
      # The format of entity item:
      # (entity_name, (entity_text, entity_bbox, entity_segments))
      nested_entity_name = tuple(item[0] for item in entity_group)
      nested_entity = tuple(item[1] for item in entity_group)
      nested_entity_items.append((nested_entity_name, nested_entity))

    all_entity_items = unchanged_entity_items + nested_entity_items

  return all_entity_items


def get_matching_result_per_doc(
    doc_groundtruth,
    doc_extractions,
    entity_name_to_match_func,
    nested_entity_name_dict = None,
    target_entity_names = None,
):
  """Matches each extracted entities in a document with the groundtruth.

  Args:
    doc_groundtruth: A list of groundtruth entity items, in the format as:
      [(entity_name, [entity_1, entity_2, ...]), ...]
    doc_extractions: A list of extracted entity items, in the format as:
      [(entity_name, entity), ...]
    entity_name_to_match_func: A dictionary mapping the entity_name to
      {Type}Match.
    nested_entity_name_dict: A dictionary mapping the nested entity name to the
      included entity names.
    target_entity_names: A set of entity names and we will only consider the
      entities whose entity name is in this set and ignore the rest. If
      target_entity_names is None, we consider all entity names.

  Returns:
    matched_pairs: A list of matched pairs of groundtruth entity and extracted
      entity.
    unmatched_groundtruth: A list of groundtruth items that have no matched
      extractions.
    unmatched_extractions: A list of extraction items that cannot match with any
      groundtruth.
    The precision/recall/f1 can calculated as:
    pre = len(matched_pairs) / (len(matched_pairs) + len(unmatched_extractions))
    rec = len(matched_pairs) / (len(matched_pairs) + len(unmatched_groundtruth))
    f1 = 2 * pre * rec / (pre + rec)
    For example,
      Extraction: X, Y, Z, A_1, A_2, B_1;
      Groundtruth: A, B, C
      (same letter means entities that can be matched with each other)
      matched pairs: [(A, A_1), (B, B_1)]
      unmatched groundtruth: [C]
      unmatched extractions: [A_2, X, Y, Z].
  """

  def is_target_entity_name(entity_item, target_entity_names,
                            nested_entity_name_dict):
    """Checks whether the entity_item is in included by target_entity_names."""
    entity_name = entity_item[0]
    # When the entity name is nested, we need to first get the real entity name
    # of it with `get_nested_entity_name` and check if this name is in the
    # `target_entity_names`.
    if is_nested_entity(entity_name):
      nested_entity_name = get_nested_entity_name(entity_name,
                                                  nested_entity_name_dict)
      if nested_entity_name in target_entity_names:
        return True
    if not is_nested_entity(entity_name) and entity_name in target_entity_names:
      return True
    return False

  # Filters the groundtruth and extractions; only keeps the ones with the target
  # entity names.
  if target_entity_names is None:
    filtered_doc_groundtruth = doc_groundtruth
    filtered_doc_extractions = doc_extractions
  else:
    filtered_doc_groundtruth = [
        entity_item for entity_item in doc_groundtruth if is_target_entity_name(
            entity_item, target_entity_names, nested_entity_name_dict)
    ]
    filtered_doc_extractions = [
        entity_item for entity_item in doc_extractions if is_target_entity_name(
            entity_item, target_entity_names, nested_entity_name_dict)
    ]

  matched_pairs = []
  unmatched_groundtruth = []
  unmatched_extractions = []

  # Build the mapping from the entity name to the entity items.
  gt_dict = collections.defaultdict(list)
  ex_dict = collections.defaultdict(list)

  for gt_entity_item in filtered_doc_groundtruth:
    gt_dict[gt_entity_item[0]] += [gt_entity_item]

  for ex_entity_item in filtered_doc_extractions:
    ex_dict[ex_entity_item[0]] += [ex_entity_item]

  # Missing extractions for some groundtruth items, aka, the entity name exists
  # in the groundtruth but not in the extractions.
  unmatched_gt_entity_names = set(gt_dict.keys()) - set(ex_dict.keys())
  for entity_name in unmatched_gt_entity_names:
    unmatched_groundtruth.extend(gt_dict[entity_name])

  # Unknown extractions, aka, the entity name exists in the extractions but not
  # in the groundtruth.
  unmatched_ex_entity_names = set(ex_dict.keys()) - set(gt_dict.keys())
  for entity_name in unmatched_ex_entity_names:
    unmatched_extractions.extend(ex_dict[entity_name])

  # Matched entity names need further check whether the entities can match.
  matched_entity_names = set(gt_dict.keys()) & set(ex_dict.keys())
  for entity_name in matched_entity_names:
    cur_gt_entity_items = gt_dict[entity_name]
    cur_ex_entity_items = ex_dict[entity_name]
    matched_ex_entity_items = []

    # The for-loops are prepared for the repeated entities where there might
    # be multiple entities for the same entity name. In the unrepeated entity
    # situation, the list only has a single element, so the complexity of the
    # for-loops is about O(n), supposing there might be multiple extracted
    # entities for the same entity name.
    for gt_entity_item in cur_gt_entity_items:
      gt_entity_list = gt_entity_item[1]
      matched_ex_entity_item = None

      for ex_entity_item in cur_ex_entity_items:
        ex_entity = ex_entity_item[1]

        # Unrepeated entities
        if not is_nested_entity(entity_name):
          match_function = entity_name_to_match_func.get(
              entity_name, DefaultMatch)
          if match_function.match(ex_entity, gt_entity_list):
            matched_ex_entity_item = ex_entity_item
            break
        # Nested entities
        else:
          gt_nested_entity = gt_entity_list[0]
          ex_nested_entity = ex_entity

          # Use the dictionary to ignore the mismatch because of the order.
          gt_nested_entity_dict = {
              sub_entity_name:
              gt_sub_entity for sub_entity_name, gt_sub_entity in zip(
                  entity_name, gt_nested_entity)
          }
          ex_nested_entity_dict = {
              sub_entity_name:
              ex_sub_entity for sub_entity_name, ex_sub_entity in zip(
                  entity_name, ex_nested_entity)
          }

          match_flag = True
          for sub_entity_name in gt_nested_entity_dict:
            gt_sub_entity = gt_nested_entity_dict[sub_entity_name]
            ex_sub_entity = ex_nested_entity_dict[sub_entity_name]
            match_function = entity_name_to_match_func.get(
                sub_entity_name, DefaultMatch)
            if not match_function.match(ex_sub_entity, [gt_sub_entity]):
              match_flag = False
              break
          if match_flag:
            matched_ex_entity_item = ex_entity_item
            break

      if matched_ex_entity_item is None:
        unmatched_groundtruth.append(gt_entity_item)
      else:
        matched_pairs.append((gt_entity_item, matched_ex_entity_item))
        matched_ex_entity_items.append(matched_ex_entity_item)

    for ex_entity_item in cur_ex_entity_items:
      if ex_entity_item not in matched_ex_entity_items:
        unmatched_extractions.append(ex_entity_item)

  return matched_pairs, unmatched_groundtruth, unmatched_extractions


def process_model_extractions(
    model_extractions,
    appearance_pattern_to_entity_names,
):
  """Removes the redundant unrepeated entities and groups the nested entities."""
  unrepeated_entity_names = appearance_pattern_to_entity_names['unrepeated']
  nested_entity_names = []
  for pattern in appearance_pattern_to_entity_names:
    if pattern not in ['repeated', 'unrepeated']:
      nested_entity_names.append(appearance_pattern_to_entity_names[pattern])

  processed_model_extraction = {
      'meta': model_extractions['meta'],
      'results': {}
  }
  for filename, doc_extractions in model_extractions['results'].items():
    doc_extractions = remove_redundant_unrepeated_entities(
        doc_extractions, unrepeated_entity_names)
    for entity_names in nested_entity_names:
      doc_extractions = group_repeated_entities_into_nested_entities(
          doc_extractions, entity_names)
    processed_model_extraction['results'][filename] = doc_extractions
  return processed_model_extraction


def evaluate_for_target_entity_names(
    model_extractions,
    benchmark,
    target_entity_names = None):
  """Evaluates the extractions from the model for a set of target entity names.

  Args:
    model_extractions: A dictionary mapping the filename to the list of
      extractions in the document.
    benchmark: A `DataUtils` object containing the annotations and more details
      about entity names, and the evaluation is conducted on the files listed in
      the `benchmark.test_filenames`.
    target_entity_names: A set of entity names to be considered. The evaluation
      result will be an average of the performance of these entity names. If we
      want the performance of nested entities, the nested entity names should be
      included here. When there is only one entity name in this set, the result
      will be performance of this entity name. When all entity names are in this
      set, the result will be the general micro pre/rec/f1. When
      `target_entity_names` is None, all entity names are considered by default.

  Returns:
    A dictionary of micro precision, recall and f1-score of the entity names in
    the `target entity_names`.
  """
  correct_num = 0
  groundtruth_num = 0
  extractions_num = 0

  entity_name_to_match_func = benchmark.entity_name_to_match_func

  nested_entity_name_dict = {}
  for pattern in benchmark.appearance_pattern_to_entity_names:
    if pattern not in ['repeated', 'unrepeated']:
      nested_entity_name_dict[
          pattern] = benchmark.appearance_pattern_to_entity_names[pattern]

  all_extractions = model_extractions['results']
  all_groundtruth = {
      filename: groundtruth
      for filename, groundtruth in benchmark.annotations.items()
      if filename in benchmark.test_filenames
  }

  for filename in all_groundtruth:
    # When the model fails to extract any entities from a certain document, we
    # set the doc_extractions as an empty list.
    doc_extractions = all_extractions.get(filename, [])

    doc_groundtruth = all_groundtruth[filename]

    matched_pairs, unmatched_groundtruth, unmatched_extractions = (
        get_matching_result_per_doc(
            doc_groundtruth,
            doc_extractions,
            entity_name_to_match_func,
            nested_entity_name_dict,
            target_entity_names,
        )
    )

    correct_num += len(matched_pairs)
    groundtruth_num += len(matched_pairs) + len(unmatched_groundtruth)
    extractions_num += len(matched_pairs) + len(unmatched_extractions)

  if extractions_num == 0:
    pre = rec = f1 = 0
  elif groundtruth_num == 0:
    # When there is no groundtruth for the entity name, the result is set as -1,
    # and this should be skipped in the final report. This happens when some
    # entity names do not appear in certain templates.
    pre = 0
    rec = 1
    f1 = 0
  elif correct_num == 0:
    pre = rec = f1 = 0
  else:
    pre = correct_num / extractions_num
    rec = correct_num / groundtruth_num
    f1 = 2 * pre * rec / (pre + rec)

  return {'precision': pre, 'recall': rec, 'f1': f1}


def evaluate(model_extractions,
             benchmark):
  """Computes the precision/recall/f1-score of the model.

  1. micro precision/recall/f1-score.
  2. macro precision/recall/f1-score.
  3. precision/recall/f1-score for each entity name.
  4. precision/recall/f1-score for the unrepeated entity name.

  Args:
    model_extractions: A dictionary mapping the filename to the list of
      extractions in the document.
    benchmark: A `DataUtils` object containing the annotations and more details
      about entity names.

  Returns:
    A dictionary of all the evaluation results.
  """
  all_results = {}

  model_extractions = process_model_extractions(
      model_extractions, benchmark.appearance_pattern_to_entity_names)

  # Micro precision, recall, f1-score
  micro_results = evaluate_for_target_entity_names(model_extractions, benchmark)
  all_results['micro_precision'] = micro_results['precision']
  all_results['micro_recall'] = micro_results['recall']
  all_results['micro_f1'] = micro_results['f1']

  # Precision, recall, f1-score for each entity name
  all_entity_names = benchmark.appearance_pattern_to_entity_names[
      'unrepeated'] + benchmark.appearance_pattern_to_entity_names['repeated']
  all_entity_names += [
      nested_entity_short_name for nested_entity_short_name in
      benchmark.appearance_pattern_to_entity_names
      if nested_entity_short_name not in ['repeated', 'unrepeated']
  ]

  macro_precision = []
  macro_recall = []
  macro_f1 = []
  for entity_name in all_entity_names:
    cur_results = evaluate_for_target_entity_names(model_extractions, benchmark,
                                                   {entity_name})
    all_results[f'{entity_name}_precision'] = cur_results['precision']
    all_results[f'{entity_name}_recall'] = cur_results['recall']
    all_results[f'{entity_name}_f1'] = cur_results['f1']

    macro_precision.append(cur_results['precision'])
    macro_recall.append(cur_results['recall'])
    macro_f1.append(cur_results['f1'])

  # Macro precision, recall, f1-score
  def avg(lst):
    lst = [x for x in lst if x != -1]
    return sum(lst) / len(lst)

  all_results['macro_precision'] = avg(macro_precision)
  all_results['macro_recall'] = avg(macro_recall)
  all_results['macro_f1'] = avg(macro_f1)

  # Unrepeated entity precsion, recall, f1-score
  unrepeated_entity_names = benchmark.appearance_pattern_to_entity_names[
      'unrepeated']
  unrepeated_results = evaluate_for_target_entity_names(
      model_extractions, benchmark, set(unrepeated_entity_names))
  all_results['unrepeated_precision'] = unrepeated_results['precision']
  all_results['unrepeated_recall'] = unrepeated_results['recall']
  all_results['unrepeated_f1'] = unrepeated_results['f1']

  return all_results
