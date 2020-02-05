# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""General purpose functions for property linking.
"""
import collections
import itertools
from language.nql import nql
import numpy as np
import tensorflow.compat.v1 as tf
import property_linking.src.bert_util as bert

FLAGS = tf.flags.FLAGS
# Uses FLAGS
# sitelinks
# layers


def reshape_to_tensor(encodings, lengths):
  """Reshape value_encodings to a tf tensor.

  Args:
    encodings: n * BERT embeddings
    lengths: length n list of sequence lengths for masking

  Returns:
    single tf tensor of dimension BERT_dim * n
  """
  seq_mask = np.stack(lengths).astype(np.float32)
  masked_type_embs = (np.sum(encodings * np.expand_dims(seq_mask, -1),
                             axis=1)
                      / np.expand_dims(np.sum(seq_mask, axis=1), -1))
  return tf.transpose(tf.convert_to_tensor(masked_type_embs), [1, 0])


def k_hot_numpy_array(context, entity_names, type_name, as_matrix=True):
  """A k-hot row vector encoding all entities.

  See one_hot_numpy_array for more details.
  It will raise nql.EntityNameError if any entity cannot be converted,
  and it will silently ignore it

  Args:
    context: context
    entity_names: entity_names
    type_name: type of the entities
    as_matrix: return as (dense) matrix
  Returns:
    A numpy array that's k hot
  """
  all_embs = context.zeros_numpy_array(type_name, as_matrix)
  for entity_name in entity_names:
    try:
      entity_emb = context.one_hot_numpy_array(
          entity_name, type_name, as_matrix)
      all_embs += entity_emb
    except nql.EntityNameError:
      pass
  return all_embs


def rank_by_frequency(kb_lines, column_no, values=None, use_map=False):
  """Rank items in kb_lines. Count only if line's value is in values.

  Args:
    kb_lines: lines of tab-separated triples
    column_no: index of column of interest
    values: optional set used to filter lines in file
    use_map: use a sitelinks file (unfortunately passed in by flag)

  Returns:
    Ranked list of items in column of interest
  """
  all_items = []
  if use_map and FLAGS.sitelinks is not None:
    with tf.gfile.GFile(FLAGS.sitelinks) as sitelinks:
      sitelink_dict = {line.split("\t")[0]: int(line.split("\t")[1])
                       for line in sitelinks}
      rank_set = []
      for line in kb_lines:
        line_items = line.strip().split("\t")
        # Need to do :2 to get rid of i/
        if line_items[column_no][2:] in sitelink_dict:
          rank_set.append((line_items[column_no],
                           int(sitelink_dict[line_items[column_no][2:]])))
      rank_set = set(rank_set)
      return [item for item, _ in sorted(rank_set, key=lambda x: -x[1])]

  for line in kb_lines:
    line_items = line.strip().split("\t")
    if values is None:
      all_items.append(line_items[column_no])
    elif values and line_items[2] in values:
      all_items.append(line_items[column_no])
  return [item for item, _ in collections.Counter(all_items).most_common()]


def set_to_string(nql_set):
  return  ", ".join(["({}, {:.3f})".format(val, weight)
                     for val, weight in nql_set])


def sparse_to_sparse_batch(sparse_examples):
  """Converts a batch of sparse (1 * N) matrices to a single (b * N) matrix.

  Args:
    sparse_examples: List of len b of sparse (1 * N) matrices

  Returns:
    A sparse (b * N) matrix
  """
  # Converts list of (cols, values) to list of ([row, col], value) *per example*
  sparse_indices_and_values = [
      [([row, col], value) for (col, value) in zip(*example)]
      for row, example in enumerate(sparse_examples)]
  # Convert list of ([row, col], value) per example to a single list.
  flat_sparse_indices_and_values = list(
      itertools.chain.from_iterable(sparse_indices_and_values))

  # flat_indices: list of [row, col] indices. row in range(b), col is a valid id
  # flat_values: list of values corresponding to flat_indices.
  # If flat_sparse_indices_and_values is empty, pass shape information forwards
  if not flat_sparse_indices_and_values:
    flat_indices, flat_values = np.empty([0, 2]), []
  else:
    flat_indices, flat_values = zip(*flat_sparse_indices_and_values)
  return flat_indices, flat_values


def convert_to_bert(all_sentences, bert_module, sess, bert_batch_size):
  """Computes BERT encodings of all_sentences.

  Args:
    all_sentences: List of InputFeatures corresponding to sentences
    bert_module: Preloaded BERT module
    sess: Preopened tf session
    bert_batch_size: batch size used to operate BERT

  Returns:
    out_emb: Bert encodings with dim (num_sentences, 1, 768)
    out_len: length of each sentence
  """
  masks = [sentence.input_mask for sentence in all_sentences]
  # Break sentences into smaller batches
  num_batches = int(1 + (len(all_sentences) / bert_batch_size))
  batched_sentences = [all_sentences[bert_batch_size * i:
                                     bert_batch_size * (i+1)]
                       for i in range(num_batches)]
  all_out_embs = []
  for batch_no, sentences in enumerate(batched_sentences):
    all_ids = [sentence.input_ids for sentence in sentences]
    all_masks = [sentence.input_mask for sentence in sentences]
    all_segments = [sentence.segment_ids for sentence in sentences]
    bert_inputs = dict(input_ids=all_ids,
                       input_mask=all_masks,
                       segment_ids=all_segments)
    bert_outputs = bert_module(inputs=bert_inputs,
                               signature="tokens",
                               as_dict=True)
    sequence_output = bert_outputs["sequence_output"]
    # sequence_output is the 12th layer.
    bert_layers = [bert.get_intermediate_layer(sequence_output, 12, int(i))
                   for i in FLAGS.layers]
    out_tf_embs = tf.concat(bert_layers, -1)
    batch_out_emb = out_tf_embs.eval(session=sess)
    all_out_embs.append(batch_out_emb)
    tf.logging.info("BERT conversion: {}/{}".format(batch_no, num_batches))
  out_emb = np.concatenate(all_out_embs, axis=0)
  tf.logging.info("Total BERT embs shape: {}".format(out_emb.shape))
  return (out_emb, masks)


def compute_query_node_overlap(builder, context, queries, overlaps, node_type):
  """Compute token containment between query and node surface forms.

  Args:
    builder: KBBuilder, contains surface forms of graph
    context: builder context
    queries: strings for computing overlap
    overlaps: list of precomputed overlaps, but each element may be None
    node_type: type of nodes for computing overlap

  Returns:
    List of (graph) ids that overlap for each category name
  """
  node_names = builder.get_names(context.get_symbols(node_type))
  reverse_dict = {node_idx: idx for idx, node_idx
                  in enumerate(context.get_symbols(node_type))}
  node_bows = [set(node_name.lower().split()) for node_name in node_names]
  query_sets = [set(query.lower().split()) for query in queries]
  tf.logging.info("Computing overlap between {} nodes and {} queries".format(
      len(node_names), len(query_sets)))
  tf.logging.info("Found {}/{} cached".format(
      len([overlap for overlap in overlaps if overlap is not None]),
      len(overlaps)))
  query_node_indicator = []
  for i, (query, overlap) in enumerate(zip(query_sets, overlaps)):
    if overlap is None or not overlap:
      one_hot = [i for i, node_bow in enumerate(node_bows)
                 if node_bow.issubset(query)]
    else:
      one_hot = sorted([reverse_dict[overlap_node_idx]
                        for overlap_node_idx in overlap
                        if overlap_node_idx in reverse_dict])
    query_node_indicator.append(one_hot)
  tf.logging.info("Done computing overlaps")
  values = [[1.0 for _ in query_node_indicator[i]]
            for i in range(len(queries))]
  return list(zip(query_node_indicator, values))


def create_node_encodings(builder,
                          context,
                          bh,
                          node_type):
  """Create encodings for nodes from graph and tokenizer.

  Args:
    builder: KBBuilder, contains surface forms of graph
    context: builder context
    bh: a BertHelper containing parameters for BERT
    node_type: type of node to compute BERT encodings for

  Returns:
    BERT encodings of unary nodes.
    length of each encoded sequence
  """
  symbol_input = [bert.InputExample(guid=None, text_a=node, label=0)
                  for node in builder.get_names(context.get_symbols(node_type))]

  symbol_features = bert.convert_examples_to_features(symbol_input,
                                                      [0],
                                                      bh.max_query_length,
                                                      bh.tokenizer)
  bert_encodings, seq_lens = convert_to_bert(symbol_features,
                                             bh.module,
                                             bh.session,
                                             bh.bert_batch_size)
  return (bert_encodings, seq_lens)


def filter_line(line, core_entities):
  label_list = line[2].strip().split("|")
  return any(filter_kb(label_list, core_entities))


def filter_kb(label_list, core_entities):
  # Return labels that exist in context
  return [label for label in label_list if label in core_entities]


def get_properties(property_list):
  # Convert "rel,val,..." list to a set of {(rel, val)}
  split_properties = [prop.strip().split(",")
                      for prop in property_list
                      if prop]
  return {(split_property[0], split_property[1])
          for split_property in split_properties}


def create_examples(builder,
                    context,
                    example_file,
                    bh,
                    prune_empty=False):
  """Create examples from file, graph, and tokenizer.

  Args:
    builder: KBBuilder, contains surface forms of graph
    context: builder context
    example_file: path to examples
    bh: a BertHelper containing BERT params
    prune_empty: optional for whether to prune examples now

  Returns:
    value_bert: BERT encodings of value nodes.
    examples: Examples for training.
  """
  # Examples
  core_entities = set(context.get_symbols("id_t"))
  with tf.gfile.GFile(example_file) as f:
    lines = [line.split("\t") for line in f]
    if prune_empty:
      line_length = len(lines)
      lines = [line for line in lines if filter_line(line, core_entities)]
      tf.logging.info("Reduced {} examples to {} after initial filter".format(
          line_length, len(lines)))
    example_names = [line[1] for line in lines]
    example_input = [bert.InputExample(guid=None, text_a=example_name, label=0)
                     for example_name in example_names]
    example_features = bert.convert_examples_to_features(example_input,
                                                         [0],
                                                         bh.max_query_length,
                                                         bh.tokenizer)

    example_labels = [filter_kb(line[2].strip().split("|"), core_entities)
                      for line in lines]

    example_property_ids = [get_properties(line[3].strip().split("|"))
                            if len(line) > 3 else ()
                            for line in lines]
    example_property_names = [[tuple(builder.get_names(list(prop)))
                               for prop in properties
                               if builder.contains(list(prop))]
                              for properties in example_property_ids]
    example_properties = list(zip(example_property_ids,
                                  example_property_names))

    example_string_overlaps = [line[4].strip().split("|")
                               if len(line) > 4 else []
                               for line in lines]
    example_links = [line[5].strip().split("|") if len(line) > 5 else []
                     for line in lines]
    def merge(a, b):
      return list(set(a+b))
    example_overlaps = [merge(a, b) for a, b in zip(example_string_overlaps,
                                                    example_links)]

  example_bert_embs, example_lens = convert_to_bert(example_features,
                                                    bh.module,
                                                    bh.session,
                                                    bh.bert_batch_size)

  example_value_indicator = compute_query_node_overlap(builder,
                                                       context,
                                                       example_names,
                                                       example_overlaps,
                                                       "val_g")
  examples = [((example_bert_embs[i], example_lens[i]), label, example_names[i],
               example_value_indicator[i], example_properties[i])
              for i, label in enumerate(example_labels)]
  if prune_empty:
    examples = [example for example in examples if example[3][0]]
    tf.logging.info(
        "Further pruned to {}/{} examples, no node overlaps.".format(
            len(examples), len(example_labels)))
  return examples


def weighted_nonneg_crossentropy(expr, target, weights=None):
  """A cross entropy operator that is appropriate for NQL outputs.

  Query expressions often evaluate to sparse vectors.  This evaluates cross
  entropy safely.

  Args:
    expr: a Tensorflow expression for some predicted values.
    target: a Tensorflow expression for target values.
    weights: an optional Tensorflow expression for weighting examples.

  Returns:
    Tensorflow expression for cross entropy.
  """
  expr_replacing_0_with_1 = \
     tf.where(expr > 0, expr, tf.ones(tf.shape(input=expr), tf.float32))
  cross_entropies = tf.reduce_sum(
      input_tensor=-target * tf.math.log(expr_replacing_0_with_1), axis=1)
  if weights is not None:
    cross_entropies = cross_entropies * weights
  return tf.reduce_mean(input_tensor=cross_entropies, axis=0)
