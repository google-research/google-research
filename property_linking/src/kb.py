# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""KBBuilder is a class that builds the knowledge base/NQL context.
"""
import random

from language.nql import util as nql_util
import tensorflow.compat.v1 as tf
from property_linking.src import util

random.seed(0)

# Need to special-case popularity and lang since it carries no information
# Need to special-case category since that's what we're testing
IGNORED_RELATIONS = {
    "i//w/item/popularity",
    "i//w/item/category",
    "i/lang",
    "i/isa",
}


def is_constant(id_string):
  # not foolproof, should really crossreference names dict
  # Or maybe just take out all constant values
  # Takes out IDs and geocoordinates
  return id_string.startswith("c")


class Builder(nql_util.ModelBuilder):
  """Builder of the KB context/graph.
  """

  def __init__(self,
               kb_file,
               cats_file,
               names_file,
               kb_dropout=0.0,
               max_relations=None,
               max_core_size=None,
               max_noncore_size=None,
               min_noncore_cutoff=0,
               max_constants_cutoff=None):

    self.kb_file = kb_file
    self.cats_file = cats_file
    self.names_file = names_file
    self.kb_dropout = kb_dropout
    self.max_relations = max_relations
    self.max_core_size = max_core_size
    self.max_noncore_size = max_noncore_size
    self.min_noncore_cutoff = min_noncore_cutoff
    self.max_constants_cutoff = max_constants_cutoff
    self.id_to_name = {}
    # Make name dict early
    with tf.gfile.GFile(self.names_file) as n_f:
      for line in n_f:
        line_data = line.strip().split("\t")
        self.id_to_name[line_data[1].strip()] = line_data[2].strip()
    tf.logging.info("Done reading <names> file {}".format(len(self.id_to_name)))

  def config_context(self, context, params=None):
    """Configures context (does the building). Called by super.build_context().

    Args:
      context: existing (possibly empty) context object
      params: unused
    """
    # Relations could be anything, automatically generate
    with tf.gfile.GFile(self.cats_file) as cp:
      cp_lines = cp.readlines()
      cat_ids = set([line.split("\t")[0] for line in cp_lines])

    with tf.gfile.GFile(self.kb_file) as kb_f:
      kb_lines = kb_f.readlines()
      kb_f_length = len(kb_lines)
      # filter here so that data cleaning can still have an effect
      kb_lines = [l for l in kb_lines if random.uniform(0, 1) > self.kb_dropout]

      # First get values, so we can get the constants
      values = util.rank_by_frequency(kb_lines, 2)
      # Separate out category labels, clean constants
      tf.logging.info("Total distinct values: {}".format(len(values)))
      values = [value for value in values if value not in cat_ids]

      tf.logging.info("Values filtered by cats: {}".format(len(values)))
      if self.max_constants_cutoff is not None:
        values = [value for i, value in enumerate(values)
                  if (i < self.max_constants_cutoff
                      or (not is_constant(value)))]
      tf.logging.info("Values filtered by constants: {}, taking top {}".format(
          len(values), self.max_noncore_size))
      tf.logging.info("Ranked values: {}".format(
          self.get_names(values[:50])))
      if self.max_noncore_size is not None:
        values = values[self.min_noncore_cutoff:self.max_noncore_size]
      values = set(values)

      relations = util.rank_by_frequency(kb_lines, 0, values=values)
      core_entities = util.rank_by_frequency(kb_lines, 1, use_map=True)

      # Figure out what nodes to prune
      if self.max_relations is not None:
        relations = relations[:self.max_relations]
      tf.logging.info("Ranked relations: {}".format(
          self.get_names(relations[:10])))
      relations = set(relations)
      # Explicitly mark these for removal
      relations -= IGNORED_RELATIONS
      if self.max_core_size is not None:
        core_entities = core_entities[:self.max_core_size]
      core_entities = set(core_entities)
      # declare the KB relations, plus the identity relation
      value_type = "v_t"
      core_type = "id_t"

      for relation in relations:
        context.declare_relation(relation, core_type, core_type)

      lines = []  # To be read by KG loader
      connected_values = set()  # For logging/debugging
      connected_entities = set()  # For logging/debugging
      for line in kb_lines:
        line_data = line.strip().split("\t")
        if (line_data[0] in relations and
            line_data[1] in core_entities and
            line_data[2] in values):
          connected_entities.add(line_data[1])
          connected_values.add(line_data[2])
          lines.append("{}\t{}\t{}".format(
              line_data[0], line_data[2], line_data[1]))

      tf.logging.info("Done reading <kg> file {}/{}".format(len(lines),
                                                            kb_f_length))

    # Determine starting states by making a unary relation out of the values.
    for value in connected_values:
      context.declare_relation(value, value_type, core_type)
      lines.append("{}\t{}\t{}".format(value, value, value))
    tf.logging.info("Done making value unary relations.")

    tf.logging.info("Values: {}".format(
        self.get_names(list(connected_values)[::600])))
    tf.logging.info("Relations: {}".format(
        self.get_names(list(relations)[::10])))
    tf.logging.info("{} relations, {}/{} values, {}/{} core entities".format(
        len(relations), len(connected_values), len(values),
        len(connected_entities), len(core_entities)))
    # load data
    context.load_kg(lines=lines)

    # create the relation group used in rules, which should mirror
    # the rel_t type used in the KG to define info for masking
    context.construct_relation_group("rel_g", "id_t", "id_t")
    context.construct_relation_group("val_g", "v_t", "id_t")

  def get_names(self, symbol_or_symbol_list):
    """Get the names of symbol(s).

    Args:
      symbol_or_symbol_list: either a single symbol or a list of symbols

    Returns:
      names of (each) symbol
    """
    if isinstance(symbol_or_symbol_list, list):
      return [self.id_to_name[symbol] for symbol in symbol_or_symbol_list]
    else:
      return self.id_to_name[symbol_or_symbol_list]

  def contains(self, symbol_or_symbol_list):
    """Get the names of symbol(s).

    Args:
      symbol_or_symbol_list: either a single symbol or a list of symbols

    Returns:
      whether all the symbol(s) are contained in the kg
    """
    if isinstance(symbol_or_symbol_list, list):
      return all([symbol in self.id_to_name
                  for symbol in symbol_or_symbol_list])
    else:
      return symbol_or_symbol_list in self.id_to_name
