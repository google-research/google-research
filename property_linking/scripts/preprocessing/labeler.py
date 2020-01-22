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

"""Adds labels to property-linking data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import difflib
import random
import sys

from language.nql import nql
from language.nql.dataset import k_hot_array_from_string_list
import numpy as np
import tensorflow as tf
import Tkinter as tk

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'base_dir',
    '',
    'base directory')
flags.DEFINE_string(
    'stem', 'zoo',
    'file stem "foo" for foo_kb.tsv, foo_names.tsv, foo_cats.tsv')
flags.DEFINE_string('output_file', None, 'base directory for output')
flags.DEFINE_integer('sample_size', 10, 'number of categories to sample')
flags.DEFINE_integer('retrieve_top_k', 50,
                     'top-count properties to retrieve with NQL')
flags.DEFINE_integer('label_top_k', 20,
                     'top-count properties to consider in labeling')
flags.DEFINE_boolean('show_relation_names', False, 'label relation.property')


def local_flag_settings(as_dict=False):
  module_dict = FLAGS.flags_by_module_dict()
  d = dict((x.name, x.value) for x in module_dict[sys.argv[0]])
  if as_dict:
    return d
  else:
    return dict2string(d)


def dict2string(d):
  return ' '.join([('%s=%r' % pair) for pair in sorted(d.items())])


def load_kg(base_dir, stem):
  context = nql.NeuralQueryContext()
  names = {}

  tf.logging.info('loading closures so they can be skipped in the KB')
  # TODO(wcohen): this is a little sloppy since we ignore the relation.
  closures = {}

  def load_closures(filename):
    for line in tf.gfile.Open(base_dir + filename):
      parts = line.strip().split('\t')
      closures['i/' + parts[0]] = ['i/' + p for p in parts[1:]]

  load_closures('closed-locations.txt')
  load_closures('closed-categories.txt')

  tf.logging.info('loading categories so they can be skipped in the KB')
  all_category_ids = set()
  for line in tf.gfile.Open(base_dir + stem + '_cats.tsv'):
    cat_id, _, _ = line.strip().split('\t')
    all_category_ids.add('i/' + cat_id)

  rels = set()
  props = set()
  ents = set()
  kg_lines = []
  num_cats_skipped = 0

  tf.logging.info('reading kg')
  for line in tf.gfile.Open(base_dir + stem + '_kb.tsv'):
    rel, head, tail = line.strip().split('\t')
    if tail in all_category_ids:
      num_cats_skipped += 1
    else:
      if rel not in rels:
        context.declare_relation(rel, 'ent_t', 'prop_t')
        rels.add(rel)
      ents.add(head)
      # if tail is something like a location of a concept
      # with superconcepts, add all the containing locations
      # or superconcepts
      tails = closures.get(tail, [tail])
      for t in tails:
        props.add(t)
        kg_lines.append('\t'.join([rel, head, t]) + '\n')
  tf.logging.info('loaded %d kb lines skipped %d categories-related props' %
                  (len(kg_lines), num_cats_skipped))

  tf.logging.info('reading names')
  context.declare_relation('prop_name', 'prop_t', 'name_t')
  context.declare_relation('ent_name', 'ent_t', 'name_t')
  context.declare_relation('rel_name', 'rel_t', 'name_t')
  for line in tf.gfile.Open(base_dir + stem + '_names.tsv'):
    _, head, tail = line.strip().split('\t')
    names[head] = tail
    if head in props:
      kg_lines.append('\t'.join(['prop_name', head, tail]))
    if head in ents:
      kg_lines.append('\t'.join(['ent_name', head, tail]))
    if head in rels:
      kg_lines.append('\t'.join(['rel_name', head, tail]))
  tf.logging.info('loading %d kg lines', len(kg_lines))
  context.load_kg(lines=kg_lines)
  tf.logging.info('loaded')
  context.construct_relation_group('rel_g', 'ent_t', 'prop_t')
  return context, names


class Labeler(object):

  def __init__(self, base_dir, stem):
    self.context, self.names = load_kg(base_dir, stem)
    self.any_rel = self.context.all('rel_g')
    self.sess = tf.Session()

  def sample_cats(self, k, base_dir, stem):
    cat_lines = []
    for line in tf.gfile.Open(base_dir + stem + '_cats.tsv'):
      cat_lines.append(line)
      sample = {}
    for line in random.sample(cat_lines, k):
      (cat_id, cat_text, cat_member_ids) = line.strip().split('\t')
      k_hot_vec = k_hot_array_from_string_list(
          self.context, 'ent_t', ['i/' + m for m in cat_member_ids.split('|')])
      k_hot_mat = np.reshape(k_hot_vec, (1, self.context.get_max_id('ent_t')))
      cat_members = self.context.as_nql(k_hot_mat, 'ent_t')
      sample[cat_id] = (cat_text, cat_members)
    return sample

  def lexical_sim(self, cat_text, prop_name):
    r1 = difflib.SequenceMatcher(None, cat_text, prop_name).ratio()
    r2 = len(prop_name) / float(len(cat_text))
    # add a little smoothing
    return (r1 + 1.0) / (r2 + 2.0)

  def label(self, cat_id, cat_text, cat_members):
    tf.logging.info('building labeler pane for %s %s' % (cat_id, cat_text))
    frequent_prop_ids = cat_members.follow(self.any_rel).eval(
        self.sess, as_top=FLAGS.retrieve_top_k)
    max_count = max([count for (_, count) in frequent_prop_ids])
    max_sim = max([
        self.lexical_sim(cat_text, prop_id)
        for (prop_id, _) in frequent_prop_ids
    ])
    tuples = []
    g = self.context.get_group('rel_g')
    for (prop_id, count) in frequent_prop_ids:
      if prop_id not in self.names:
        # skip this - probably something very general, anyway
        # it is never directly linked
        pass
      else:
        # find the relation(s) connecting prop_id and the cat_members
        if FLAGS.show_relation_names:
          prop = self.context.one(prop_id, 'prop_t')
          connecting_triples = cat_members.follow(
              g.subject_rel, -1) & prop.follow(g.object_rel, -1)
          connecting_rel_dict = connecting_triples.follow(g.relation_rel).eval(
              self.sess)
          rel_names = [self.names.get(r) for r in connecting_rel_dict]
        else:
          rel_names = ['']
        prop_name = self.names.get(prop_id, '***%s***' % prop_id)
        sim_score = self.lexical_sim(cat_text, prop_name) / max_sim
        count_score = float(count) / float(max_count)
        combined_score = 2.0 / (1.0 / sim_score + 1.0 / count_score)
        for rel in rel_names:
          tuples.append((combined_score, count, sim_score,
                         rel + '.' + prop_name, prop_id))
        if len(tuples) > FLAGS.label_top_k:
          break

    root = tk.Tk()
    root.geometry('1600x1000')
    font = ('Arial Bold', 16)
    root.title('Labeler')
    label = tk.Label(root, text='%s [%s]' % (cat_text, cat_id), font=font)
    label.grid(column=0, row=0)
    states = {}

    def add_checkbox(key, text):
      states[key] = tk.BooleanVar()
      states[key].set(False)
      checkbox = tk.Checkbutton(root, text=text, var=states[key], font=font)
      checkbox.grid(sticky='w', column=0, row=len(states))

    add_checkbox('_missing', 'Some necessary constraints are missing')
    add_checkbox('_redundant', 'Some checked constraints are redundant')
    add_checkbox('_approximate',
                 'Some checked constraints are close but not perfect')
    add_checkbox('_error', 'Something else is wrong with this example')
    for tup in sorted(tuples, reverse=True):
      add_checkbox(tup[-1], '%f\t%f\t%f\t%s [%s]' % tup)
    root.mainloop()
    return [prop_id for prop_id in states if states[prop_id].get()]

  def inspect(self, cat_id, cat_text, cat_members):
    print()
    print('category', cat_id, cat_text)
    freq_props = cat_members.follow(self.any_rel).prop_name().eval(
        self.sess, as_top=20)
    max_count = max([count for (_, count) in freq_props])
    max_sim = max([
        self.lexical_sim(cat_text, prop_name) for (prop_name, _) in freq_props
    ])
    tuples = []
    for (prop_name, count) in freq_props:
      sim_score = self.lexical_sim(cat_text, prop_name) / max_sim
      count_score = float(count) / float(max_count)
      # geometric_mean = 2.0 / (1.0 / sim_score + 1.0 / count_score)
      mean_score = (sim_score + count_score) / 2.0
      tuples.append((mean_score, count_score, sim_score, prop_name))
    for tup in sorted(tuples, reverse=True):
      print(' > %f\t%f\t%f\t%s' % tup)

  def inspect1(self, cat_id, cat_text, cat_members):
    print('category', cat_id, cat_text)
    print(' mid> ', cat_members.eval(self.sess))
    print(' m_n> ', cat_members.ent_name().eval(self.sess))
    print(' pid> ', cat_members.follow(self.any_rel).eval(self.sess, as_top=20))
    print(
        ' p_n> ',
        cat_members.follow(self.any_rel).prop_name().eval(self.sess, as_top=20))


def main(unused_args):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('local flags %r' % local_flag_settings)
  labeler = Labeler(FLAGS.base_dir, FLAGS.stem)
  cat_sample = labeler.sample_cats(FLAGS.sample_size, FLAGS.base_dir,
                                   FLAGS.stem)

  if FLAGS.output_file is None:
    output_file = '/tmp/labels_' + FLAGS.stem + '.txt'
  else:
    output_file = FLAGS.output_file
  tf.logging.info('writing labels to %r' % output_file)
  with tf.gfile.Open(output_file, 'a') as out_fp:
    for cat_id in cat_sample:
      (cat_text, cat_members) = cat_sample[cat_id]
      labels = labeler.label(cat_id, cat_text, cat_members)
      out_fp.write('\t'.join([cat_id] + sorted(labels)) + '\n')


if __name__ == '__main__':
  tf.app.run()
