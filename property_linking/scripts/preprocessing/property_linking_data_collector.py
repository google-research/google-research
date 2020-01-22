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

"""Create data for property linking experiments from Sling's KB."""

import collections
import time

import sling
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string(
    'stem', '/tmp/proplink', 'stem for all created files')
tf.flags.DEFINE_string(
    'top_categories', 'Q43501', 'comma-separated wikidata KB ids Qxxxx,Qxxx')

tf.flags.DEFINE_string(
    'sling_kb_file',
    '',
    'where to find sling kb')

CATEGORY_ID = 'Q4167836'  # class naming wikidata 'category' lists
INSTANCE_OF_ID = 'P31'  # property linking frame to a class it is instance of
SUBCLASS_OF_ID = 'P279'  # property linking class to a superclass
MIN_CAT_SIZE = 5  # discard category lists that don't have this many instances


class SlingExtractor(object):
  """Extract property-linking data from wikidata with sling.
  """

  def load_kb(self):
    """Load self.names and self.kb.
    """
    tf.logging.info('loading and indexing kb...')
    start = time.time()
    self.kb = sling.Store()
    self.kb.load(FLAGS.sling_kb_file)
    self.kb.freeze()
    tf.logging.info('loading took %.3f sec' % (time.time() - start))
    # these are used a lot
    self.instance_of = self.kb[INSTANCE_OF_ID]
    self.category = self.kb[CATEGORY_ID]
    # just in case
    self.english_cats = self.type_freq = None
    # space for kb construction
    self.collected_edges = collections.defaultdict(set)
    self.collected_names = {}
    self.collected_cat_mems = {}

  def frames(self, filter_category=None, filter_english=None):
    """Iterate over all sling Frames in the KB.


    Args:
      filter_category: if True return category frames;
        if False return non-category frames; if None return all
      filter_english: similar for language = /lang/en

    Yields:
      frames that pass the filters.
    """
    start = time.time()
    for n, f in enumerate(self.kb):
      is_category_frame = self.instance_of in f and f[
          self.instance_of] == self.category
      is_english = 'lang' in f and f['lang'].id == '/lang/en'
      if filter_category is not None and is_category_frame != filter_category:
        pass
      elif filter_english is not None and is_english != filter_english:
        pass
      else:
        yield f
      if n > 0 and not n % 500000:
        tf.logging.info('processed %d frames in %.3f sec' %
                        (n, (time.time() - start)))

  def frames_in_type(self, top_type_id):
    """Iterate over all Wikidata KB frames below some top category.

    Args:
      top_type_id: id of the top category

    Yields:
      frames for instances in the top category, either directly
      or by inclusion.
    """
    if self.type_freq is None:
      self.collect_type_freqs()
    # compute the set of types below this top category
    base_types = set()
    for ty in self.type_freq:
      if self.isa(ty, top_type_id):
        base_types.add(ty)
    tf.logging.info('%d base types of %r name %r' %
                    (len(base_types), top_type_id, self.kb[top_type_id].name))

    # find all frames marked as instances of something in base_types
    for f in self.frames(filter_category=False, filter_english=True):
      if self.instance_of in f:
        f_top_type_id = f[self.instance_of].id
        if f_top_type_id in base_types:
          yield f

  def collect_english_cats(self):
    """Populate a list of KB frames that are English-language categories.

    The list is stored in self.english_cats
    """
    tf.logging.info('collecting english categories')
    self.english_cats = list(
        self.frames(filter_english=True, filter_category=True))

  def collect_type_freqs(self):
    """Populate a dict that maps the id of each KB type to a count.

    The dict is stored in self.type_freq.
    """
    tf.logging.info('collecting type_freq, mapping type_id -> # instances')
    self.type_freq = collections.defaultdict(int)
    for f in self.frames():
      if self.instance_of in f:
        self.type_freq[f[self.instance_of].id] += 1
    # not sure why this is needed - do some things have Null id's?
    del self.type_freq[None]

  def isa(self, type_id, supertype_id):
    """Test if one type is a subtype of another.

    Args:
      type_id: first type
      supertype_id: second type

    Returns:
      True or false
    """
    return supertype_id in self.ancestors(type_id)

  def get_filtered_cat_members(self, cats, top_type_id):
    """Get the members of each of a list of categories.

    Candidates for membership are all objects that are under the type named by
    top_type_id.

    Args:
      cats: a list of category ids
      top_type_id: a type id

    Returns:
      dict mapping category ids in cats to a set of ids for the members
    """
    tf.logging.info(
        'returning filtered membership for %d categories' % len(cats))
    cat_set = set(cats)
    cat_members = collections.defaultdict(set)
    n = m = 0
    for f in self.frames_in_type(top_type_id):
      if '/w/item/category' in f:
        # get all categories for f - might be more than one
        for (key, val) in f:
          if key.id == '/w/item/category' and val in cat_set:
            # this means f.id is in category val.id
            n += 1
            if val.id not in cat_members:
              m += 1
            cat_members[val.id].add(f.id)
            if not n % 10000:
              tf.logging.info('%d non-empty categories now with %d members' %
                              (m, n))
    return cat_members

  def ancestors(self, cat_id):
    """Return set of ids for all superclasses of a category, include itself.

    Args:
      cat_id: id of the category to find ancestors of

    Returns:
      the set of all things in the KB of which the category with id cat_id
      is a subclass.
    """
    result = set()
    subclass_of = self.kb[SUBCLASS_OF_ID]
    self._collect_ancestors(result, subclass_of, cat_id)
    return result

  def _collect_ancestors(self, buf, subclass_of, cat_id):
    if cat_id not in buf:
      buf.add(cat_id)
      for key, val in self.kb[cat_id]:
        if key == subclass_of and val.id:
          self._collect_ancestors(buf, subclass_of, val.id)

  def filter_by_pivot(self, pivot_type_id):
    """Map categories to members.

    Restricted to categories with at least some members under the pivot_type_id
    in the ontology, and only members under this ontology node.

    Args:
      pivot_type_id: a type id

    Returns:
      pair (cat_mems, all_mems) where cat_mems maps category ids to
        sets of member ids, all_mems is the union of all category members.
    """
    all_cat_mems = self.get_filtered_cat_members(self.english_cats,
                                                 pivot_type_id)
    cat_mems = dict(
        (c, ms) for (c, ms) in all_cat_mems.items() if len(ms) >= MIN_CAT_SIZE)
    for c, ms in cat_mems.items():
      self.collected_cat_mems[c] = ms
    # members of the useful categories
    all_mems = set()
    for mem_set in cat_mems.values():
      for mem in mem_set:
        all_mems.add(mem)
    tf.logging.info('%d categories and %d entities under type %r' %
                    (len(cat_mems), len(all_mems), pivot_type_id))
    return cat_mems, all_mems

  def build_subset(self, pivot_type_id):
    _, all_mems = self.filter_by_pivot(pivot_type_id)
    self.collect_all_triples(all_mems)

  def collect_all_triples(self, all_mems):
    """Collect all triples concerning a given set of frames.

    Stores in self.collected_edges[subject] a set of pairs (relation, object).
    Also stores in self.collected_names[subject] a name string.
    Here the stored subject, object, relation are either 'c/string_constant'
      or 'i/frame_id'.

    Args:
      all_mems: a set of frame ids.
    """

    self.ignored_frames = 0
    def _collect_triple(subj_id, rel, obj, intermediate=None):
      """Collect edges/names for a single triple.

      Args:
        subj_id: a frame id
        rel: a relation
        obj: a sling Frame
        intermediate: if true, this was called recursively with a nested
          frame that was linked to by the 'intermediate' relation
      """
      def as_str(x):
        return 'i/' + x.id if isinstance(x, sling.Frame) else 'c/' + str(x)
      def as_name(x):
        return x.name if isinstance(x, sling.Frame) and 'name' in x else str(x)

      extended_subj = as_str(self.kb[subj_id])  # will be called i/Qxxx
      self.collected_names[extended_subj] = self.kb[subj_id].name
      if intermediate is None:
        self.collected_names[as_str(rel)] = as_name(rel)
        self.collected_names[as_str(obj)] = as_name(obj)
        self.collected_edges[extended_subj].add((as_str(rel), as_str(obj)))
      elif isinstance(obj, sling.Frame) and obj.id is None:
        # a double-nested frame
        self.ignored_frames += 1
        if self.ignored_frames < 10:
          tf.logging.info('doubly nested frame %r ignored' % obj.data())
      else:
        # nested frame
        extended_rel = as_str(intermediate) + '^' + as_str(rel)
        self.collected_names[extended_rel] = as_name(
            intermediate) + '^' + as_name(rel)
        self.collected_names[as_str(obj)] = as_name(obj)
        self.collected_edges[extended_subj].add((extended_rel, as_str(obj)))

    # generate triples and collect them
    for m in all_mems:
      for k, v in self.kb[m]:
        # special case, a mediator-like node
        if isinstance(v, sling.Frame) and v.id is None:
          for k1, v1 in v:
            _collect_triple(m, k1, v1, k)
        else:
          _collect_triple(m, k, v)
    tf.logging.info('%d doubly-nested frames ignored' % self.ignored_frames)

  def write_collected(self, names_file, kb_file, cat_file):
    """Write collected entity names and relations to a NQL KB file.

    Args:
      names_file: will store mapping from NQL id to 'name'
      kb_file: will store all other relations
      cat_file: will store categories encoded as TSV file with
        columns category id, category name, and category members
        (members is pipe-separated and contains the NQL ids)
    """
    with open(names_file, 'w') as fp:
      for kb_id, name in self.collected_names.items():
        fp.write('\t'.join(['name', kb_id, name]) + '\n')
    with open(kb_file, 'w') as fp:
      for kb_id, tail_set in self.collected_edges.items():
        for (rel, tail_id) in tail_set:
          fp.write('\t'.join([rel, kb_id, tail_id]) + '\n')
    with open(cat_file, 'w') as fp:
      for c, ms in self.collected_cat_mems.items():
        fp.write(c + '\t' + self.kb[c].name + '\t')
        fp.write('|'.join(ms) + '\n')


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  s = SlingExtractor()
  s.load_kb()
  s.collect_english_cats()
  s.collect_type_freqs()

  tf.logging.info('will write to %s*.tsv' % FLAGS.stem)
  for pivot_type_id in FLAGS.top_categories.split(','):
    tf.logging.info('collecting triples for %r' % pivot_type_id)
    s.build_subset(pivot_type_id)
    s.write_collected(
        FLAGS.stem + '_names.tsv',
        FLAGS.stem + '_kb.tsv',
        FLAGS.stem + '_cats.tsv')

if __name__ == '__main__':
  tf.app.run()
