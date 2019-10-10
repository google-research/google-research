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

"""Find superclasses.

  Examples:

  closure_inference.py --sling_kb_file <kb> --alsologtostderr

  # for locations
  closure_inference.py --sling_kb_file <kb> --alsologtostderr
     --infile <infile> --closing_rel_id P131
"""

import time

import sling
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string(
    'outfile', '/tmp/closed.tsv', 'created file')
tf.flags.DEFINE_string(
    'infile',
    '',
    'input file')
# probably P131 = administrative region of, or P279 = subclass_of
tf.flags.DEFINE_string(
    'closing_rel_id', 'P279', 'relation to use to close')
tf.flags.DEFINE_string(
    'sling_kb_file',
    '',
    'where to find sling kb')
tf.flags.DEFINE_string(
    'blacklist_substring',
    'metaclass',
    'discard superclasses with this substring in the name')

tf.flags.DEFINE_boolean(
    'trace_closures',
    False,
    'give ridiculously long debug output')


def load_kb():
  """Load self.names and self.kb.

  Returns:
    sling kb
  """
  tf.logging.info('loading and indexing kb...')
  start = time.time()
  kb = sling.Store()
  kb.load(FLAGS.sling_kb_file)
  kb.freeze()
  tf.logging.info('loading took %.3f sec' % (time.time() - start))
  return kb


def closure(kb, closing_relation_id, cat_id):
  """Return set of ids for logical closure of a category/region.

  Args:
     kb: a sling kb
     closing_relation_id: SUBCLASS_OF_ID or REGION_OF_ID
     cat_id: id of the category to find ancestors of

  Returns:
    the set of all things in the KB of which the category with id cat_id
    is a subclass.
  """

  result = set()
  closer = kb[closing_relation_id]
  _collect_ancestors(kb, result, closer, cat_id)

  def blacklisted(qid):
    name = kb[qid].name
    return name and name.find(FLAGS.blacklist_substring) >= 0

  if FLAGS.blacklist_substring:
    return [e for e in result if not blacklisted(e)]
  else:
    return result


def _collect_ancestors(kb, buf, closer, cat_id):
  if cat_id not in buf:
    buf.add(cat_id)
    for key, val in kb[cat_id]:
      if key == closer and val.id:
        _collect_ancestors(kb, buf, closer, val.id)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  kb = load_kb()
  tf.logging.info('will write to %s*.tsv' % FLAGS.outfile)
  tf.logging.info('closing with %s [%s]' % (
      kb[FLAGS.closing_rel_id].name,
      kb[FLAGS.closing_rel_id]))
  with tf.gfile.Open(FLAGS.outfile, 'w') as out_fp:
    for line in tf.gfile.GFile(FLAGS.infile):
      qid = line.strip()
      if qid.startswith('i/'):
        qid = qid[len('i/'):]
      closed = closure(kb, FLAGS.closing_rel_id, qid)
      out_fp.write('\t'.join([qid] + list(closed)) + '\n')
      if FLAGS.trace_closures:
        if len(closed) > 1:
          tf.logging.info('closing %s [%s]' % (kb[qid].name, qid))
          for super_qid in closed:
            tf.logging.info(' ==> %s [%s]' % (kb[super_qid].name, super_qid))

if __name__ == '__main__':
  tf.app.run()
