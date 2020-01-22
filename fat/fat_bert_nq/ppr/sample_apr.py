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

"""Sample file to show PPR output for apr_lib.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from fat.fat_bert_nq.ppr import apr_lib

flags = tf.flags
FLAGS = flags.FLAGS


class SampleApr(object):
  """This is a sample APR test, which accepts entities and prints the facts."""

  def sample_apr(self):
    """sample apr function for printing facts."""
    FLAGS.full_wiki = True
    FLAGS.apr_dir = 'Directory Name'
    apr = apr_lib.ApproximatePageRank()
    seeds = [
        'Q7755', 'Q878070', 'Q428148', 'Q679847', 'Q2609670', 'Q174834',
        'Q188628'
    ]
    unique_facts = apr.get_facts(
        seeds, topk=200, alpha=0.9, seed_weighting=True)
    facts = sorted(unique_facts, key=lambda tup: tup[1][1], reverse=True)
    nl_facts = ' . '.join([
        str(x[0][0][1]) + ' ' + str(x[1][0][1]) + ' ' + str(x[0][1][1])
        for x in facts
    ])
    tf.logging.info('Extracted facts: %s', nl_facts)
