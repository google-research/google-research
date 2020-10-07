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

"""Tests for multi_resolution_rec.util."""

import collections
import itertools

import tensorflow.compat.v1 as tf
from multi_resolution_rec import util


class UtilTest(tf.test.TestCase):

  def test_process_item_list(self):
    item_list = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
    x, q, t, y = util._process_item_list(item_list, maxseqlen=2)
    self.assertEqual(x, [1, 2])
    self.assertEqual(q, [2, 3])
    self.assertEqual(t, [2, 3])
    self.assertEqual(y, [2, 3])
    x, q, t, y = util._process_item_list(item_list, maxseqlen=5)
    self.assertEqual(x, [0, 0, 0, 1, 2])
    self.assertEqual(q, [0, 0, 1, 2, 3])
    self.assertEqual(t, [0, 0, 1, 2, 3])
    self.assertEqual(y, [0, 0, 1, 2, 3])

  def test_create_tf_dataset(self):
    data_dict = {
        1: [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)],
        2: [(5, 5, 5), (6, 6, 6), (7, 7, 7), (8, 8, 8)],
        3: [(9, 9, 9), (10, 10, 10), (11, 11, 11), (12, 12, 12)]
    }
    dataset, _, _ = util.create_tf_dataset(
        data_dict=data_dict,
        batch_size=2,
        itemnum=1,
        query_map={1: 'test'},
        maxquerylen=1,
        maxseqlen=4,
        token_drop_prob=0,
        user_query_seed=collections.defaultdict(lambda: 0),
        randomize_input=False,
    )

    batch = dataset.make_one_shot_iterator().get_next()
    user_ids, items, queries, times, labels = self.evaluate([
        batch['user_ids'], batch['items'], batch['queries'], batch['times'],
        batch['labels']
    ])
    self.assertAllEqual(user_ids, [1, 2])
    self.assertAllEqual(items, [[0, 1, 2, 3], [0, 5, 6, 7]])
    self.assertAllEqual(queries, [[1, 2, 3, 4], [5, 6, 7, 8]])
    self.assertAllEqual(times, [[1, 2, 3, 4], [5, 6, 7, 8]])
    self.assertAllEqual(labels, [[1, 2, 3, 4], [5, 6, 7, 8]])
    user_ids, items, queries, times, labels = self.evaluate([
        batch['user_ids'], batch['items'], batch['queries'], batch['times'],
        batch['labels']
    ])
    self.assertAllEqual(user_ids, [3, 1])
    self.assertAllEqual(items, [[0, 9, 10, 11], [0, 1, 2, 3]])
    self.assertAllEqual(queries, [[9, 10, 11, 12], [1, 2, 3, 4]])
    self.assertAllEqual(times, [[9, 10, 11, 12], [1, 2, 3, 4]])
    self.assertAllEqual(labels, [[9, 10, 11, 12], [1, 2, 3, 4]])

  def test_create_vocab_from_querymap(self):
    query_map = {
        1: 'food',
        2: 'food snack',
        3: 'food snack granola'
    }
    vocab, query_word_ids = util._create_vocab_from_querymap(
        query_map, maxquerylen=3)
    self.assertAllEqual(vocab, {
        'food': 1,
        'snack': 2,
        'granola': 3
    })
    self.assertAllEqual(query_word_ids, {
        1: [1],
        2: [1, 2],
        3: [1, 2, 3]
    })

  def test_drop_query_tokens_randomly(self):
    users = [1, 2]
    queries = [1, 2]
    query_word_ids = {1: [1, 2, 3], 2: [3, 4, 5, 1]}
    seed = 1
    user_query_pairs = list(itertools.product(users, queries))
    user_query_seed = collections.defaultdict(lambda: 0)
    for pair in user_query_pairs:
      user_query_seed[util._get_user_query_key(*pair)] = seed
      seed += 1

    # Test for prob 0.5.
    prob = 0.5
    u0_q0_1 = util._drop_query_tokens_randomly(users[0], queries[0],
                                               user_query_seed, query_word_ids,
                                               prob)
    u0_q0_2 = util._drop_query_tokens_randomly(users[0], queries[0],
                                               user_query_seed, query_word_ids,
                                               prob)
    u0_q1_1 = util._drop_query_tokens_randomly(users[0], queries[1],
                                               user_query_seed, query_word_ids,
                                               prob)
    u0_q1_2 = util._drop_query_tokens_randomly(users[0], queries[1],
                                               user_query_seed, query_word_ids,
                                               prob)
    u1_q0_1 = util._drop_query_tokens_randomly(users[1], queries[0],
                                               user_query_seed, query_word_ids,
                                               prob)
    u1_q0_2 = util._drop_query_tokens_randomly(users[1], queries[0],
                                               user_query_seed, query_word_ids,
                                               prob)
    u1_q1_1 = util._drop_query_tokens_randomly(users[1], queries[1],
                                               user_query_seed, query_word_ids,
                                               prob)
    u1_q1_2 = util._drop_query_tokens_randomly(users[1], queries[1],
                                               user_query_seed, query_word_ids,
                                               prob)

    self.assertAllEqual(u0_q0_1, u0_q0_2)
    self.assertAllEqual(u0_q1_1, u0_q1_2)
    self.assertAllEqual(u1_q0_1, u1_q0_2)
    self.assertAllEqual(u1_q1_1, u1_q1_2)

if __name__ == '__main__':
  tf.test.main()
