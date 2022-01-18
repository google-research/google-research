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

"""Utils for loading data and conducting model evaluation."""

from collections import defaultdict  # pylint: disable=g-importing-member
import copy
import random

import numpy as np
import tensorflow.compat.v1 as tf


def data_partition(fname):
  """Creates data partitions.

  Args:
    fname: Path to the text data file. Each line is a pair of user, item id,
    query id and timestamp of the interaction. Note that queries are
    synthetically generated from item category information, hence each item has
    one associated query.
      For example:
      ```1 3 q3 t1_1
         1 2 q2 t1_2
         1 3 q3 t1_3
         2 1 q1 t2_1
         2 2 q2 t2_2
         ...
      ```
      Then for user `1`, the sequence of items, queries and times are
      respectively [3, 2, 3], [q3, q2, q3] and [t1_1, t1_2, t1_3].
      Assume user/item id/query id starts from 1.
  Returns:
    user_train: Training dataset. A dict keyed by user id and valued by an
      ordered list of (item id, query id, time) triplets.
    user_valid: Validation dataset.
    user_test: Test dataset.
    usernum: Total number of users.
    querynum: Total number of queries.
    itemnum: Total number of items.
  """
  usernum = 0
  itemnum = 0
  querynum = 0
  seednum = 0
  seq_len_sum = 0.0
  user_list = defaultdict(list)
  user_train = {}
  user_valid = {}
  user_test = {}

  # Holds unique user-query seeds for randomness control on query dropping
  # component. We later use these seeds to have same randomness when processing
  # the user histories for different purposes such as train/eval/test.
  user_query_seed = defaultdict(lambda: 0)

  # Holds item popularity statistics, which is used for popularity based
  # negative sampling. Each item popularity score starts with one in order to
  # ensure a non-zero probability of sampling.
  item_popularity = defaultdict(lambda: 1)
  with tf.gfile.Open(fname, "r") as f:
    for line in f:  # pylint: disable=g-builtin-op
      u, i, q, t = line.rstrip().split(" ")
      u_q_key = _get_user_query_key(u, q)
      if u_q_key not in user_query_seed:
        # Store unique seed for each user-query pair.
        seednum += 1
        user_query_seed[u_q_key] = seednum
      u = int(u)
      q = int(q)
      t = int(t)
      i = int(i)
      usernum = max(u, usernum)
      querynum = max(q, querynum)
      itemnum = max(i, itemnum)
      item_popularity[i] += 1
      user_list[u].append([i, q, t])

  for user in user_list:
    nfeedback = len(user_list[user])
    seq_len_sum += nfeedback
    if nfeedback < 3:
      user_train[user] = user_list[user]
      user_valid[user] = []
      user_test[user] = []
    else:
      user_train[user] = user_list[user][:-2]
      user_valid[user] = []
      user_valid[user].append(user_list[user][-2])
      user_test[user] = []
      user_test[user].append(user_list[user][-1])

  tf.logging.info(
      "max user: {}, max query: {}, max item: {}, max seed: {}".format(
          usernum, querynum, itemnum, seednum))
  tf.logging.info(
      "Size of user_query_seed dictionary: {}".format(len(user_query_seed)))
  tf.logging.info("Average sequence length: %.2f" %
                  (seq_len_sum / len(user_train)))

  return [
      user_train, user_valid, user_test, usernum, querynum, itemnum,
      user_query_seed, item_popularity
  ]


def _get_user_query_key(u, q):
  return "{}_{}".format(u, q)


def _process_item_list(item_list, maxseqlen):
  """Processes item list.

  This function does two main edits to the item_list.
  1. It does zero padding (from left) to have a fixed maxseqlen size.
  2. It excludes the last item id since the next item (label) associated with it
  falls beyond the training set.

  Example:
    item_list = [(1, q1, t1), (2, q2, t2), (3, q3, t3), (4, q4, t4)]
    If maxseqlen = 6, returns
    x = [0, 0, 0, 1, 2, 3],
    q = [0, 0, q1, q2, q3, q4],
    t = [0, 0, t1, t2, t3, t4],
    y = [0, 0, 1, 2, 3, 4].
    If maxseqlen = 3, returns
    x = [1, 2, 3],
    q= [q2, q3, q4]
    t = [t2, t3, t4],
    y = [2, 3, 4].
  Args:
    item_list: A list of (item id, query id) tuples.
    maxseqlen: Max sequence length.
  Returns:
    x: A list of length `maxseqlen` as the new item sequence.
    q: A list of length 'maxseqlen' as the new query sequence.
    t: A list of length 'maxseqlen' as the new time sequence.
    y: Positive next item for each position.
  """
  x = [0] * maxseqlen
  q = [0] * maxseqlen
  t = [0] * maxseqlen
  y = [0] * maxseqlen

  nxt = item_list[-1]
  idx = maxseqlen - 1
  for i in reversed(item_list[:-1]):
    x[idx] = i[0]
    q[idx] = nxt[1]
    t[idx] = nxt[2]
    y[idx] = nxt[0]
    nxt = i
    idx -= 1
    if idx == -1:
      break
  if idx >= 0:  # Meaning if the max_len isn't achieved in upper loop.
    q[idx] = nxt[1]
    t[idx] = nxt[2]
    y[idx] = nxt[0]
  return x, q, t, y


def _create_vocab_from_querymap(query_map, maxquerylen):
  """Creates a query token vocab and query to token ids map.

  Args:
    query_map: A dict keyed by query id, and valued by the query text.
    maxquerylen: Maximum query length.
  Returns:
    vocab: A dict keyed by token id, and valued by the token index (sorted by
    popularity).
    query_word_ids: A dict keyed by query id, and valued by a list of token ids.
  """
  word_counts = defaultdict(lambda: 0)
  query_lengths = []
  for _, query_text in query_map.items():
    words = query_text.rstrip().split(" ")
    query_lengths.append(len(words))
    for word in words[-maxquerylen:]:
      word_counts[word.lower()] += 1
  tf.logging.info("Max/Min/Avg query length: %.2f %.2f %.2f" %
                  (max(query_lengths), min(query_lengths),
                   (sum(query_lengths) / len(query_lengths)))
                  )
  vocab = {}
  idx = 1  # Reserve 0 for none word id
  for key, _ in sorted(word_counts.items(), key=lambda i: i[1], reverse=True):
    vocab[key] = idx
    idx += 1
  tf.logging.info("Vocab is created. Size: %d (maxquerylen: %d)" %
                  (len(vocab), maxquerylen))

  query_word_ids = defaultdict(list)
  for query_id, query_text in query_map.items():
    words = query_text.rstrip().split(" ")[-maxquerylen:]
    query_word_ids[int(query_id)] = [vocab[word.lower()] for word in words]
  return vocab, query_word_ids


def _left_zero_padded_list(l, maxseqlen):
  """Zero pads the given list (from left) to length of maxseqlen."""
  padded = [0] * maxseqlen
  idx = maxseqlen - 1
  for i in reversed(l):
    padded[idx] = i
    idx -= 1
    if idx == -1:
      break
  return padded


def _drop_query_tokens_randomly(user_id, query_id, user_query_seed,
                                query_word_ids, prob):
  if prob <= 0.:  # We don't drop any tokens.
    return query_word_ids[query_id]
  np.random.seed(user_query_seed[_get_user_query_key(user_id, query_id)])
  return [
      token for token in query_word_ids[query_id] if np.random.rand() > prob
  ]


def presample_popularity_negatives(start, end, size, item_popularity,
                                   num_presamples):
  """Presamples negatives for popularity based negative sampling strategy.

  Args:
    start: Smallest item id to consider while sampling.
    end: Largest item id to consider while sampling.
    size: Number of items to sample per list.
    item_popularity: Dictionary mapping item ids to their popularity score.
    num_presamples: Number of negative sample lists.
  Returns:
    A (nested) list consisting of 'num_presamples' many negative sample item
    lists, each with 'size' many item ids.
  """

  # Assign popularity based probability items.
  items_weights_within_range = np.array([
      item_popularity[i] for i in range(start, end)
  ])
  factor = 1.0 / sum(items_weights_within_range)
  presamples = []
  for _ in range(num_presamples):
    presamples.append(
        list(
            np.random.choice(
                range(start, end),
                size=size,
                p=items_weights_within_range * factor,
                replace=False,
                ),
            ),
        )
  return presamples


def _random_sample_k(k, start, end, exclude):
  """Samples k many ids from [start, end) excluding the items in 'exclude'."""
  sampled = []
  for _ in range(k):
    t = np.random.randint(start, end)
    while t in exclude:
      t = np.random.randint(start, end)
    sampled.append(t)
  return sampled


def _sample_negatives(start, end, size, exclude, presampled_negatives):
  """Samples a list of negative item ids.

  For popularity based sampling, we randomly pick form pre-sampled negatives for
  faster execution. This requires a non-empty presampled_negatives argument.
  If the passed presampled_negatives is an empty list, then we instead randomly
  sample 'size' items.

  Args:
    start: Smallest item id to consider while sampling.
    end: Largest item id to consider while sampling.
    size: Number of items to sample.
    exclude: A set of items to exclude from sampling.
    presampled_negatives: A nested list of presampled negative items. Used with
    popularity based sampling.
  Returns:
    A list of (negative) item ids with length equal to 'size'.
  """
  if not presampled_negatives:
    return _random_sample_k(size, start, end, exclude)
  else:
    # If we already pre-sampled the popularity based negatives, then we will
    # select one uniformly at random. Note that this might result in less than
    # 'size' items after (possibly) removing exclude items.
    sampled_idx = np.random.randint(0, len(presampled_negatives))
    sampled = [
        i for i in presampled_negatives[sampled_idx]
        if i not in exclude
    ]
    # In case any of the presampled items were seen in excluded list, we will
    # re-sample those uniformly at random from the items set. This should
    # generally correspond to a few samples.
    if len(sampled) < size:
      sampled.extend(_random_sample_k(size - len(sampled), start, end, exclude))
    return sampled


def create_tf_dataset(data_dict,
                      batch_size,
                      itemnum,
                      query_map,
                      maxquerylen,
                      maxseqlen,
                      token_drop_prob,
                      user_query_seed,
                      randomize_input=False,
                      random_seed=None):
  """Gets a tf dataset.

  Args:
    data_dict: A dict keyed by user id, and valued by an ordered list of (item
      id,query id) tuples.
    batch_size: Batch size.
    itemnum: num of items to recommend.
    query_map: A dict keyed by query id, and valued by the query text.
    maxquerylen: Maximum query length.
    maxseqlen: Maximum item sequence length.
    token_drop_prob: Probability of dropping a token from a query.
    user_query_seed: Dictionary mapping user-query pairs to unique randomness
    seeds.
    randomize_input: Whether to randomize input.
    random_seed: Random seed for shuffling.
  Returns:
    A `tf.data.Dataset`.
  """
  # Fetch vocab and query_word_ids dict
  vocab, query_word_ids = _create_vocab_from_querymap(query_map, maxquerylen)

  # Preprocess `data_dict`.
  user_ids = []
  item_seq = []  # Holds item ids.
  query_seq = []  # Holds query ids.
  query_words_seq = []  # Holds word ids per query.
  time_seq = []  # Holds integer timestamps.
  label_seq = []  # Holds label (item) ids.
  for user_id, item_list in data_dict.items():
    items, queries, times, labels = _process_item_list(item_list, maxseqlen)
    user_ids.append(user_id)
    item_seq.append(items)
    query_seq.append(queries)
    time_seq.append(times)
    label_seq.append(labels)
    words = []  # Stores word ids for each query
    for query_id in queries:
      if query_id > 0:
        words.append(
            _left_zero_padded_list(
                _drop_query_tokens_randomly(user_id, query_id, user_query_seed,
                                            query_word_ids, token_drop_prob),
                maxquerylen))
      else:
        words.append(_left_zero_padded_list([], maxquerylen))
    query_words_seq.append(words)
  d = tf.data.Dataset.from_tensor_slices({
      "user_ids": user_ids,
      "items": item_seq,  # Item sequence.
      "queries": query_seq,  # Query sequence.
      "query_words": query_words_seq,  # Query word sequence.
      "times": time_seq,  # Time sequence.
      "labels": label_seq,  # Label sequence.
  }).repeat(None)

  if randomize_input:
    d = d.shuffle(batch_size * 100, seed=random_seed)
  d = d.batch(batch_size)

  def _batch_random_negatives(batch_data):
    item_seq = batch_data["items"]
    label_seq = batch_data["labels"]
    random_neg = tf.random.uniform(
        shape=tf.shape(label_seq),
        minval=1,
        maxval=itemnum + 1,
        dtype=item_seq.dtype)
    batch_data["random_neg"] = tf.where(
        tf.math.not_equal(label_seq, 0),  # 0 means missing.
        random_neg,
        tf.zeros_like(random_neg))
    return batch_data

  d = d.map(_batch_random_negatives)
  return d, vocab, query_word_ids


def evaluate(model,
             dataset,
             query_word_ids,
             maxseqlen,
             maxquerylen,
             sess,
             token_drop_prob,
             neg_sample_size,
             presampled_negatives,
             eval_on="test"):
  """Evaluates model on three different data splits: train/valid/test.

  Args:
    model: Model to use for evaluation.
    dataset: A list including data splits and other related variables.
    query_word_ids: A dict keyed by query id, and valued by a list of token ids.
    maxseqlen: Maximum item sequence length.
    maxquerylen: Maximum query length.
    sess: Tf session.
    token_drop_prob: Probability of dropping a token from a query.
    neg_sample_size: Number of negatives to sample for ranking.
    presampled_negatives: A list of pre-sampled negatives. If empty, negatives
    are randomly sampled. If not empty, we randomly pick one from presampled.
    eval_on: Data split to evaluate: train, eval, test.
  Returns:
    Average NDCG@10 and HIT@10 scores.
  """
  [train, valid, test, usernum, _, itemnum,
   user_query_seed, _] = copy.deepcopy(dataset)

  ndcg = 0.0
  ht = 0.0
  valid_user = 0.0
  user_neg_sample_sizes = []

  if usernum > 10000:
    users = random.sample(range(1, usernum + 1), 10000)
  else:
    users = range(1, usernum + 1)
  for u in users:
    # Safety check.
    if u not in train:
      continue
    # Discard users without sufficient history.
    if len(train[u]) < 1 or len(valid[u]) < 1 or len(test[u]) < 1:
      continue

    seq = np.zeros([maxseqlen], dtype=np.int32)
    q_seq = np.zeros([maxseqlen], dtype=np.int32)
    q_words_seq = [[0]*maxquerylen for _ in range(maxseqlen)]
    t_seq = np.zeros([maxseqlen], dtype=np.int32)
    idx = maxseqlen - 1

    if eval_on == "train":
      q_nxt = train[u][-1][1]
      t_nxt = train[u][-1][2]
      idx = maxseqlen
    elif eval_on == "valid":
      q_nxt = valid[u][0][1]
      t_nxt = valid[u][0][2]
    elif eval_on == "test":
      seq[idx] = valid[u][0][0]
      q_seq[idx] = test[u][0][1]
      q_words_seq[idx] = _left_zero_padded_list(
          _drop_query_tokens_randomly(u, test[u][0][1], user_query_seed,
                                      query_word_ids, token_drop_prob),
          maxquerylen)
      t_seq[idx] = test[u][0][2]
      idx -= 1
      q_nxt = valid[u][0][1]
      t_nxt = valid[u][0][2]
    else:
      raise ValueError(
          "You may only select train, valid, or test for evalution")

    for i, q, t in reversed(train[u]):
      #  For the train case, we need to skip the latest item.
      if eval_on == "train" and idx == maxseqlen:
        idx -= 1
        continue
      seq[idx] = i
      q_seq[idx] = q_nxt
      q_words_seq[idx] = _left_zero_padded_list(
          _drop_query_tokens_randomly(u, q_nxt, user_query_seed, query_word_ids,
                                      token_drop_prob), maxquerylen)
      t_seq[idx] = t_nxt

      q_nxt = q
      t_nxt = t
      idx -= 1
      if idx == -1:
        break
    if idx >= 0:  # Meaning if the max_len isn't achieved in upper loop.
      q_seq[idx] = q_nxt  # We need to add one more query due to misalignment.
      q_words_seq[idx] = _left_zero_padded_list(
          _drop_query_tokens_randomly(u, q_nxt, user_query_seed, query_word_ids,
                                      token_drop_prob),
          maxquerylen)  # And its tokens.
      t_seq[idx] = t_nxt

    rated = set(list(zip(*train[u]))[0])
    rated.add(0)
    if eval_on == "train":
      ground_th_item = train[u][-1][0]
    elif eval_on == "valid":
      ground_th_item = valid[u][0][0]
    else:  # At this point, it can only be test.
      ground_th_item = test[u][0][0]
    rated.add(ground_th_item)
    item_ids = [ground_th_item]

    # Sample negatives for evaluation.
    item_ids.extend(
        _sample_negatives(1, itemnum + 1, neg_sample_size, rated,
                          presampled_negatives))
    user_neg_sample_sizes.append(len(item_ids))

    predictions = -model.predict(sess, [u], [seq], [q_seq], [q_words_seq],
                                 [t_seq], item_ids)
    predictions = predictions[0]

    rank = predictions.argsort().argsort()[0]

    valid_user += 1

    if rank < 10:
      ndcg += 1 / np.log2(rank + 2)
      ht += 1

  tf.logging.info("Eval is done. Average negative sample size: {}".format(
      (1.0 * sum(user_neg_sample_sizes)) / len(user_neg_sample_sizes)))
  tf.logging.info("Eval is computed over {} many users.".format(valid_user))

  return ndcg / valid_user, ht / valid_user
