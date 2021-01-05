# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Runs shilling attack experiment in Section 4 of associated manuscript."""


# Imports
from __future__ import print_function

import collections
import copy
import json
import os
import random
import time
from call_glove import GloVe
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from glove_util import count_cooccurrences
import numpy
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale
import tensorflow.compat.v1 as tf

# User-defined hyperparameters for the experiment.
DATA_FILE = 'movielens/ml-100k/u.data'
SAVE_DIR = 'experiment_data/shilling'
NUMBER_OF_EXPERIMENTS = 10
FRACTION_ATTACKERS = 0.05
FRACTION_ATTACKERS_KNOWN = 0.5
NUMBER_TO_SHILL = 10
NUMBER_TARGETS = 1
NUMBER_TO_ATTACK = 100
SEED = 0
NEIGHBORS_TO_EVAL = 20
NUMBER_GLOVE_ITERATIONS = 20
WALK_LENGTH = 5
WINDOW_SIZE = 5
PERCENTILE_THRESHOLD = 90.0
OVERWRITE_EMBEDDINGS = True
DO_EVAL = True

# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=dangerous-default-value
# pylint: disable=invalid-name


if not os.path.isdir(SAVE_DIR):
  os.makedirs(SAVE_DIR)


# MovieLens Utils
def filter_by_weight(graph, min_weight):
  """Ensure edges >= min_weight are all that's left."""
  new_graph = collections.defaultdict(dict)

  for u in graph:
    for i in graph[u]:
      if graph[u][i] >= min_weight:
        new_graph[u][i] = graph[u][i]

  return new_graph


def make_graph_from_array(array):
  graph = collections.defaultdict(dict)

  item_set = set()
  user_set = set()

  for x in range(array.shape[0]):
    user = 'u_' + str(array[x][0])
    item = 'i_' + str(array[x][1])
    graph[user][item] = array[x][2]
    user_set.add(user)
    item_set.add(item)

  return graph, user_set, item_set


def load_movielens_graph(filepath):
  array = numpy.loadtxt(open(filepath), dtype=numpy.int64)
  return make_graph_from_array(array)


def make_movielens_undirected(graph):

  newgraph = collections.defaultdict(dict)

  for x in graph.keys():
    for y in graph[x].keys():
      newgraph[x][y] = graph[x][y]
      newgraph[y][x] = graph[x][y]

  return newgraph


def count_item_popularity(graph):
  cnt = collections.Counter()

  for x in graph:
    for y in graph[x]:
      cnt[y] += 1

  return cnt


def count_item_avg_score(graph):
  score_cnt = collections.Counter()
  item_cnt = collections.Counter()

  for x in graph:
    for y in graph[x]:
      score_cnt[y] += graph[x][y]
      item_cnt[y] += 1

  avg_score = {}

  for x in item_cnt:
    avg_score[x] = score_cnt[x] / float(item_cnt[x])

  return avg_score


# DeepWalk Utils


def load_edgelist(filepath, src_first=True, undirected=True):
  graph = collections.defaultdict(dict)

  with open(filepath) as f:
    for line in f:
      line = line.strip().split()

      if src_first:
        src = line[0]
        dst = line[1]
      else:
        src = line[1]
        dst = line[0]

      graph[src][dst] = 1

      if undirected:
        graph[dst][src] = 1

  return graph


def sample_next_node(graph, node):
  d = graph[node]
  v_list = sorted(d.keys())
  num = len(v_list)
  if num > 0:
    random_value = numpy.random.choice(num)
    return v_list[random_value]
  else:
    return node


def generate_random_walks(graph,
                          walks_per_node,
                          walk_length,
                          random_seed=12345):
  random.seed(random_seed)
  for node in sorted(graph):
    for _ in range(walks_per_node):
      walk = [node]
      for _ in range(walk_length):
        walk.append(sample_next_node(graph, walk[-1]))
      yield walk


def remove_users_from_walks(walks):
  filtered_walks = []

  for walk in walks:
    new_walk = []
    for w in walk:
      if 'u' not in w:
        new_walk.append(w)
    filtered_walks.append(new_walk)

  return filtered_walks


class EpochLogger(CallbackAny2Vec):
  """Callback to log information about training."""

  def __init__(self):
    self.epoch = 0

  def on_epoch_begin(self, model):
    print('Epoch #{} start'.format(self.epoch))

  def on_epoch_end(self, model):
    print('Epoch #{} end'.format(self.epoch))
    self.epoch += 1


def RunDeepWalk(graph,
                num_walks_node,
                walk_length,
                embedding_dim,
                iterations,
                window=5,
                remove_users=False,
                random_seed=12345):

  sentences = sorted(
      generate_random_walks(
          graph,
          walks_per_node=num_walks_node,
          walk_length=walk_length,
          random_seed=random_seed))
  if remove_users:
    sentences = remove_users_from_walks(sentences)
  random.seed(random_seed)
  random.shuffle(sentences)
  model = None
  model = Word2Vec(
      sentences=sentences,
      min_count=0,
      sg=1,
      hs=1,
      negative=0,
      size=embedding_dim,
      seed=random_seed,
      sample=0,
      workers=12,
      window=window,
      iter=iterations)
  model.train(
      sentences,
      total_examples=model.corpus_count,
      epochs=model.epochs,
      callbacks=[EpochLogger()])
  return model, sentences


def DeepWalkMovielens(G,
                      num_walks_node=100,
                      walk_length=5,
                      embedding_dim=128,
                      iterations=10,
                      window=5,
                      remove_users=False,
                      make_undirected_within=False,
                      random_seed=12345):
  # make movielens undirected
  if make_undirected_within:
    G_undirected = make_movielens_undirected(G)
  else:
    G_undirected = G

  # create embeddings
  embedding, walks = RunDeepWalk(
      G_undirected,
      num_walks_node=num_walks_node,
      walk_length=walk_length,
      embedding_dim=embedding_dim,
      iterations=iterations,
      window=window,
      remove_users=remove_users,
      random_seed=random_seed)

  # extract symbol: vector embedding table
  embedding_map = {}
  for i, v in enumerate(embedding.wv.index2word):
    embedding_map[v] = embedding.wv.vectors[i]

  return embedding_map, walks


# Attack Utils
def top_k_add(graph,
              target_items=None,
              attackers=None,
              target_probability=1.0,
              attack_weight=5,
              attack_table={}):
  """assume graph is [users x items]."""

  # add ratings for attackers
  for a in attackers:
    for t in target_items:
      if random.uniform(0, 1.) < target_probability:
        if attack_weight is None:
          graph[a][t] = attack_table[t]
        else:
          graph[a][t] = attack_weight


def top_k_high_degree_attack(graph,
                             target_items=None,
                             attackers=None,
                             to_attack=None,
                             popular_video_count=100,
                             to_attack_probability=1.0,
                             target_probability=1.0,
                             random_seed=12345):
  """rate popular (trending) items, then add ours [users x items]."""

  numpy.random.seed(random_seed)

  new_graph = graph

  # who are the attackers?
  if not attackers:
    attackers = list(
        numpy.random.choice(
            graph.keys(), size=int(len(graph) * 0.01), replace=False))

  if not to_attack:
    popular_items = count_item_popularity(graph)
    # grab the most common nodes to attack
    to_attack = list(
        list(zip(*popular_items.most_common(popular_video_count)))[0])

  # attack popular stuff
  top_k_add(
      new_graph,
      target_items=to_attack,
      attackers=attackers,
      target_probability=to_attack_probability)

  # attack the target
  top_k_add(
      new_graph,
      target_items=target_items,
      attackers=attackers,
      target_probability=target_probability)
  return new_graph


# Monet Utils
def make_attack_covariate(user_item_graph, known_attackers, normalize=False):

  total_users = collections.defaultdict(int)
  bad_users = collections.defaultdict(int)

  for u in user_item_graph:
    for i in user_item_graph[u]:
      total_users[i] += 1
      if u in known_attackers:
        bad_users[i] += 1

  attack_covariate = {}

  squared_sum = 0.0
  for i in total_users:
    if i in bad_users:
      attack_covariate[i] = [float(bad_users[i])]
      squared_sum += float(bad_users[i])**2.0
    else:
      attack_covariate[i] = [0.0]

  if normalize:
    for i in attack_covariate:
      attack_covariate[i] = [attack_covariate[i][0] / numpy.sqrt(squared_sum)]
  return attack_covariate


# Extract weights from a keyed vector object
def extract_weights(keyed_vectors, tokens):
  return numpy.array([keyed_vectors[t] for t in tokens])


# Extract all weights in easily usable dict
def extract_all_weights(model_obj, tokens):
  """Extracts numpy-style weights from gensim-style MONET returns.

  Args:
    model_obj: output from the GloVe call (MONET or otherwise)
    tokens: a list of tokens, which are node labels for the original graph

  Returns:
    return_dict: a keyed dict of numpy matrices ordered by tokens:
      W: the sum of the input and output topology embeddings
      (If the GloVe model did not have metadata terms, these are None):
      Z: the sum of the input and output metadata embeddings
      H1: the input metadata transformation
      H2: the output metadata transormation
      E: if the GloVe model included covariates, this is [W, Z]. Otherwise, [W].
      W0: If the MONET unit was used, these are the un-SVD'd topology embeddings
  """
  return_dict = {}
  # Topology embeddings
  return_dict['W'] = (
      extract_weights(model_obj['topo_input'], tokens) +
      extract_weights(model_obj['topo_outpt'], tokens))
  # Metadata embeddings
  if model_obj['meta_input'] is not None:

    return_dict['Z'] = (
        extract_weights(model_obj['meta_input'], tokens) +
        extract_weights(model_obj['meta_outpt'], tokens))
    return_dict['H1'] = model_obj['meta_trans_input']
    return_dict['H2'] = model_obj['meta_trans_outpt']
    return_dict['E'] = numpy.concatenate([return_dict['W'], return_dict['Z']],
                                         axis=1)
  else:
    return_dict['Z'] = None
    return_dict['H1'] = None
    return_dict['H2'] = None
    return_dict['E'] = numpy.concatenate([return_dict['W']], axis=1)

  # Base topology embeddings
  if 'topo_input_raw' in return_dict:
    return_dict['W0'] = (
        extract_weights(model_obj['topo_input_raw'], tokens) +
        extract_weights(model_obj['topo_outpt_raw'], tokens))
  return return_dict


def monet_item_embed(user_item_graph,
                     item_metadata,
                     walks=None,
                     num_walks_node=100,
                     walk_length=WALK_LENGTH,
                     VECTOR_SIZE=128,
                     COVARIATE_SIZE=1,
                     WINDOW_SIZE=WINDOW_SIZE,
                     NUM_ITERATIONS=10,
                     BATCH_SIZE=100,
                     METHOD='MONET',
                     use_w2v=False,
                     DB_LEVEL=1.0,
                     random_seed=12345):

  if walks is None:
    # generate random walks
    G_undirected = make_movielens_undirected(user_item_graph)
    walks = list(
        generate_random_walks(
            G_undirected,
            walks_per_node=num_walks_node,
            walk_length=walk_length))
    walks = remove_users_from_walks(walks)
    random.shuffle(walks)
    # counter = collections.Counter([w for walk in walks for w in walk])
    flat_walks = []
    for walk in walks:
      for w in walk:
        flat_walks.append(w)
  counter = collections.Counter(flat_walks)
  tokens = sorted(counter.keys())
  # call monet
  with tf.Graph().as_default(), tf.Session() as session:
    with tf.device('/cpu:0'):
      weight_dict_monet = None
      if METHOD == 'MONET0':
        weight_dict_monet = GloVe(
            walks,
            session,
            metadata=item_metadata,
            vector_size=VECTOR_SIZE,
            covariate_size=COVARIATE_SIZE,
            use_monet=False,
            window_size=WINDOW_SIZE,
            iters=NUM_ITERATIONS,
            batch_size=BATCH_SIZE,
            random_seed=random_seed)
      elif METHOD == 'MONET':
        weight_dict_monet = GloVe(
            walks,
            session,
            metadata=item_metadata,
            vector_size=VECTOR_SIZE,
            covariate_size=COVARIATE_SIZE,
            use_monet=True,
            window_size=WINDOW_SIZE,
            iters=NUM_ITERATIONS,
            db_level=DB_LEVEL,
            use_w2v=use_w2v,
            batch_size=BATCH_SIZE,
            random_seed=random_seed)
      else:
        weight_dict_monet = GloVe(
            walks,
            session,
            vector_size=VECTOR_SIZE,
            use_monet=False,
            window_size=WINDOW_SIZE,
            iters=NUM_ITERATIONS,
            batch_size=BATCH_SIZE,
            random_seed=random_seed)
  monet_weights = extract_all_weights(weight_dict_monet, tokens)
  monet_embeddings = {
      x[0]: x[1] for x in zip(tokens, monet_weights['W'].tolist())
  }
  monet_covariates = None
  if METHOD == 'MONET':
    monet_covariates = {
        x[0]: x[1] for x in zip(tokens, monet_weights['Z'].tolist())}
  return monet_embeddings, monet_covariates, monet_weights, tokens


def nlp_baseline(embeddings, metadata, tokens):
  # prepare attack direction vector
  attack_vector = numpy.zeros(shape=embeddings['i_1'].shape)
  safe_vector = numpy.zeros(shape=embeddings['i_1'].shape)
  safe_count = 0
  for token in tokens:
    if metadata[token][0] > 0.0:
      attack_vector = attack_vector + embeddings[token] * metadata[token][0]
    else:
      safe_vector = safe_vector + embeddings[token]
      safe_count += 1
  attack_vector = (
      attack_vector / float(numpy.sum(list(metadata.values()))) -
      safe_vector / float(safe_count))
  # regress out of embedding matrix
  embed_mat = extract_weights(embeddings, tokens)
  projection_diff = numpy.matmul(
      (numpy.matmul(embed_mat, numpy.transpose(attack_vector)) /
       numpy.dot(attack_vector, attack_vector))[:, numpy.newaxis],
      attack_vector[numpy.newaxis, :])
  return embed_mat - projection_diff

# Scoring Utils


def score_embedding(embedding, target_ids, attacker_ids):
  """Computes the distances in unit ball space of a set of one embeddings from another."""
  embeddings_targets = [
      embedding[x] / numpy.linalg.norm(embedding[x]) for x in target_ids
  ]
  embeddings_attackers = [
      embedding[x] / numpy.linalg.norm(embedding[x]) for x in attacker_ids
  ]
  distances = numpy.inner(embeddings_targets, embeddings_attackers)
  return distances


# score_ranking_attack
def score_ranking_attack(embedding,
                         to_shill_ids,
                         target_ids,
                         number_neighbors=20):
  normalized_embeddings = {}

  for x in embedding:
    norm_scale = numpy.linalg.norm(embedding[x])
    normalized_embeddings[x] = embedding[x] / (
        norm_scale if norm_scale > 1e-10 else 1.0)

  ordered_ids = list(sorted(embedding.keys()))
  ordered_embeddings = []

  ordered_id_targets = {}

  for idx, x in enumerate(ordered_ids):
    # remove non-item embeddings
    if 'i' in x:
      if x in to_shill_ids:
        ordered_id_targets[idx] = x
      ordered_embeddings.append(normalized_embeddings[x])

  X = numpy.vstack(ordered_embeddings)
  nbrs = NearestNeighbors(
      n_neighbors=number_neighbors + 1, algorithm='brute',
      metric='cosine').fit(X)
  X_find = numpy.vstack([normalized_embeddings[x] for x in target_ids])
  distances, indices = nbrs.kneighbors(X_find)

  total_found_in_topk = 0

  for row in indices:
    for item in row[1:]:
      if item in ordered_id_targets:
        total_found_in_topk += 1

  return distances, indices, total_found_in_topk


def row_normalize(mat):
  row_sqss = numpy.sqrt(numpy.sum(mat**2.0, axis=1))
  return mat / row_sqss[:, None]


def embedding_similarity(embeddings, scale_embeddings=False):
  if scale_embeddings:
    embeddings = row_normalize(embeddings)
  return numpy.matmul(embeddings, numpy.transpose(embeddings))


def compute_distance_correlation(embedding_dict1,
                                 embedding_dict2,
                                 tokens,
                                 unattacked_indx,
                                 scale_embeddings=True):
  distances1 = embedding_similarity(
      extract_weights(embedding_dict1, tokens)[unattacked_indx, :],
      scale_embeddings=scale_embeddings)
  distances2 = embedding_similarity(
      extract_weights(embedding_dict2, tokens)[unattacked_indx, :],
      scale_embeddings=scale_embeddings)
  return pearsonr(distances1.flatten(), distances2.flatten())[0]


# Eval Utils


def save_embeddings(weights, save_dir, tokens, name):
  with open(os.path.join(save_dir, '%s_embeddings.txt' % name), 'w') as f:
    numpy.savetxt(f, extract_weights(weights, tokens))


def load_embeddings(save_dir, name, tokens):
  with open(os.path.join(save_dir, '%s_embeddings.txt' % name)) as f:
    weights = numpy.loadtxt(f)
  return {t: weights for (t, weights) in zip(tokens, weights)}


def nearest_neighbors_by_score(neighbors, scores, n=10):
  index_order = numpy.argsort(scores)[-n:]
  return [neighbors[i] for i in numpy.flip(index_order)]


def get_sorted_distances(embedding_dict,
                         tokens,
                         normalize_rows=False,
                         normalize_cols=False):
  embeddings = extract_weights(embedding_dict, tokens)
  if normalize_rows:
    E = row_normalize(embeddings)
  if normalize_cols:
    E = scale(embeddings, with_mean=False)
  else:
    E = embeddings
  dists = numpy.matmul(E, numpy.transpose(E))
  sorted_distances = {}
  for i, t in enumerate(tokens):
    dist_v = dists[i]
    sorted_distances[t] = [
        (tokens[j], dist_v[j]) for j in numpy.flip(numpy.argsort(dist_v))
    ]
  return sorted_distances


def avg_mrrs(embeddings,
             cooccurrences,
             tokens,
             ignore_answers=[],
             nns=[1, 5, 10, 20],
             normalize_rows=False,
             normalize_cols=False):
  max_nn = numpy.max(nns)
  mrrs = {nn: {} for nn in nns}
  sorted_distances = get_sorted_distances(embeddings, tokens, normalize_rows,
                                          normalize_cols)
  ignore_answers_set = set(ignore_answers)
  for item_label in tokens:
    neighbors = []
    scores = []
    item_cdict = cooccurrences[item_label]
    for n, s in sorted(item_cdict.items()):
      if n != item_label and n not in ignore_answers_set:
        neighbors.append(n)
        scores.append(s)
    nearest_neighbors = nearest_neighbors_by_score(neighbors, scores, max_nn)
    nn_sets = {nn: set(nearest_neighbors[:nn]) for nn in nns}
    mrr_scores = {nn: [] for nn in nns}
    i = 0
    while nn_sets[max_nn]:  # this used to explicitly check length
      ns_pair = sorted_distances[item_label][i]
      for nn in nns:
        if ns_pair[0] in nn_sets[nn]:
          mrr_scores[nn].append(1.0 / (i + 1))
          nn_sets[nn].remove(ns_pair[0])
      i += 1
    for nn in nns:
      mrrs[nn][item_label] = numpy.mean(mrr_scores[nn])
  return mrrs


def compute_mrr_curve(embeddings,
                      cooccurrences,
                      tokens,
                      ignore_answers=[],
                      nns=list(range(1, 21)),
                      target_set=None,
                      normalize_rows=False,
                      normalize_cols=False):
  mrrs = avg_mrrs(embeddings, cooccurrences, tokens, ignore_answers, nns,
                  normalize_rows, normalize_cols)
  if target_set is None:
    return [
        numpy.mean(list(mrr_dict.values())) for _, mrr_dict in mrrs.items()
    ]
  else:
    return [
        numpy.mean([v
                    for _, v in mrr_dict.items()])
        for _, mrr_dict in mrrs.items()
    ]


# Experiment Loop

# load movielens
G_prime, user_set, item_set = load_movielens_graph(DATA_FILE)

# Methods vec
methods = ['deepwalk', 'glove', 'monet0', 'monet', 'random', 'nlp']

# Helper function to get method name from debias (DB) level
monet_alpha_encoder = lambda x: 'monet%0.2f' % x

# Set up debias levels
DB_LEVELS = [v / 100.0 for v in list(range(75, 100, 5)) + [50, 25]]
methods.extend([monet_alpha_encoder(db_level) for db_level in DB_LEVELS])

G_prime = make_movielens_undirected(G_prime)
G_prime = filter_by_weight(G_prime, min_weight=4)
user_set = set([u for u in G_prime if u[0] == 'u'])
item_set = set([i for i in G_prime if i[0] == 'i'])
results = []

for exp_no in range(NUMBER_OF_EXPERIMENTS):
  print('Performing experiment: ' + str(exp_no))

  time_dict = {}

  exp_dir = os.path.join(SAVE_DIR, 'experiment%d' % exp_no)
  EXP_SEED = SEED + exp_no
  if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)

  if OVERWRITE_EMBEDDINGS:
    # select attackers
    numpy.random.seed(EXP_SEED)
    attackers = list(
        numpy.random.choice(
            sorted(user_set),
            size=int(len(G_prime) * FRACTION_ATTACKERS),
            replace=False))
    to_shills = list(
        numpy.random.choice(
            sorted(item_set), size=NUMBER_TO_SHILL, replace=False))
    known_attackers = list(
        numpy.random.choice(
            attackers,
            size=int(len(attackers) * FRACTION_ATTACKERS_KNOWN),
            replace=False))

    # uniform random attack
    targets = list(
        numpy.random.choice(
            sorted(item_set), size=NUMBER_TARGETS, replace=False))
    G_attacked = top_k_high_degree_attack(
        copy.deepcopy(G_prime),
        target_items=to_shills,
        attackers=attackers,
        to_attack=targets,
        random_seed=EXP_SEED)
    G_attacked = make_movielens_undirected(G_attacked)
    item_metadata = make_attack_covariate(G_attacked, set(known_attackers))
    item_metadata_normalized = make_attack_covariate(G_attacked,
                                                     set(known_attackers), True)

    # Save the metadata
    with open(os.path.join(exp_dir, 'item_metadata.txt'), 'w') as f:
      f.write(json.dumps(item_metadata))

    embeddings_orig, walks_orig = DeepWalkMovielens(
        G_prime,
        remove_users=True,
        walk_length=WALK_LENGTH,
        random_seed=EXP_SEED)
    stime = time.time()
    embeddings_prime, walks_prime = DeepWalkMovielens(
        G_attacked,
        remove_users=True,
        walk_length=WALK_LENGTH,
        random_seed=EXP_SEED)
    time_dict['deepwalk_time'] = time.time() - stime

    # Save the walks
    with open(os.path.join(exp_dir, 'walks_orig.txt'), 'w') as f:
      f.write(json.dumps(walks_orig))
    with open(os.path.join(exp_dir, 'walks_prime.txt'), 'w') as f:
      f.write(json.dumps(walks_prime))

    # Glove
    stime = time.time()
    glove_weights = monet_item_embed(
        G_attacked,
        item_metadata,
        NUM_ITERATIONS=NUMBER_GLOVE_ITERATIONS,
        METHOD='GloVe',
        random_seed=EXP_SEED)
    time_dict['glove_time'] = time.time() - stime
    glove_topology, glove_covariates, all_glove_weights, tokens = glove_weights
    print('done with glove')

    # Save the tokens
    with open(os.path.join(exp_dir, 'tokens.txt'), 'w') as f:
      f.write(json.dumps(tokens))

    # After getting tokens, able to save both deepwalk and glove weights
    save_embeddings(embeddings_prime, exp_dir, tokens, 'deepwalk')
    save_embeddings(glove_topology, exp_dir, tokens, 'glove')

    # MONET0
    stime = time.time()
    monet0_weights = monet_item_embed(
        G_attacked,
        item_metadata_normalized,
        NUM_ITERATIONS=NUMBER_GLOVE_ITERATIONS,
        METHOD='MONET0',
        random_seed=EXP_SEED)
    time_dict['monet0_time'] = time.time() - stime
    monet0_topology, monet0_covariates, all_monet0_weights, tokens = monet0_weights
    save_embeddings(monet0_topology, exp_dir, tokens, 'monet0')

    # MONET
    stime = time.time()
    monet_weights = monet_item_embed(
        G_attacked,
        item_metadata_normalized,
        NUM_ITERATIONS=NUMBER_GLOVE_ITERATIONS,
        METHOD='MONET',
        random_seed=EXP_SEED)
    time_dict['monet_time'] = time.time() - stime
    monet_topology, monet_covariates, all_monet_weights, tokens = monet_weights
    save_embeddings(monet_topology, exp_dir, tokens, 'monet')

    # MONET with different regs
    for db_level in DB_LEVELS:
      monet_weights = monet_item_embed(
          G_attacked,
          item_metadata_normalized,
          NUM_ITERATIONS=NUMBER_GLOVE_ITERATIONS,
          METHOD='MONET',
          DB_LEVEL=db_level,
          random_seed=EXP_SEED)
      monet_topology, monet_covariates, all_monet_weights, tokens = monet_weights
      save_embeddings(monet_topology, exp_dir, tokens,
                      monet_alpha_encoder(db_level))

    # Save the configuration of the experiment
    exp_config = {
        'targets': targets,
        'items_to_shill': to_shills,
        'attackers': attackers,
        'known_attackers': known_attackers,
        'NEIGHBORS_TO_EVAL': NEIGHBORS_TO_EVAL,
        'FRACTION_ATTACKERS': FRACTION_ATTACKERS,
        'FRACTION_ATTACKERS_KNOWN': FRACTION_ATTACKERS_KNOWN,
        'NUMBER_TO_SHILL': NUMBER_TO_SHILL,
        'NUMBER_TARGETS': NUMBER_TARGETS,
        'NUMBER_TO_ATTACK': NUMBER_TO_ATTACK,
        'SEED': SEED,
        'NUMBER_GLOVE_ITERATIONS': NUMBER_GLOVE_ITERATIONS
    }
    with open(os.path.join(exp_dir, 'exp_config.txt'), 'w') as f:
      f.write(json.dumps(exp_config))

    with open(os.path.join(exp_dir, 'timing_results.txt'), 'w') as f:
      f.write(json.dumps(time_dict))

  if DO_EVAL:
    # Load the tokens and exp_config
    with open(os.path.join(exp_dir, 'tokens.txt')) as f:
      tokens = json.loads(f.read())
    with open(os.path.join(exp_dir, 'exp_config.txt')) as f:
      exp_config = json.loads(f.read())

    # Load the walks
    with open(os.path.join(exp_dir, 'walks_orig.txt')) as f:
      walks_orig = json.loads(f.read())
    with open(os.path.join(exp_dir, 'walks_prime.txt')) as f:
      walks_prime = json.loads(f.read())

    attacked_vids = exp_config['targets'] + exp_config['items_to_shill']

    # Load the embeddings
    topology_weights = {
        method: load_embeddings(exp_dir, method, tokens)
        for method in methods
        if method not in ['decor_max', 'decor_sum', 'random', 'nlp']
    }

    # Load timing results
    with open(os.path.join(exp_dir, 'timing_results.txt')) as f:
      time_dict = json.loads(f.read())

    # Load the item metadata
    with open(os.path.join(exp_dir, 'item_metadata.txt')) as f:
      item_metadata = json.loads(f.read())

    # Compute standard fairness decorrelated matrices
    stime = time.time()
    topology_weights['nlp'] = dict(
        zip(tokens,
            nlp_baseline(topology_weights['glove'], item_metadata, tokens)))
    time_dict['nlp_time'] = time.time() - stime

    # Compute metrics
    bad_items_count = {}
    for method in topology_weights:
      bad_items_count.update({
          method: score_ranking_attack(
              topology_weights[method],
              exp_config['items_to_shill'],
              exp_config['targets'],
              number_neighbors=exp_config['NEIGHBORS_TO_EVAL'])[2]})

    exp_result = {'experiment': exp_no, 'SEED': SEED}
    exp_result.update(time_dict)
    exp_result.update(bad_items_count)

    # Compute random embeddings
    numpy.random.seed(EXP_SEED)
    topology_weights['random'] = {
        t: numpy.random.normal(size=(128,)) for t in topology_weights['glove']
    }

    # Compute distance set correlations
    attacked_set = set(attacked_vids)
    unattacked_indx = [i for i, t in enumerate(tokens) if t not in attacked_set]
    for method in topology_weights:
      exp_result.update({
          ('%s_vs_deepwalk_distcorr' % method):
              compute_distance_correlation(topology_weights[method],
                                           topology_weights['deepwalk'], tokens,
                                           unattacked_indx)
      })
      exp_result.update({
          ('%s_vs_glove_distcorr' % method):
              compute_distance_correlation(topology_weights[method],
                                           topology_weights['glove'], tokens,
                                           unattacked_indx)
      })
    results.append(exp_result)

    # Compute cooccurrences
    (cooccurrence_list, index_vocab_list, vocab_index_lookup,
     tokenized_cooccurrences) = count_cooccurrences(walks_prime, 5)

    # Compute mrr curves
    for method in topology_weights:
      normalize_rows = False
      normalize_cols = False
      exp_result.update({
          ('%s_mrr_curve_full' % method): [
              compute_mrr_curve(
                  topology_weights[method],
                  tokenized_cooccurrences,
                  tokens,
                  normalize_rows=normalize_rows,
                  normalize_cols=normalize_cols)
          ]
      })
      exp_result.update({
          ('%s_mrr_curve_noattacked' % method): [
              compute_mrr_curve(
                  topology_weights[method],
                  tokenized_cooccurrences,
                  tokens,
                  ignore_answers=attacked_vids,
                  normalize_rows=normalize_rows,
                  normalize_cols=normalize_cols)
          ]
      })
      exp_result.update({
          ('%s_mrr_curve_noattacked_justtargets' % method): [
              compute_mrr_curve(
                  topology_weights[method],
                  tokenized_cooccurrences,
                  tokens,
                  ignore_answers=attacked_vids,
                  target_set=exp_config['targets'],
                  normalize_rows=normalize_rows,
                  normalize_cols=normalize_cols)
          ]
      })
    with open(os.path.join(exp_dir, str(exp_no) + '.txt'), 'w') as f:
      f.write(json.dumps(exp_result))
