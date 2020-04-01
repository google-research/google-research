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

"""Runs political blogs experiment in Section 4 of associated manuscript."""

# Imports
from __future__ import print_function

import collections
import copy
import json
import operator
import os
import random
import time
from call_glove import GloVe
from eval_utils import load_numpy_matrix
from eval_utils import save_numpy_matrix
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import matplotlib.pyplot as plt
import numpy
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import tensorflow.compat.v1 as tf

# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=dangerous-default-value
# pylint: disable=invalid-name

# User-defined hyperparameters for the experiment.
DATA_DIR = 'polblogs'
SAVE_DIR = 'experiment_data/polblogs'
NUM_RUNS = 10
WALKS_PER_NODE = 80
WALK_LENGTH = 40
WINDOW_SIZE = 10
VECTOR_SIZE = 16
COVARIATE_SIZE = 2
NUM_ITERATIONS = 5
BATCH_SIZE = 100
RANDOM_SEED = 12345
OVERWRITE_EMBEDDINGS = True
DO_SCORES = True

if not os.path.isdir(SAVE_DIR):
  os.makedirs(SAVE_DIR)

PLOT_DIR = os.path.join(SAVE_DIR, 'plots')
if not os.path.isdir(PLOT_DIR):
  os.mkdir(PLOT_DIR)

numpy.random.seed(RANDOM_SEED)


# Load graph
def load_polblogs_graph(filepath):
  array = numpy.loadtxt(open(filepath), dtype=numpy.int64)
  graph = collections.defaultdict(dict)
  node_set = set()
  for x in range(array.shape[0]):
    source = str(array[x][0])
    target = str(array[x][1])
    graph[source][target] = 1
    graph[target][source] = 1
    node_set.add(source)
    node_set.add(target)

  return graph, node_set


# Deepwalk next-node random walk sampler
def sample_next_node(graph, n):
  d = graph[n]
  v_list = list(d.keys())
  num = len(v_list)
  if num > 0:
    random_value = numpy.random.choice(num)
    return v_list[random_value]
  else:
    return n


# Deepwalk random walk sampler
def generage_random_walks(graph, walks_per_node, walk_length):
  for n in graph.keys():
    for _ in range(walks_per_node):
      walk = [n]
      for _ in range(walk_length):
        walk.append(sample_next_node(graph, walk[-1]))
      yield walk


def get_keyed_vector(l):
  numbers = l.strip().split()
  return {str(int(numbers[0])): [float(x) for x in numbers[1:]]}


def load_embeddings(ff):
  _ = ff.readline()
  model = {}
  for l in ff:
    model.update(get_keyed_vector(l))
  return model


# Extract weights from a keyed vector object
def extract_weights(keyed_vectors, tokens):
  return numpy.array([keyed_vectors[t] for t in tokens])


# Extract all weights in easily usable dict
def extract_all_weights(model_obj, tokens):
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
    return_dict['E'] = return_dict['W']

  # Base topology embeddings
  if 'topo_input_raw' in return_dict:
    return_dict['W0'] = (
        extract_weights(model_obj['topo_input_raw'], tokens) +
        extract_weights(model_obj['topo_outpt_raw'], tokens))
  return return_dict


# Plot TSNEs with label colors
COLORS = ['red', 'blue', 'orange', 'green']


def plot_2d_embeddings(embeddings,
                       label_matrix,
                       title='Title Here',
                       top=10,
                       reverse=True,
                       plot_size=12,
                       pntsize=6,
                       savefile=None,
                       do_legend=False,
                       show_axes=True,
                       wrap_points=False,
                       titlesize=4,
                       subtitle='',
                       subtitlesize=4,
                       ticksize=16):
  # Filter samples with no labels
  retained_samples = numpy.argwhere(numpy.sum(label_matrix, axis=1))[:, 0]
  x1 = embeddings[retained_samples, 0]
  x2 = embeddings[retained_samples, 1]
  label_matrix = label_matrix[retained_samples, :]
  labels = [p[1] for p in list(numpy.argwhere(label_matrix))]

  # Filter the label set if necessary
  if len(set(labels)) > top:
    item_counts = dict([(label, labels.count(label)) for label in set(labels)])
    sorted_counts = sorted(
        item_counts.items(), key=operator.itemgetter(1), reverse=reverse)
    good_labels = set()
    for entry in sorted_counts[:top]:
      good_labels.add(entry[0])

    x1 = numpy.array(
        [x1[i] for i in range(len(labels)) if labels[i] in good_labels])
    x2 = numpy.array(
        [x2[i] for i in range(len(labels)) if labels[i] in good_labels])
    good_example_labels = [label for label in labels if label in good_labels]
    labels = good_example_labels

  # Split the data into groups
  label_set = set(labels)
  data_groups = [None] * len(label_set)
  for i, label in enumerate(label_set):
    indx = [j for j in range(len(labels)) if labels[j] == label]
    data_groups[i] = (x1[indx], x2[indx])

  # Make the plot
  fig = plt.figure(figsize=(plot_size, plot_size))
  if wrap_points:
    plt.xlim(numpy.min(x1), numpy.max(x1))
    plt.ylim(numpy.min(x2), numpy.max(x2))
  ax = fig.add_subplot(1, 1, 1)
  for i, data_group in enumerate(data_groups):
    x, y = data_group
    ax.scatter(x, y, s=pntsize, c=COLORS[i], edgecolors='none', label=i)
  if not subtitle:
    plt.title(title, fontsize=titlesize)
  else:
    plt.suptitle(title, fontsize=titlesize)
    plt.title(subtitle, fontsize=subtitlesize)

  if do_legend:
    plt.legend(loc=1)

  # Modify axes
  frame1 = plt.gca()
  frame1.axes.get_xaxis().set_visible(show_axes)
  frame1.axes.get_yaxis().set_visible(show_axes)
  plt.rc('xtick', labelsize=ticksize)
  plt.rc('ytick', labelsize=ticksize)

  # Save or plot
  if savefile:
    print('saving not plotting')
    with open(savefile, 'w') as f:
      plt.savefig(f)
  else:
    print('plotting not saving')
    plt.show()


def show_results(macro_scores, micro_scores, training_ratios):
  for r in training_ratios:
    print('%0.2f: mic %0.5f, mac %0.5f' % (r, micro_scores[r], macro_scores[r]))


def get_f1_score(L, W, average):
  return f1_score(L, W, average=average) if average else f1_score(L, W)[0]


def score_results(weights,
                  labels,
                  num_fits=30,
                  training_ratios=numpy.arange(0.01, 0.10, 0.01),
                  max_iter=1000,
                  scale_columns=True):
  n = weights.shape[0]
  if scale_columns:
    weights = scale(weights, with_mean=False, axis=0)
  macro_scores = dict(zip(training_ratios, [0.0] * len(training_ratios)))
  micro_scores = dict(zip(training_ratios, [0.0] * len(training_ratios)))
  for r in training_ratios:
    macros = 0.0
    micros = 0.0
    for _ in range(num_fits):
      training_sample = numpy.random.choice(list(range(n)), int(n * r))
      multi_linsvm = OneVsRestClassifier(LinearSVC(max_iter=max_iter))
      multi_linsvm.fit(weights[training_sample], labels[training_sample])
      macros += f1_score(
          labels, multi_linsvm.predict(weights), average='macro') / num_fits
      micros += f1_score(
          labels, multi_linsvm.predict(weights), average='micro') / num_fits
    macro_scores[r] = macros
    micro_scores[r] = micros
  return macro_scores, micro_scores


def compute_leakage(m1, m2):
  return numpy.linalg.norm(
      numpy.matmul(
          numpy.transpose(scale(m1, with_mean=False)),
          scale(m2, with_mean=False)))


# Utils for getting embedding distance correlation
def row_normalize(mat):
  row_sqss = numpy.sqrt(numpy.sum(mat**2.0, axis=1))
  return mat / row_sqss[:, None]


def embedding_similarity(embeddings, scale_embeddings=False):
  if scale_embeddings:
    embeddings = row_normalize(embeddings)
  return numpy.matmul(embeddings, numpy.transpose(embeddings))


def compute_distance_correlation(embeddings1,
                                 embeddings2,
                                 scale_embeddings=True):
  distances1 = embedding_similarity(
      embeddings1, scale_embeddings=scale_embeddings)
  distances2 = embedding_similarity(
      embeddings2, scale_embeddings=scale_embeddings)
  return pearsonr(distances1.flatten(), distances2.flatten())[0]

# Load the graph
G, _ = load_polblogs_graph(os.path.join(DATA_DIR, 'graph.txt'))

# Load the blog attributes
with open(os.path.join(DATA_DIR, 'party_cvrt.txt')) as f:
  party_cvrt_data = load_embeddings(f)

# Load the memberships and get tokens
memships = {}
with open(os.path.join(DATA_DIR, 'memberships.txt')) as f:
  for line in f:
    line_split = line.strip().split()
    memships.update({line_split[0]: int(line_split[1])})
tokens = sorted(memships.keys())

# Construct party labels
party_labels = numpy.zeros(shape=(len(memships), 2))
for i, node in enumerate(tokens):
  party_labels[i, memships[node]] = 1.0

# Get random walks
walks = list(
    generage_random_walks(
        G, walks_per_node=WALKS_PER_NODE, walk_length=WALK_LENGTH))
random.shuffle(walks)
walks_fn = os.path.join(SAVE_DIR, 'walks')
with open(walks_fn, 'w') as f:
  f.write(json.dumps(walks))


class EpochLogger(CallbackAny2Vec):

  def __init__(self):
    self.epoch = 0

  def on_epoch_begin(self, model):
    print('Epoch #{} start'.format(self.epoch))

  def on_epoch_end(self, model):
    print('Epoch #{} end'.format(self.epoch))
    self.epoch += 1


def RunDeepWalk(sentences, embedding_dim, iterations, window=5):
  model = None
  model = Word2Vec(
      sentences=sentences,
      min_count=0,
      sg=1,
      hs=1,
      negative=0,
      size=embedding_dim,
      seed=0,
      sample=0,
      workers=12,
      window=window,
      iter=iterations)
  model.train(
      sentences,
      total_examples=model.corpus_count,
      epochs=model.epochs,
      callbacks=[EpochLogger()])
  return model


def DeepWalkPolblogs(sentences, embedding_dim=128, iterations=10, window=5):
  # create embeddings
  embedding = RunDeepWalk(
      sentences,
      embedding_dim=embedding_dim,
      iterations=iterations,
      window=window)
  print(embedding.wv.vectors.shape)
  embedding_map = {}
  for i, v in enumerate(embedding.wv.index2word):
    embedding_map[v] = embedding.wv.vectors[i]

  return embedding_map


# More scoring utils
def scores(X,
           y,
           random_state=12345,
           scoring='accuracy',
           training_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
           C_log_lims=[-6, 6],
           gamma_log_lims=[-6, 6],
           cv=5,
           n_jobs=64):
  # Set up training & scoring
  n = X.shape[0]
  svm_scores = {
      'linear': [0.0] * len(training_ratios),
      'rbf': [0.0] * len(training_ratios)
  }
  gamma_range = numpy.logspace(-6, 1, 6)
  C_range = numpy.logspace(C_log_lims[0], 1, C_log_lims[1])
  gamma_range = numpy.logspace(gamma_log_lims[0], 1, gamma_log_lims[1])
  lin_pipe = Pipeline([('scale', StandardScaler()), ('clf', LinearSVC())])
  lin_param_grid = dict(clf__C=C_range)
  rbf_pipe = Pipeline([('scale', StandardScaler()), ('clf', SVC())])
  rbf_param_grid = dict(clf__C=C_range, clf__gamma=gamma_range)
  for j, r in enumerate(training_ratios):
    print('--training ratio %0.3f' % r)
    # Choose training set
    numpy.random.seed(random_state + j)
    train_set = numpy.random.randint(low=0, high=n, size=int(r * n))
    train_set_set = set(train_set)
    X_train = X[train_set]
    y_train = y[train_set]
    X_test = numpy.array([v for i, v in enumerate(X) if i not in train_set_set])
    y_test = numpy.array([v for i, v in enumerate(y) if i not in train_set_set])
    print('----lin')
    # Fit and score Linear SVM
    lin_grid = GridSearchCV(
        lin_pipe,
        param_grid=lin_param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        scoring=scoring)
    lin_grid.fit(X_train, y_train)
    svm_scores['linear'][j] = lin_grid.score(X_test, y_test)
    print('----rbf')
    # Fit and score RBF SVM
    rbf_grid = GridSearchCV(
        rbf_pipe,
        param_grid=rbf_param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        scoring=scoring)
    rbf_grid.fit(X_train, y_train)
    svm_scores['rbf'][j] = rbf_grid.score(X_test, y_test)
  return svm_scores


def save_weights_object(savedir, weights):
  if not os.path.isdir(savedir):
    os.makedirs(savedir)
  for m_type in weights:
    if weights[m_type] is not None:
      save_numpy_matrix(os.path.join(savedir, m_type), weights[m_type])


def load_weights_object(savedir, M_types=['E', 'W', 'Z', 'H1', 'H2']):
  weights = {}
  for M_type in M_types:
    weights[M_type] = load_numpy_matrix(os.path.join(savedir, M_type))
  return weights


# Prepare adversarial labels
adv_labels = {
    k: [float(x > 0.0) for x in v] for k, v in party_cvrt_data.items()
}

# Set up training & scoring
methods = ['adv1', 'deepwalk', 'glove', 'monet', 'monet0', 'random']
embeddings = {name: None for name in methods}
score_dict = {name: None for name in methods}
score_dicts = []
leakage_dict = copy.deepcopy(score_dict)
time_dict = {k: [] for k in score_dict}
distance_correlation_dict = copy.deepcopy(time_dict)

print('NUM_RUNS is %d' % NUM_RUNS)

for i in range(NUM_RUNS):
  print('------\n\n\n\n')
  print('i = %d' % i)
  print('------\n\n\n\n')
  rep_save_path = os.path.join(SAVE_DIR, str(i))
  embeddings_were_run = False
  # Run DeepWalk
  method_save_path = os.path.join(rep_save_path, 'deepwalk')
  if not os.path.isfile(method_save_path) or OVERWRITE_EMBEDDINGS:
    embeddings_were_run = True
    t0 = time.time()
    weight_dict_deepwalk = DeepWalkPolblogs(
        walks,
        embedding_dim=VECTOR_SIZE,
        iterations=NUM_ITERATIONS,
        window=WINDOW_SIZE)
    time_dict['deepwalk'].append(time.time() - t0)
    embeddings['deepwalk'] = {
        'W': extract_weights(weight_dict_deepwalk, tokens)
    }
    save_weights_object(method_save_path, embeddings['deepwalk'])
  else:
    embeddings['deepwalk'] = load_weights_object(method_save_path)

  # #@title Run standard GloVe
  method_save_path = os.path.join(rep_save_path, 'glove')
  if not os.path.isfile(method_save_path) or OVERWRITE_EMBEDDINGS:
    embeddings_were_run = True
    t0 = time.time()
    with tf.Graph().as_default(), tf.Session() as session:
      with tf.device('/cpu:0'):
        weight_dict_glove = GloVe(
            walks,
            session,
            vector_size=VECTOR_SIZE,
            window_size=WINDOW_SIZE,
            iters=NUM_ITERATIONS,
            batch_size=BATCH_SIZE,
            random_seed=RANDOM_SEED + i)
    time_dict['glove'].append(time.time() - t0)
    embeddings['glove'] = extract_all_weights(weight_dict_glove, tokens)
    save_weights_object(method_save_path, embeddings['glove'])
  else:
    embeddings['glove'] = load_weights_object(method_save_path)

  # #@title Run GloVe with naive MONET (no SVD residualization)
  method_save_path = os.path.join(rep_save_path, 'monet0')
  if not os.path.isfile(method_save_path) or OVERWRITE_EMBEDDINGS:
    embeddings_were_run = True
    t0 = time.time()
    with tf.Graph().as_default(), tf.Session() as session:
      with tf.device('/cpu:0'):
        weight_dict_glove = GloVe(
            walks,
            session,
            vector_size=VECTOR_SIZE,
            metadata=party_cvrt_data,
            covariate_size=COVARIATE_SIZE,
            window_size=WINDOW_SIZE,
            iters=NUM_ITERATIONS,
            batch_size=BATCH_SIZE,
            random_seed=RANDOM_SEED + i)
    time_dict['monet0'].append(time.time() - t0)
    embeddings['monet0'] = extract_all_weights(weight_dict_glove, tokens)
    save_weights_object(method_save_path, embeddings['monet0'])
  else:
    embeddings['monet0'] = load_weights_object(method_save_path)

  # Run MONET
  method_save_path = os.path.join(rep_save_path, 'monet')
  if not os.path.isfile(method_save_path) or OVERWRITE_EMBEDDINGS:
    embeddings_were_run = True
    t0 = time.time()
    with tf.Graph().as_default(), tf.Session() as session:
      with tf.device('/cpu:0'):
        weight_dict_monet = GloVe(
            walks,
            session,
            metadata=party_cvrt_data,
            vector_size=VECTOR_SIZE,
            covariate_size=COVARIATE_SIZE,
            use_monet=True,
            window_size=WINDOW_SIZE,
            iters=NUM_ITERATIONS,
            batch_size=BATCH_SIZE,
            random_seed=RANDOM_SEED + i)
    time_dict['monet'].append(time.time() - t0)
    embeddings['monet'] = extract_all_weights(weight_dict_monet, tokens)
    save_weights_object(method_save_path, embeddings['monet'])
  else:
    embeddings['monet'] = load_weights_object(method_save_path)

  # Run Adversary with lr = 0.001
  method_save_path = os.path.join(rep_save_path, 'adv1')
  if not os.path.isfile(method_save_path) or OVERWRITE_EMBEDDINGS:
    embeddings_were_run = True
    t0 = time.time()
    with tf.Graph().as_default(), tf.Session() as session:
      with tf.device('/cpu:0'):
        glove_adv_1 = GloVe(
            walks,
            session,
            vector_size=VECTOR_SIZE,
            window_size=WINDOW_SIZE,
            iters=NUM_ITERATIONS,
            batch_size=BATCH_SIZE,
            random_seed=12345 + i,
            adv_lam=10.0,
            adv_dim=8,
            adv_lr=0.001,
            adv_labels=adv_labels)
    time_dict['adv1'].append(time.time() - t0)
    embeddings['adv1'] = extract_all_weights(glove_adv_1, tokens)
    save_weights_object(method_save_path, embeddings['adv1'])
  else:
    embeddings['adv1'] = load_weights_object(method_save_path)

  # Get random embeddings
  method_save_path = os.path.join(rep_save_path, 'random')
  if not os.path.isfile(method_save_path) or OVERWRITE_EMBEDDINGS:
    embeddings_were_run = True
    numpy.random.seed(RANDOM_SEED + i)
    embeddings['random'] = {
        'W': numpy.random.normal(size=embeddings['glove']['W'].shape)
    }
    save_weights_object(method_save_path, embeddings['random'])
  else:
    embeddings['random'] = load_weights_object(method_save_path)

  # Save/load timing results
  if embeddings_were_run:
    with open(os.path.join(rep_save_path, 'timing'), 'w') as f:
      f.write(json.dumps(time_dict))
  else:
    with open(os.path.join(rep_save_path, 'timing')) as f:
      time_dict = json.loads(f.read().strip())

  # Get metadata importances
  monet0_importances = numpy.matmul(embeddings['monet0']['H1'],
                                    numpy.transpose(embeddings['monet0']['H2']))
  monet_importances = numpy.matmul(embeddings['monet']['H1'],
                                   numpy.transpose(embeddings['monet']['H2']))
  print('saving monet0_importances to %s' %
        (os.path.join(rep_save_path, 'monet0_importances')))
  print('monet0 importances:')
  print(monet0_importances)
  save_numpy_matrix(
      os.path.join(rep_save_path, 'monet0_importances'), monet0_importances)
  print('saving monet_importances to %s' %
        (os.path.join(rep_save_path, 'monet_importances')))
  print('monet importances:')
  print(monet_importances)
  save_numpy_matrix(
      os.path.join(rep_save_path, 'monet_importances'), monet_importances)

  # Get leakages
  print('computing leakages')
  for method in embeddings:
    if embeddings[method] is not None:
      if 'monet' in method:
        leakage_dict[method] = float(
            compute_leakage(embeddings[method]['Z'], embeddings[method]['W']))
      else:
        leakage_dict[method] = float(
            compute_leakage(party_labels, embeddings[method]['W']))
  with open(os.path.join(rep_save_path, 'leakage_dict'), 'w') as f:
    f.write(json.dumps(leakage_dict))

  # Get multi-train-ratio eval
  score_dict_path = os.path.join(rep_save_path, 'score_dict')
  y = party_labels[:, 0]
  if not os.path.isfile(score_dict_path) or embeddings_were_run or DO_SCORES:
    for method in embeddings:
      if embeddings[method] is not None:
        print('computing eval_scores for method %s' % method)
        score_dict[method] = scores(
            embeddings[method]['W'], y, random_state=RANDOM_SEED + i * 100)
    with open(score_dict_path, 'w') as f:
      f.write(json.dumps(score_dict))
  else:
    print('loading eval scores')
    with open(score_dict_path) as f:
      score_dict = json.loads(f.read())
  score_dicts.append(score_dict)
