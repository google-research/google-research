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

"""Analyze trained ResNets."""

import functools
from itertools import combinations
import json
import os
import pickle
import random
import re

from absl import app
from absl import flags
from absl import logging
import numpy as np
from scipy.spatial.distance import cosine
from scipy.special import logsumexp, softmax
from scipy.stats import entropy
from sklearn.decomposition import TruncatedSVD, randomized_svd
import tensorflow.compat.v2 as tf
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
from do_wide_and_deep_networks_learn_the_same_things.cifar_train import load_test_data, preprocess_data
from do_wide_and_deep_networks_learn_the_same_things.efficient_CKA import *
from do_wide_and_deep_networks_learn_the_same_things.linear_probe import load_linear_probe_test_data

tf.enable_v2_behavior()

FLAGS = flags.FLAGS
flags.DEFINE_integer('cka_batch', 256, 'Batch size used to approximate CKA')
flags.DEFINE_integer('cka_iter', 10,
                     'Number of iterations to run minibatch CKA approximation')
flags.DEFINE_string('model_dir', '', 'Path to where the trained model is saved')
flags.DEFINE_integer('model_depth', 14,
                     'Only run analysis for models of this depth')
flags.DEFINE_integer('model_width', 1,
                     'Only run analysis for models of this width multiplier')
flags.DEFINE_string('experiment_dir', None,
                    'Path to where the trained model is saved')
flags.DEFINE_integer('model_depth2', 20,
                     'Only run analysis for models of this depth')
flags.DEFINE_integer('model_width2', 2,
                     'Only run analysis for models of this width multiplier')
flags.DEFINE_integer('epoch', 300, 'Use ckpt from this epoch')
flags.DEFINE_string('dataset', 'cifar10',
                    'Name of dataset used (CIFAR-10 of CIFAR-100)')


def get_configs(experiment_dir):
  """Return all unique width-depth configs in an experiment directory."""
  all_configs = set()
  for model_dir in tf.io.gfile.listdir(experiment_dir):
    if not tf.io.gfile.isdir(os.path.join(experiment_dir, model_dir)):
      continue
    if not model_dir.startswith('cifar'):
      continue
    depth, width = parse_depth_width(model_dir)
    all_configs.add((depth, width))
  return list(all_configs)


def get_cifar_labels(dataset_name='cifar10', data_path=None):
  """Get test labels from different versions of CIFAR-10 datasets (full or subsampled)."""
  dataset = load_test_data(128, dataset_name=dataset_name, data_path=data_path)
  all_labels = []
  for _, labels in dataset:
    all_labels.extend(labels.numpy())
  all_labels = np.array(all_labels)
  return all_labels


def get_preds(model_dir):
  """Get predictions on test set from a model directory."""
  preds = pickle.load(
      tf.io.gfile.GFile(os.path.join(model_dir, 'test_preds.pkl'), 'rb'))
  preds_int = np.argmax(preds, axis=1)
  return preds_int


def load_cka(model_dir):
  """Load internal CKA (between a layer and every other layer of the same model) from a model directory."""
  files = [
      f for f in tf.io.gfile.listdir(model_dir) if 'cka_within_model_256.pkl' in f
  ]
  pkl_path = os.path.join(model_dir, files[0])
  cka = pickle.load(tf.io.gfile.GFile(pkl_path, 'rb'))
  return cka


def error_class_distribution(preds, labels):
  """Compute the distribution of wrong predictions across all classes."""
  error_idx = np.where(preds != labels)[0]
  error_class = labels[error_idx]
  return list(error_idx), list(error_class)


def save_layer_names(experiment_dir):
  """Load all models in an experiment directory and save all layer names within each model."""
  for model_dir in tf.io.gfile.listdir(experiment_dir):
    if 'depth-' not in model_dir or 'copy-' in model_dir:
      continue
    logging.info(model_dir)
    save_path = os.path.join(experiment_dir, model_dir + '.txt')
    if tf.io.gfile.exists(save_path):
      logging.info('Aborting...')
      continue
    model = tf.keras.models.load_model(os.path.join(experiment_dir, model_dir))
    all_layers = [layer.name for layer in model.layers]
    with tf.io.gfile.GFile(save_path, 'w') as f:
      json.dump({'layer_names': all_layers}, f)


def cka_within_model_variance(experiment_dir):
  """Compute variance of internal CKA scores for all model configurations in an experiment directory."""
  all_model_prefix = [
      m for m in tf.io.gfile.listdir(FLAGS.experiment_dir)
      if 'cifar-' in m and 'copy-' not in m and 'txt' not in m
  ]
  for prefix in all_model_prefix:
    all_cka_within = []
    all_models = [prefix] + [prefix + ('-copy-%d' % c) for c in range(1, 10)]
    for model in all_models:
      pkl_path = os.path.join(experiment_dir, model, 'cka_within_model_256.pkl')
      if not tf.io.gfile.exists(pkl_path):
        continue
      all_cka_within.append(
          pickle.load(tf.io.gfile.GFile(pkl_path, 'rb')).numpy().flatten())
    all_cka_within = np.vstack(all_cka_within)
    cka_within_var = np.var(all_cka_within, axis=0)
    cka_within_var = list(cka_within_var.flatten().astype(np.float64))
    save_path = os.path.join(experiment_dir,
                             prefix + '_cka_within_model_256_variance.txt')
    with tf.io.gfile.GFile(save_path, 'w') as f:
      json.dump({'variance': cka_within_var}, f)


def cka_across_model_variance(experiment_dir):
  """Compute variance of between-model CKA scores for all model configurations in an experiment directory."""
  experiment_dir_cka = os.path.join(experiment_dir, 'cka_across_models')
  all_prefix = set(
      ['_'.join(f.split('_')[:-1])
       for f in tf.io.gfile.listdir(experiment_dir_cka)])
  for prefix in all_prefix:
    all_files = [
        f for f in tf.io.gfile.listdir(experiment_dir_cka)
        if f.startswith(prefix)
    ]
    all_cka_across = []
    for f in all_files:
      pkl_path = os.path.join(experiment_dir, 'cka_across_models', f)
      all_cka_across.append(pickle.load(
          tf.io.gfile.GFile(pkl_path, 'rb')).flatten())
    all_cka_across = np.vstack(all_cka_across)
    cka_across_var = np.var(all_cka_across, axis=0)
    cka_across_var = list(cka_across_var.flatten().astype(np.float64))
    save_path = os.path.join(experiment_dir,
                             prefix + '_cka_across_model_256_variance.txt')
    with tf.io.gfile.GFile(save_path, 'w') as f:
      json.dump({'variance': cka_across_var}, f)


def normalize_activations(act):
  """Normalize along each row so that the norm of activations produced by each example is 1."""
  act = act.reshape(act.shape[0], -1)
  act_norm = np.linalg.norm(act, axis=1)
  act /= act_norm[:, None]
  return act


def get_activations(images, model, normalize_act=False):
  """Return a list of activations obtained from a model on a set of images."""
  input_layer = model.input
  layer_outputs = [layer.output for layer in model.layers]
  get_layer_outputs = K.function(input_layer, layer_outputs)
  activations = get_layer_outputs(images)
  if normalize_act:
    activations = [normalize_activations(act) for act in activations]
  return activations


def save_predictions(dataset, model_dir):
  """Save test predictions made by the model saved in a given directory."""
  model = tf.keras.models.load_model(model_dir)
  preds = model.predict(dataset, verbose=1)
  preds = np.apply_along_axis(softmax, 1, preds)  # apply softmax
  out_dir = os.path.join(model_dir, 'test_preds.pkl')
  with tf.io.gfile.GFile(out_dir, 'wb') as f:
    pickle.dump(preds, f)


def parse_depth_width(filename, return_seed=False):
  """Extract model configuration (depth, width), and optionally model seed number, from a filename."""
  if 'depth-' in filename:
    depth = re.findall(r'depth-\d+', filename)[0]
    depth = int(depth.split('-')[-1])
  else:
    depth = re.findall(r'depth_\d+', filename)[0]
    depth = int(depth.split('_')[-1])
  if 'width' not in filename:
    width = 1
  elif 'width-' in filename:
    width = re.findall(r'width-\d+', filename)[0]
    width = int(width.split('-')[-1])
  else:
    width = re.findall(r'width_\d+', filename)[0]
    width = int(width.split('_')[-1])
  if return_seed:
    if 'copy' not in filename:
      copy = 0
    else:
      copy = re.findall(r'copy-\d+', filename)[0]
      copy = int(copy.split('-')[-1])
    return depth, width, copy
  else:
    return depth, width


def find_stack_markers(model):
  """Finds the layers where a new stack starts."""
  stack_markers = []
  for i, layer in enumerate(model.layers):
    if i == 0:
      continue
    if 'conv' in layer.name:
      conv_weights_shape = layer.get_weights()[0].shape
      if conv_weights_shape[-1] != conv_weights_shape[-2] and conv_weights_shape[
          0] != 1 and conv_weights_shape[-2] % 16 == 0:
        stack_markers.append(i)
  assert len(stack_markers) == 2
  return stack_markers


def convert_one_hot(preds, n_class=10):
  """Convert integer predictions to one-hot encodings."""
  n_data = preds.shape[0]
  one_hot_preds = np.zeros((n_data, n_class))
  one_hot_preds[np.arange(preds.shape[0]), preds] = 1
  return one_hot_preds


def mean_kl(data1, data2):
  """Compute mean KL divergence (across all examples) between 2 data distributions."""
  log_softmax1 = data1 - logsumexp(data1, axis=-1, keepdims=True)
  log_softmax2 = data2 - logsumexp(data2, axis=-1, keepdims=True)
  kl = np.sum(np.exp(log_softmax1) * (log_softmax1 - log_softmax2), -1)
  return np.mean(kl)


def hellinger(p, q):
  """Compute Hellinger distance between 2 distributions."""
  return np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)


def prediction_entropy(experiment_dir, n_copies=10):
  """Compute entropy of test predictions for all model configurations in an experiment directory."""
  all_prefix = []
  for model_dir in tf.io.gfile.listdir(experiment_dir):
    if 'copy' not in model_dir and 'cifar' in model_dir:
      all_prefix.append(os.path.join(experiment_dir, model_dir))

  entropy_result = {}
  for m in all_prefix:
    depth, width = parse_depth_width(m)
    all_files = [m]
    for i in range(n_copies):
      model_dir = m + '-copy-%d' % i
      if tf.io.gfile.exists(model_dir):
        all_files.append(model_dir)
    model_pairs = combinations(all_files, 2)
    all_entropies = []
    for m1, m2 in model_pairs:
      data1 = pickle.load(tf.io.gfile.GFile(
          os.path.join(m1, 'test_preds.pkl'), 'rb'))
      data2 = pickle.load(tf.io.gfile.GFile(
          os.path.join(m2, 'test_preds.pkl'), 'rb'))
      if np.isnan(data1).any() or np.isnan(data2).any():
        print(m1, m2)
        continue
      all_entropies.extend(
          [entropy(data1[i, :], data2[i, :]) for i in range(data1.shape[0])])

    if all_entropies:
      avg_entropy = sum(all_entropies) / len(all_entropies)
      entropy_result[(depth, width)] = avg_entropy
  return entropy_result


def compare_preds_empirical_probability(experiment_dir, n_example=10000):
  """Compare empirical distribution of predictions (computed from 10 seeds) for all model configurations in an experiment directory."""
  count_dict = {}
  pred_dict = {}

  for model_dir in tf.io.gfile.listdir(experiment_dir):
    model_dir = os.path.join(experiment_dir, model_dir)
    if not tf.io.gfile.isdir(model_dir) or 'cifar' not in model_dir:
      continue
    depth, width = parse_depth_width(model_dir)
    data = pickle.load(
        tf.io.gfile.GFile(os.path.join(model_dir, 'test_preds.pkl'), 'rb'))
    pred_label = np.argmax(data, axis=1)
    pred_label = convert_one_hot(pred_label)
    if (depth, width) in count_dict:
      count_dict[(depth, width)] += 1
    else:
      count_dict[(depth, width)] = 1
    if (depth, width) in pred_dict:
      pred_dict[(depth, width)] += pred_label
    else:
      pred_dict[(depth, width)] = pred_label

  for k, v in pred_dict.items():
    v /= count_dict[k]
    assert ([s == 1 for s in np.sum(v, axis=1)])
    v = v.flatten()
    pred_dict[k] = v

  all_models = list(pred_dict.keys())
  all_models.sort()
  n_models = len(all_models)
  heatmap_hellinger = np.zeros((n_models, n_models))
  for i in range(n_models):
    for j in range(i + 1):
      hellinger_val = hellinger(pred_dict[all_models[i]],
                                pred_dict[all_models[j]]) / n_example
      heatmap_hellinger[i][j] = hellinger_val
      heatmap_hellinger[j][i] = hellinger_val
  return heatmap_hellinger, all_models


def compare_preds_similarity(experiment_dir, n_example=10000):
  """Compute KL divergence of prediction distributions for each pair of model configuration in an experiment directory.

  TODO: average over all seeds
  """
  true_labels = get_cifar_labels()
  all_models = []
  for model_dir in tf.io.gfile.listdir(experiment_dir):
    if 'cifar-' not in model_dir or 'copy' in model_dir:
      continue
    pred_path = os.path.join(experiment_dir, model_dir, 'test_preds.pkl')
    if tf.io.gfile.exists(pred_path):
      all_models.append(pred_path)
  n_models = len(all_models)

  all_models.sort(key=parse_depth_width)
  model_configs = [parse_depth_width(m) for m in all_models]

  all_pickles = []
  for path in all_models:
    with tf.io.gfile.GFile(path, 'rb') as f:
      data = pickle.load(f)
    assert data.shape[0] == n_example
    all_pickles.append(data)

  heatmap_abs_sim_score = np.ones((n_models, n_models))
  heatmap_kl = np.zeros((n_models, n_models))
  accuracy = np.zeros((n_models,))
  for i in range(n_models):
    data1 = all_pickles[i]
    pred_label1 = np.argmax(data1, axis=1)
    accuracy[i] = np.mean(pred_label1 == true_labels)
    for j in range(i + 1):
      data2 = all_pickles[j]
      pred_label2 = np.argmax(data2, axis=1)
      abs_sim_score = np.sum(pred_label1 == pred_label2) / n_example
      heatmap_abs_sim_score[i, j] = abs_sim_score
      heatmap_abs_sim_score[j, i] = abs_sim_score
      heatmap_kl[i, j] = mean_kl(data1, data2)
      heatmap_kl[j, i] = mean_kl(data2, data1)  # KL is asymmetric.

  return heatmap_kl, heatmap_abs_sim_score, accuracy, model_configs


def compute_sparsity(model_dir):
  """Compute level of sparsity in activations."""
  model = tf.keras.models.load_model(model_dir)
  test_dataset = load_test_data(100)
  total_sparsity = None
  count = 0
  for images, _ in test_dataset:
    count += 1
    activations = get_activations(images, model)
    all_sparsity = []
    for act in activations:
      act = act.flatten()
      all_sparsity.append(np.mean(act == 0))
    if total_sparsity is None:
      total_sparsity = np.array(all_sparsity)
    else:
      total_sparsity += np.array(all_sparsity)

  total_sparsity /= count
  save_path = os.path.join(model_dir, 'activation_sparsity.txt')
  with tf.io.gfile.GFile(save_path, 'w') as f:
    json.dump({'sparsity': list(total_sparsity)}, f)


def compute_sparsity_per_unit(model_dir):
  """Compute level of sparsity in each layer activation."""
  model = tf.keras.models.load_model(model_dir)
  test_dataset = load_test_data(100)
  total_sparsity = None
  count = 0
  for images, _ in test_dataset:
    count += 1
    activations = get_activations(images, model)
    all_sparsity = []
    for act in activations:
      act = act.reshape([act.shape[0], -1])
      all_sparsity.append(np.mean(act == 0, axis=0))
    if total_sparsity is None:
      total_sparsity = [np.array(s) for s in all_sparsity]
    else:
      for i, s in enumerate(total_sparsity):
        total_sparsity[i] = s + all_sparsity[i]

  total_sparsity = [s / count for s in total_sparsity]
  save_path = os.path.join(model_dir, 'activation_sparsity_per_unit.txt')
  with tf.io.gfile.GFile(save_path, 'w') as f:
    json.dump({'sparsity': [list(s) for s in total_sparsity]}, f)


def convert_bn_to_train_mode(model):
  """Convert a trained model with batch norm layers to run in train mode."""
  bn_layers = [
      i for i, layer in enumerate(model.layers)
      if 'batch_normalization' in layer.name
  ]
  model_config = model.get_config()
  for i in bn_layers:
    model_config['layers'][i]['inbound_nodes'][0][0][-1]['training'] = True
  new_model = model.from_config(model_config)
  for i, layer in enumerate(new_model.layers):
    layer.set_weights(model.layers[i].get_weights())
  return new_model


def epoch_pc(experiment_dir,
             batch_size=256,
             data_path=None,
             dataset_name='cifar10',
             use_train_mode=False,
             n_iter=1):
  """For each model ckpt, compute first PC of activations of each layer in that ckpt and save to a file."""
  if use_train_mode:
    out_dir = os.path.join(experiment_dir,
                           'first_pc_all_epochs_bn_train_mode.pkl')
  else:
    out_dir = os.path.join(experiment_dir, 'first_pc_all_epochs.pkl')
  logging.info(out_dir)
  if tf.io.gfile.exists(out_dir):
    result = pickle.load(tf.io.gfile.GFile(out_dir, 'rb'))
  else:
    result = {}
  test_dataset = load_test_data(
      batch_size, data_path=data_path, dataset_name=dataset_name,
      shuffle=True).repeat()
  epoch_files = [f for f in tf.io.gfile.listdir(experiment_dir) if 'weights' in f]
  epoch_files.append('')  # include initialization
  for epoch_file in epoch_files:
    if 'ckpt' in epoch_file:
      epoch_no = int(epoch_file.split('.')[1])
    else:
      epoch_no = 0
    if epoch_no % 10 != 0:
      continue
    if epoch_no in result:
      continue
    model = tf.keras.models.load_model(os.path.join(experiment_dir, epoch_file))
    if use_train_mode:
      model = convert_bn_to_train_mode(model)

    n_layers = len(model.layers)
    avg_variance_explained = np.zeros((n_layers,))
    avg_pc = np.zeros((n_layers, batch_size))
    it = 0
    for images, _ in test_dataset:
      it += 1
      if it > n_iter:
        break
      all_activations = get_activations(images, model)
      for i, act in enumerate(all_activations):
        act = act.reshape(act.shape[0], -1)
        act -= np.mean(act, axis=0)
        svd = TruncatedSVD(n_components=1, random_state=0)
        svd.fit(act.T)
        avg_variance_explained[i] += svd.explained_variance_ratio_[0]
        act_pc = svd.components_.squeeze()
        avg_pc[i, :] += act_pc

    avg_variance_explained /= n_iter
    avg_pc /= n_iter
    result[epoch_no] = (avg_pc, avg_variance_explained)
    with tf.io.gfile.GFile(out_dir, 'wb') as f:
      pickle.dump(result, f)


def compute_cka_internal(model_dir,
                         data_path=None,
                         dataset_name='cifar10',
                         use_batch=True,
                         use_train_mode=False,
                         normalize_act=False):
  """Compute CKA score of each layer in a model to every other layer in the same model."""
  if dataset_name == 'cifar10':
    if use_train_mode:
      filename = 'cka_within_model_%d_bn_train_mode.pkl' % FLAGS.cka_batch
    else:
      filename = 'cka_within_model_%d.pkl' % FLAGS.cka_batch
  else:
    suffix = dataset_name.split('_')[-1]
    if use_train_mode:
      filename = 'cka_within_model_%d_%s_bn_train_mode.pkl' % (FLAGS.cka_batch,
                                                               suffix)
    else:
      filename = 'cka_within_model_%d_%s.pkl' % (FLAGS.cka_batch, suffix)
  if normalize_act:
    filename = filename.replace('.pkl', '_normalize_activations.pkl')
  out_dir = os.path.join(model_dir, filename)
  if tf.io.gfile.exists(out_dir):
    return

  model = tf.keras.models.load_model(model_dir)
  if use_train_mode:
    model = convert_bn_to_train_mode(model)

  n_layers = len(model.layers)
  cka = MinibatchCKA(n_layers)
  if use_batch:
    for _ in range(FLAGS.cka_iter):
      dataset = load_test_data(
          FLAGS.cka_batch,
          shuffle=True,
          dataset_name=dataset_name,
          n_data=10000)
      for images, _ in dataset:
        cka.update_state(get_activations(images, model, normalize_act))
  else:
    dataset = load_test_data(
        FLAGS.cka_batch, data_path=data_path, dataset_name=dataset_name)
    all_images = tf.concat([x[0] for x in dataset], 0)
    cka.update_state(get_activations(all_images, model))
  heatmap = cka.result().numpy()
  logging.info(out_dir)
  with tf.io.gfile.GFile(out_dir, 'wb') as f:
    pickle.dump(heatmap, f)


def epoch_cka(experiment_dir,
              use_batch=True,
              data_path=None,
              dataset_name='cifar10',
              last_epoch=300,
              curr_epoch=None,
              use_train_mode=False):
  """Compute CKA score of each layer in a final model to every other layer of the same model at earlier stages of training."""
  n_layers = 0
  epoch_files = [f for f in tf.io.gfile.listdir(experiment_dir) if 'weights' in f]
  if curr_epoch is not None:
    if curr_epoch == 0:
      epoch_files = ['']
    else:
      epoch_files = [f for f in epoch_files if 'weights.%d.' % curr_epoch in f]
  model_pairs = [(f, 'weights.%d.ckpt' % last_epoch) for f in epoch_files]
  if use_train_mode:
    save_dir = os.path.join(experiment_dir, 'cka_across_epochs_bn_train_mode')
  else:
    save_dir = os.path.join(experiment_dir, 'cka_across_epochs')
  if not tf.io.gfile.exists(save_dir):
    tf.io.gfile.mkdir(save_dir)

  for m1, m2 in model_pairs:
    if 'ckpt' not in m1:
      epoch_no = 0
    else:
      epoch_no = int(m1.split('.')[1])
    if epoch_no % 10 != 0:
      if epoch_no > 10:
        continue
    if epoch_no == last_epoch:
      continue
    out_dir = os.path.join(
        save_dir, 'batch_%d_epoch_%d_epoch_%d.pkl' %
        (FLAGS.cka_batch, epoch_no, last_epoch))
    logging.info(out_dir)
    if tf.io.gfile.exists(out_dir):
      logging.info('Aborting...')
      continue

    model1 = tf.keras.models.load_model(os.path.join(experiment_dir, m1))
    model2 = tf.keras.models.load_model(os.path.join(experiment_dir, m2))
    if use_train_mode:
      model1 = convert_bn_to_train_mode(model1)
      model2 = convert_bn_to_train_mode(model2)

    if not n_layers:
      n_layers = len(model1.layers)

    cka = MinibatchCKA(n_layers, across_models=True)
    if use_batch:
      for _ in range(FLAGS.cka_iter):
        dataset = load_test_data(
            FLAGS.cka_batch,
            shuffle=True,
            data_path=data_path,
            dataset_name=dataset_name)
        for images, _ in dataset:
          activations1 = get_activations(images, model1)
          activations2 = get_activations(images, model2)
          cka.update_state_across_models(activations1, activations2)

    heatmap = cka.result().numpy()
    with tf.io.gfile.GFile(out_dir, 'wb') as f:
      pickle.dump(heatmap, f)


def compute_across_seed_cka(depth,
                            width,
                            use_batch=True,
                            data_path=None,
                            dataset_name='cifar10',
                            normalize_act=False):
  """For pairs of models that share the same depth & width, compute CKA score of each layer in a model to every other layer in the other model."""
  model_files = [
      f for f in tf.io.gfile.listdir(FLAGS.experiment_dir) if 'width-%d-' %
      width in f and 'depth-%d' % depth in f and not f.endswith('.txt')
  ]
  n_layers = 0
  model_pairs = combinations(model_files, 2)
  model_pairs = list(model_pairs)
  pair_count = 0
  for m1, m2 in model_pairs:
    _, _, copy1 = parse_depth_width(m1, return_seed=True)
    _, _, copy2 = parse_depth_width(m2, return_seed=True)
    if copy_1 > 10 or copy_2 > 10:
      continue
    pair_count += 1
    out_dir = os.path.join(
        FLAGS.experiment_dir, 'cka_across_models',
        'cka_across_models_depth_%d_width_%d_batch_%d_copy_%d_copy_%d.pkl' %
        (depth, width, FLAGS.cka_batch, copy1, copy2))
    if normalize_act:
      out_dir = out_dir.replace('.pkl', '_normalize_activations.pkl')
    logging.info(out_dir)
    if tf.io.gfile.exists(out_dir):
      logging.info('Aborting...')
      continue

    model1 = tf.keras.models.load_model(os.path.join(FLAGS.experiment_dir, m1))
    model2 = tf.keras.models.load_model(os.path.join(FLAGS.experiment_dir, m2))
    if not n_layers:
      n_layers = len(model1.layers)

    cka = MinibatchCKA(n_layers, across_models=True)
    #cka2 = MinibatchCKA(n_layers * 2)
    if use_batch:
      for _ in range(FLAGS.cka_iter):
        dataset = load_test_data(
            FLAGS.cka_batch,
            shuffle=True,
            data_path=data_path,
            dataset_name=dataset_name)
        for images, _ in dataset:
          activations1 = get_activations(images, model1, normalize_act)
          activations2 = get_activations(images, model2, normalize_act)
          cka.update_state_across_models(activations1, activations2)
          #test_CKA(n_layers, n_layers, activations1, activations2, cka1=cka, cka2=cka2)

    heatmap = cka.result().numpy()
    with tf.io.gfile.GFile(out_dir, 'wb') as f:
      pickle.dump(heatmap, f)


def compute_width_depth_cka(model_list1,
                            model_list2,
                            use_batch=True,
                            data_path=None,
                            dataset_name='cifar10',
                            normalize_act=False):
  """Computes CKA score of each layer in a model to every other layer in another model."""
  all_pairs = [(i, j) for i in model_list1 for j in model_list2]
  random.seed(0)
  if len(all_pairs) < 20:
    model_pairs = random.sample(all_pairs, len(all_pairs))
  else:
    model_pairs = random.sample(all_pairs, 20)
  for m1, m2 in model_pairs:
    if m1 == m2:  # check if we pick the same seed
      continue
    depth1, width1, copy1 = parse_depth_width(m1, return_seed=True)
    depth2, width2, copy2 = parse_depth_width(m2, return_seed=True)
    logging.info(m1, m2)
    out_dir = os.path.join(
        FLAGS.experiment_dir, 'cka_across_models',
        'cka_across_models_depth_%d_width_%d_copy_%d_depth_%d_width_%d_copy_%d_batch_%d.pkl'
        % (depth1, width1, copy1, depth2, width2, copy2, FLAGS.cka_batch))
    if normalize_act:
      out_dir = out_dir.replace('.pkl', 'normalize_activations.pkl')
    logging.info(out_dir)
    if tf.io.gfile.exists(out_dir):
      logging.info('Aborting...')
      continue

    model1 = tf.keras.models.load_model(m1)
    model2 = tf.keras.models.load_model(m2)
    n_layers = len(model1.layers)
    n_layers2 = len(model2.layers)
    cka = MinibatchCKA(n_layers, n_layers2, across_models=True)
    #cka2 = MinibatchCKA(n_layers + n_layers2)
    if use_batch:
      for _ in range(FLAGS.cka_iter):
        dataset = load_test_data(
            FLAGS.cka_batch,
            shuffle=True,
            data_path=data_path,
            dataset_name=dataset_name)
        for images, _ in dataset:
          activations1 = get_activations(images, model1, normalize_act)
          activations2 = get_activations(images, model2, normalize_act)
          cka.update_state_across_models(activations1, activations2)
          #test_CKA(n_layers, n_layers2, activations1, activations2, cka1=cka, cka2=cka2)
    heatmap = cka.result().numpy()
    with tf.io.gfile.GFile(out_dir, 'wb') as f:
      pickle.dump(heatmap, f)


def get_cosine_sim_activation(model_dir,
                              data_path=None,
                              dataset_name='cifar10'):
  """Computes cosine similarity between activations of layers within the same block."""
  out_dir = os.path.join(model_dir, 'cosine_sim_within_model.pkl')
  if tf.io.gfile.exists(out_dir):
    return
  model = tf.keras.models.load_model(model_dir)
  n_layers = len(model.layers)
  dataset = load_test_data(1, data_path=data_path, dataset_name=dataset_name)
  result = np.zeros((n_layers, n_layers))
  n_data = 0
  for images, _ in dataset:
    n_data += 1
    activations = get_activations(images, model)
    for i in range(n_layers):
      for j in range(i + 1, n_layers):
        if activations[i].shape != activations[j].shape:
          continue
        dist = cosine(activations[i].flatten(), activations[j].flatten())
        result[i][j] += dist
        result[j][i] += dist
  result /= n_data
  result = 1 - result
  logging.info(out_dir)
  with tf.io.gfile.GFile(out_dir, 'wb') as f:
    pickle.dump(result, f)


def parse_linear_probe_results(model_path, pooling=False):
  """Summarize linear probe accuracies for all Batch Norm and Add layers."""
  depth, width = parse_depth_width(model_path)
  model = tf.keras.models.load_model(model_path)
  depth, width = parse_depth_width(model_path)
  layer_name_file = [
      f for f in tf.io.gfile.listdir(os.path.dirname(model_path))
      if f.endswith('txt') and ('depth-%d-width-%d' % (depth, width)) in f
  ][0]
  layer_names = json.load(
      tf.io.gfile.GFile(
          os.path.join(os.path.dirname(model_path), layer_name_file),
          'r'))['layer_names']
  add_layers = [i for i, l in enumerate(layer_names) if 'add' in l]
  bn_layers = [i - 1 for i in add_layers]
  add_lp_acc, bn_lp_acc = [], []

  for list_idx, layer_list in enumerate([add_layers, bn_layers]):
    for layer_idx in layer_list:
      logging.info('layer %d', layer_idx)
      layer_output = model.layers[layer_idx].output
      total_dim = np.prod(np.array(layer_output.get_shape().as_list()[1:]))
      test_dataset = load_linear_probe_test_data(model, layer_idx, (total_dim,),
                                                 512)
      if pooling:
        all_files = [
            f for f in tf.io.gfile.listdir(model_path)
            if 'layer-%d-' % layer_idx in f and 'pooling' in f
        ]
      else:
        all_files = [
            f for f in tf.io.gfile.listdir(model_path)
            if 'layer-%d-' % layer_idx in f and 'pooling' not in f
        ]
      best_acc = 0
      for f in all_files:
        lp_model = tf.keras.models.load_model(os.path.join(model_path, f))
        lp_model.compile(
            'sgd',
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['acc'])

        n_correct_preds, n_val = 0, 0
        for (_, (images, labels)) in enumerate(test_dataset.take(-1)):
          logits = lp_model(images, training=False)
          correct_preds = tf.equal(tf.argmax(input=logits, axis=1), labels)
          n_correct_preds += correct_preds.numpy().sum()
          n_val += int(tf.shape(labels)[0])
        acc = n_correct_preds / n_val
        if acc > best_acc:
          best_acc = acc

      if list_idx == 0:
        add_lp_acc.append(best_acc)
      else:
        bn_lp_acc.append(best_acc)

  results = {}
  for layer, acc in zip(add_layers, add_lp_acc):
    results[layer] = acc
  for layer, acc in zip(bn_layers, bn_lp_acc):
    results[layer] = acc
  if pooling:
    save_path = os.path.join(model_path, 'linear_probe_results_pooling.pkl')
  else:
    save_path = os.path.join(model_path, 'linear_probe_results.pkl')
  pickle.dump(results, tf.io.gfile.GFile(save_path, 'wb'))


def save_pc_values(dataset_name,
                   model_dir,
                   batch_size=10000,
                   top=None,
                   bottom=None,
                   save_dim_coefficients=False,
                   frac=0.5):
  """Save first PC-related metadata for each layer activation.

  Either output projected values onto the first PC and the
  fraction of variation explained, or the PC itself.
  """
  out_dir = os.path.join(model_dir, 'all_pc_values_%d.pkl' % batch_size)
  out_dir_explained_ratio = os.path.join(
      model_dir, 'all_pc_explained_ratio_%d.pkl' % batch_size)
  if top is not None:  # remove outliers and recalculate PC
    out_dir = out_dir.replace('.pkl', '_no_outlier_remove_%.2f.pkl' % frac)
    out_dir_explained_ratio = out_dir_explained_ratio.replace(
        '.pkl', '_no_outlier_remove_%.2f.pkl' % frac)

  if tf.io.gfile.exists(out_dir_explained_ratio):
    exit()

  model = tf.keras.models.load_model(model_dir)
  if 'weights' in model_dir or 'copy-10' in model_dir or 'copy-11' in model_dir or 'copy-12' in model_dir:
    model = convert_bn_to_train_mode(model)
    out_dir = out_dir.replace('.pkl', '_bn_train_mode.pkl')
    out_dir_explained_ratio = out_dir_explained_ratio.replace(
        '.pkl', '_bn_train_mode.pkl')

  test_dataset = load_test_data(batch_size, dataset_name=dataset_name)
  images, _ = test_dataset.__iter__().next()
  all_activations = get_activations(images, model)
  bs = images.numpy().shape[0]
  all_pc_values = []
  all_pc_explained_ratio = []
  all_coefficients = []
  all_coefficients_no_outliers = []
  for i, act in enumerate(all_activations):
    if top is not None and (i > top or i < bottom):
      continue
    act = act.reshape(bs, -1)
    processed_act = act - np.mean(act, axis=0)
    svd = TruncatedSVD(n_components=1, random_state=0)
    svd.fit(processed_act.T)
    act_pc = svd.components_.squeeze()

    if save_dim_coefficients:  # save the PC itself
      U, _, _ = randomized_svd(
          processed_act.T, n_components=1, n_iter=5, random_state=None)
      all_coefficients.append(U.squeeze())

    if top is None:
      all_pc_values.append(act_pc)
      all_pc_explained_ratio.append(svd.explained_variance_ratio_[0])
    else:  # remove outliers for the corresponding layers
      n_examples = len(act_pc)
      outlier_idx = np.argsort(
          np.abs(act_pc))[int(n_examples *
                              frac):]  # remove the bottom {frac} of the data
      selected_idx = np.array([i for i in range(bs) if i not in outlier_idx])
      act_no_outlier = act[selected_idx, :]
      processed_act = act_no_outlier - np.mean(act_no_outlier, axis=0)

      if save_dim_coefficients:
        U, _, _ = randomized_svd(
            processed_act.T, n_components=1, n_iter=5, random_state=None)
        all_coefficients_no_outliers.append(U.squeeze())
      else:
        svd = TruncatedSVD(n_components=1, random_state=0)
        svd.fit(processed_act.T)
        act_pc = svd.components_.squeeze()
        all_pc_values.append(act_pc)
        all_pc_explained_ratio.append(svd.explained_variance_ratio_[0])

  if save_dim_coefficients:
    out_dir = os.path.join(model_dir,
                           'all_dim_coefficients_%d.pkl' % batch_size)
    pickle.dump(all_coefficients, tf.io.gfile.GFile(out_dir, 'wb'))
    out_dir = os.path.join(
        model_dir,
        'all_dim_coefficients_%d_no_outlier_remove_half.pkl' % batch_size)
    pickle.dump(all_coefficients_no_outliers, tf.io.gfile.GFile(out_dir, 'wb'))
  else:
    pickle.dump(all_pc_values, tf.io.gfile.GFile(out_dir, 'wb'))
    pickle.dump(all_pc_explained_ratio,
                tf.io.gfile.GFile(out_dir_explained_ratio, 'wb'))


def CKA_without_dominant_images(dataset_name,
                                model_dir,
                                batch_size=10000,
                                top=None,
                                bottom=None,
                                frac=0.5):
  """Compute internal CKA when {frac} of most dominant images are removed."""
  # TODO: include option for concatenating activations from top to bottom and computing first PC from there
  if dataset_name == 'cifar10':
    filename = 'cka_within_model_%d_remove_%.2f_dominant_egs.pkl' % (
        FLAGS.cka_batch, frac)
  else:
    suffix = dataset_name.split('_')[-1]
    filename = 'cka_within_model_%d_%s_remove_%.2f_dominant_egs.pkl' % (
        FLAGS.cka_batch, suffix)
  out_dir = os.path.join(model_dir, filename)
  if tf.io.gfile.exists(out_dir):
    return

  model = tf.keras.models.load_model(model_dir)
  test_dataset = load_test_data(batch_size, dataset_name=dataset_name)
  images, _ = test_dataset.__iter__().next()
  all_activations = get_activations(images, model)
  bs = images.numpy().shape[0]

  if bottom is None:
    act = all_activations[top]
    act = act.reshape(bs, -1)
    processed_act = act - np.mean(act, axis=0)
    svd = TruncatedSVD(n_components=1, random_state=0)
    svd.fit(processed_act.T)
    act_pc = svd.components_.squeeze()
    n_examples = len(act_pc)
    outlier_idx = np.argsort(
        np.abs(act_pc))[-int(n_examples *
                            frac):]  # remove the top {frac} most dominant datapoints

  # Create a new filtered dataset
  test_dataset = tfds.load(name=dataset_name, split='test', as_supervised=True)
  test_dataset = test_dataset.batch(1)
  all_images, all_labels = [], []
  count = 0
  for data in test_dataset.as_numpy_iterator():
    all_images.append(data[0].squeeze())
    all_labels.append(data[1].item())
    count += 1
    if count - 1 in outlier_idx:
      continue
  all_images = np.stack(all_images)

  # Compute internal CKA with the new dataset
  n_layers = len(model.layers)
  cka = MinibatchCKA(n_layers)
  for _ in range(FLAGS.cka_iter):
    new_dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
    new_dataset = new_dataset.map(
        functools.partial(preprocess_data, is_training=False))
    new_dataset = new_dataset.shuffle(buffer_size=int(batch_size * frac))
    new_dataset = new_dataset.batch(FLAGS.cka_batch, drop_remainder=False)

    for images, _ in new_dataset:
      cka.update_state(get_activations(images, model))
  heatmap = cka.result().numpy()
  with tf.io.gfile.GFile(out_dir, 'wb') as f:
    pickle.dump(heatmap, f)


def main(argv):
  #epoch_cka(FLAGS.experiment_dir, dataset_name=FLAGS.dataset, use_train_mode=True, curr_epoch=0)
  #epoch_pc(FLAGS.experiment_dir, dataset_name=FLAGS.dataset, use_train_mode=True)
  #save_pc_values(FLAGS.dataset, FLAGS.experiment_dir, 10000, frac=0.01, top=450, bottom=460)
  #compute_cka_internal(FLAGS.experiment_dir, dataset_name=FLAGS.dataset)
  CKA_without_dominant_images(FLAGS.dataset, FLAGS.experiment_dir, top=300, frac=0.01)


if __name__ == '__main__':
  app.run(main)
