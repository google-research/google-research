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

# Lint as: python3
"""Activation clustering model.
"""

import collections
import functools
import os
import types

from activation_clustering import utils
from dec_da.ConvIDEC import ConvIDEC
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import tensorflow as tf
import yaml


class ACModel:
  """Activation clustering model.
  """

  def __init__(self, baseline_model, clustering_config,
               work_dir='/tmp/acmodel', restore=False, activation_model=None):
    """Activation clustering model.

    Args:
      baseline_model: a tf.keras model.
      clustering_config: a list of (activation_name, activation_config) pairs,
        where activation_config is a Python dict with required key 'n_clusters'
        and optional key 'filters'.
        The key 'filters' must map to a list of positive integers of length 4.
        If not provided, the default [32, 64, 128, 20] is used.  These values
        specify the number of filters in the convolutional autoencoder used
        internally by the DEC model, and the last value is the embedding
        dimension.

        For example:
        ```
        clustering_config = [
          ('activation', {'n_clusters': 5}),
          ('activation_18', {'n_clusters': 10, 'filters': [32, 64, 128, 10]})
        ]
        ```
      work_dir: filepath path where cached activations and clustering
        models will be written to and can be restored from later.
      restore: whether to restore models from work_dir.
      activation_model: used for extracting activations, if not provided it
        will be the baseline model
    """
    self.baseline_model = baseline_model
    self.clustering_config = clustering_config
    self.work_dir = work_dir
    self.activation_model = activation_model or self.baseline_model

    # validate filters
    for _, config in self.clustering_config:
      if 'filters' in config and len(config['filters']) != 4:
        raise NotImplementedError('The dependency dec-da library requires '
                                  '`filters` to be a list of 4 positive '
                                  'integers if provided in clustering_config.')

    if not os.path.exists(self.work_dir):
      os.makedirs(self.work_dir)

    self.activation_names = [p[0] for p in self.clustering_config]

    # baseline_model is assumed to be a classification model
    self.n_classes = self.baseline_model.output.shape[1]

    # the full surrogate model is the composition of clustering model followed
    # by the empirical posterior probability model given latent clusters.
    self.clustering_models = {}

    # for each activation, empirical_posteriors is a mapping from the cluster
    # assignment to the classes.
    self.empirical_posteriors = [
        np.zeros((config['n_clusters'], self.n_classes))
        for _, config in self.clustering_config
    ]

    # cache of training embeddings for querying nearest examples.
    self.training_embeddings = {}

    if not restore:
      with open(os.path.join(self.work_dir, 'clustering_config.yaml'),
                'w') as f:
        f.write(yaml.safe_dump(self.clustering_config))

      self.baseline_model.save(os.path.join(self.work_dir, 'baseline_model.h5'))
      if self.activation_model is not self.baseline_model:
        self.activation_model.save(
            os.path.join(self.work_dir, 'activation_model.h5'))

  @classmethod
  def restore(cls, work_dir):
    """Restores previously trained activation clustering model.

    Args:
      work_dir: the directory to restore models from.

    Returns:
      An ACModel instance.
    """
    print('restoring configuration')
    with open(os.path.join(work_dir, 'clustering_config.yaml'), 'r') as f:
      clustering_config = yaml.safe_load(f.read())

    print('restoring baseline model')
    baseline_model = tf.keras.models.load_model(
        os.path.join(work_dir, 'baseline_model.h5'))
    activation_model_path = os.path.join(work_dir, 'activation_model.h5')
    if os.path.exists(activation_model_path):
      activation_model = tf.keras.models.load_model(activation_model_path)
    else:
      activation_model = None

    ac_model = cls(
        baseline_model,
        clustering_config,
        work_dir,
        restore=True,
        activation_model=activation_model)
    ac_model.build_clustering_models()

    print('restoring clustering models')
    for activation_name in ac_model.activation_names:
      clustering_model = ac_model.clustering_models[activation_name]
      filepath = os.path.join(work_dir, 'clustering_{}'.format(activation_name),
                              'model_final.h5')
      clustering_model.load_weights(filepath)

    print('restoring empirical posteriors')
    empirical_posteriors_path = os.path.join(work_dir, 'empirical_posteriors')
    ac_model.empirical_posteriors = joblib.load(
        os.path.join(empirical_posteriors_path, 'empirical_posteriors.joblib'))

    print('restoring training embeddings')
    output_filename = os.path.join(work_dir, 'training_embeddings.npz')
    with open(output_filename, 'rb') as f:
      with np.load(f) as data:
        ac_model.training_embeddings = dict(data)

    return ac_model

  def get_centroids_list(self):
    """Gets list of clustering centroids.

    Returns:
      A list of len(self.activation_names) numpy arrays of shape
      (n_clusters, embedding_dim).
    """
    centroids_list = []
    for activation_name in self.activation_names:
      clustering_model = self.clustering_models[activation_name]

      centroids = clustering_model.model.get_layer(
          'clustering').clusters.numpy()

      centroids_list.append(centroids)

    return centroids_list

  def get_activations(self, data_batches, activation_names=None):
    """Gets activations from input data.

    Args:
      data_batches: iterable batches of (features, labels).  labels can be None.
      activation_names: if provided, get only specified activations.

    Returns:
      A dictionary whose keys are activation_names and values are numpy arrays.
    """
    if activation_names is None:
      activation_names = self.activation_names

    keys = activation_names + ['feature', 'label']
    activations_dict = {key: [] for key in keys}
    for i, batch in enumerate(data_batches):
      print('processing batch {}'.format(i))
      features, labels = batch
      acts = utils.get_activations(self.activation_model, activation_names,
                                   features)

      for an, act in zip(activation_names, acts):
        activations_dict[an].append(act)

      activations_dict['feature'].append(features)
      if labels is not None:
        activations_dict['label'].append(labels)

    if labels is None:
      del activations_dict['label']

    for k, v in activations_dict.items():
      activations_dict[k] = np.concatenate(v, axis=0)

    return activations_dict

  def get_activations_from_features(self, features, activation_names=None):
    """Gets activations_dict from a single batch of features.

    Args:
      features: a batch of features that can be input for baseline_model.
      activation_names: a list of activation names to get activations for.

    Returns:
       A dictionary whose keys are activation_names and values are numpy arrays.
    """
    data_batches = [(features, None)]

    return self.get_activations(data_batches, activation_names)

  def cache_activations(self, data_batches, tag='train'):
    """Gets activations from input data, and writes to disk.

    The cached activations will be written to:
    {self.work_dir}/activations/activations_{tag}.npz

    Each file stores a dict of numpy array with self.activation_names as keys.

    Args:
      data_batches: iterable batches of (features, labels).
      tag: added to the activation cache filename, e.g. 'train', 'test'.
    """
    activations_dict = self.get_activations(data_batches)

    activations_path = os.path.join(self.work_dir, 'activations')
    output_filename = os.path.join(activations_path,
                                   'activations_{}.npz'.format(tag))
    if not os.path.exists(activations_path):
      os.makedirs(activations_path)

    with open(output_filename, 'wb') as f:
      np.savez(f, **activations_dict)

  def load_activations_dict(self, activations_filename):
    """Loads activations_dict from disk.

    Args:
      activations_filename: filename for cached activations created by calling
        self.cache_activations.

    Returns:
       A dictionary whose keys are activation_names and values are numpy arrays.
    """
    with open(activations_filename, 'rb') as f:
      with np.load(f) as data:
        activations_dict = dict(data)

    return activations_dict

  def build_clustering_models(self):
    """Builds a clustering model for each activation.

    Populates the dict self.clustering_models.
    """
    default_filters = [32, 64, 128, 20]
    activation_shapes = utils.get_activation_shapes(self.activation_model,
                                                    self.activation_names)

    for (activation_name,
         activation_config), input_shape in zip(self.clustering_config,
                                                activation_shapes):
      n_clusters = activation_config['n_clusters']
      filters = activation_config.get('filters', default_filters)

      clustering_model = ConvIDEC(
          input_shape=input_shape,
          filters=filters,
          n_clusters=n_clusters
      )

      optimizer = tf.keras.optimizers.Adam()
      clustering_model.compile(
          optimizer=optimizer, loss=['kld', 'mse'], loss_weights=[0.1, 1.0])

      self.clustering_models[activation_name] = clustering_model

    self._patch_clustering_models_predict()

  def _patch_clustering_models_predict(self):
    """Patch clustering models' predict method to use predict_on_batch.
    """
    for clustering_model in self.clustering_models.values():
      # monkey patch the predict method
      instance_predict = functools.partial(
          utils.batched_predict_on_batch,
          model=clustering_model.model,
          batch_size=1024,
          index=0)
      clustering_model.predict = types.MethodType(instance_predict,
                                                  clustering_model)

  def fit(self, activations_dict, epochs=3, maxiter=280):
    """Fits the full activation clustering model.

    Components: clustering models and empirical posteriors.

    Args:
      activations_dict: Python dict of cached activations created by calling
        self.get_activations.
      epochs: DEC parameter.  The number of epochs during the pretraining phase.
      maxiter: DEC parameter.  The maximum number of iterations for clustering.
    """
    self.fit_clustering_models(activations_dict, epochs=epochs, maxiter=maxiter)
    self.fit_empirical_posteriors(activations_dict)

    print('caching training embeddings')
    self.training_embeddings = dict(
        zip(self.activation_names,
            self.clustering_extract_features(
                activations_dict=activations_dict)))

    output_filename = os.path.join(self.work_dir, 'training_embeddings.npz')
    with open(output_filename, 'wb') as f:
      np.savez(f, **self.training_embeddings)

  def fit_clustering_models(self, activations_dict, epochs=3, maxiter=280):
    """Fits the clustering models.

    Trained models are saved in:
    {self.work_dir}/clustering_{activation_name}/

    Args:
      activations_dict: Python dict of cached activations created by calling
        self.get_activations.
      epochs: DEC parameter.  The number of epochs during the pretraining phase.
      maxiter: DEC parameter.  The maximum number of iterations for clustering.
    """

    for activation_name, _ in self.clustering_config:
      clustering_model_save_dir = os.path.join(
          self.work_dir, 'clustering_{}'.format(activation_name))
      if not os.path.exists(clustering_model_save_dir):
        os.makedirs(clustering_model_save_dir)

      print('pretraining clustering model for activation: {}'.format(
          activation_name))
      activations = activations_dict[activation_name]
      clustering_model = self.clustering_models[activation_name]
      clustering_model.pretrain(
          activations,
          epochs=epochs,
          batch_size=32,
          save_dir=clustering_model_save_dir)

      print(
          'fitting clustering model for activation: {}'.format(activation_name))
      clustering_model.fit(
          activations,
          maxiter=maxiter,
          batch_size=32,
          save_dir=clustering_model_save_dir)

  def _clustering_model_call(self,
                             method_name,
                             features=None,
                             activations_dict=None):
    """Helper for calling methods on clustering models."""
    if activations_dict is None:
      activations_dict = self.get_activations_from_features(features)

    results = []
    for activation_name, _ in self.clustering_config:
      clustering_model = self.clustering_models[activation_name]
      activations = activations_dict[activation_name]
      method = getattr(clustering_model, method_name)
      results.append(method(activations))

    return results

  def clustering_predict(self, features=None, activations_dict=None):
    """Gets cluster assignment scores.

    Args:
      features: a batch of input data that the baseline model can process.
      activations_dict: Python dict of cached activations created by calling
        self.get_activations.

    Returns:
      A list of len(self.activation_names) numpy arrays of shape (batch_size,
      n_clusters).
    """
    return self._clustering_model_call(
        'predict',
        features=features,
        activations_dict=activations_dict)

  def clustering_predict_labels(self, features=None, activations_dict=None):
    """Gets cluster assignment.

    Args:
      features: a batch of input data that the baseline model can process.
      activations_dict: Python dict of cached activations created by calling
        self.get_activations.

    Returns:
      A list of len(self.activation_names) lists of length batch_size.
    """
    return self._clustering_model_call(
        'predict_labels',
        features=features,
        activations_dict=activations_dict)

  def clustering_extract_features(self, features=None, activations_dict=None):
    """Gets embedding.

    Args:
      features: a batch of input data that the baseline model can process.
      activations_dict: Python dict of cached activations created by calling
        self.get_activations.

    Returns:
      A list of len(self.activation_names) numpy arrays of shape
      (batch_size, embedding_dim)
    """
    return self._clustering_model_call(
        'extract_features',
        features=features,
        activations_dict=activations_dict)

  def fit_empirical_posteriors(self, activations_dict):
    """Fits the empirical posteriors.

    The empirical posteriors map clustering models' cluster assignments to
    labels.

    Args:
      activations_dict: Python dict of cached activations created by calling
        self.get_activations.
    """
    print('fitting empirical posteriors')
    clustering_labels = self.clustering_predict_labels(
        activations_dict=activations_dict)

    for i, (_, config) in enumerate(self.clustering_config):
      # cluster assignment of the i-th activation
      clustering_assignment = clustering_labels[i]

      # this is a numpy array of shape (n_clusters, n_classes)
      h = self.empirical_posteriors[i]

      n_clusters = config['n_clusters']
      for j in range(n_clusters):
        # indices of training examples that got assigned to cluster j
        idx = np.argwhere(clustering_assignment == j)[:, 0]

        # get counts of labels among examples in the j-th cluster.
        counter = collections.Counter(activations_dict['label'][idx])
        h[j] = [counter.get(k, 0.0) for k in range(self.n_classes)]

        # normalize row sums
        h[j] /= np.sum(h[j])

    empirical_posteriors_path = os.path.join(self.work_dir,
                                             'empirical_posteriors')
    if not os.path.exists(empirical_posteriors_path):
      os.makedirs(empirical_posteriors_path)
    joblib.dump(self.empirical_posteriors, os.path.join(
        empirical_posteriors_path, 'empirical_posteriors.joblib'))

  def predict_proba(self, features=None, activations_dict=None, weights=None):
    """Returns surrogate model's predicted probabilities.

    Args:
      features: a batch of input data that the baseline model can process.
      activations_dict: Python dict of cached activations created by calling
        self.get_activations.
      weights: a list of weights (need not be normalized), one for each
        activation.
    """
    # Equal weights are used if not provided by the user.
    if weights is None:
      weights = [1.0] * len(self.activation_names)

    scores = self.clustering_predict(
        features=features, activations_dict=activations_dict)

    def p(example_score):
      # example_score is a list of lentgh n_activations, for one data example
      # each element is a vector of shape (n_clusters,)

      result = np.zeros(self.n_classes)

      # iterate through activations and sum up the contributions to probability.
      for s, h, w in zip(example_score, self.empirical_posteriors, weights):
        # output probability based on one activation's clustering
        activation_prob = np.matmul(s, h)

        result += w * activation_prob

      return result / len(self.activation_names)

    return np.array(list(map(p, zip(*scores))))

  def evaluate(self, features=None, activations_dict=None, y=None):
    """Returns the surrogate model's predicted class labels.

    Args:
      features: a batch of input data that the baseline model can process.
      activations_dict: Python dict of cached activations created by calling
        self.get_activations.
      y: a list of target labels to calcualte accuracy scores with, typically
        the ground truth labels (for surrogate model's accuracy) or the labels
        predicted by the baseline model (for fidelity).

    Returns:
      A float, accuracy score.
    """
    print('evaluating')
    y_pred = self.predict(features=features, activations_dict=activations_dict)

    return accuracy_score(y, y_pred)

  def predict(self, features=None, activations_dict=None):
    """Returns the surrogate model's predicted class labels.

    Args:
      features: a batch of input data that the baseline model can process.
      activations_dict: Python dict of cached activations created by calling
        self.get_activations.
    """
    prob_array = self.predict_proba(
        features=features, activations_dict=activations_dict)
    y_pred = np.argmax(prob_array, axis=-1)

    return y_pred

  def _distances(self, test_embeddings):
    """Gets list of Euclidean distances from training_embeddings.

    Args:
      test_embeddings: a dict of embeddings with keys in self.activation_names.

    Returns:
      A list of len(self.activation_names) numpy arrays of shape
      (n_test, n_train).
    """
    distances = []
    for activation_name in self.activation_names:
      test_emb = test_embeddings[activation_name]
      train_emb = self.training_embeddings[activation_name]

      # dis has shape (n_test, n_train)
      dis = cdist(test_emb, train_emb, 'euclidean')
      distances.append(dis)

    return distances

  def query(self, features=None, activations_dict=None, weights=None, k=10):
    """Queries the training embeddings for indices of nearest examples.

    Args:
      features: a batch of input data that the baseline model can process.
      activations_dict: Python dict of cached activations created by calling
        self.get_activations.
      weights: List of relative weights to use for the activations.
      k: the number of indices to return.

    Returns:
      `k` indices from the training embeddings.
    """
    if weights is None:
      weights = [1.0] * len(self.activation_names)

    # a list of len(activation_names) numpy arrays of shape
    # (n_test, embedding_dim)
    test_embeddings = self.clustering_extract_features(
        features=features, activations_dict=activations_dict)

    # make this into a dict to use self._distances
    test_embeddings = dict(zip(self.activation_names, test_embeddings))

    distances = self._distances(test_embeddings)

    # weighted_distances has shape (n_test, n_train)
    weighted_distances = np.sum(
        [w * dis for w, dis in zip(weights, distances)],
        axis=0)

    ind = np.argsort(weighted_distances, axis=-1)

    return ind[:, :k]

  def concept_indices(self, k=10):
    """Queries the training embeddings for indices nearest cluster centroids.

    Args:
      k: the number of nearest embeddings to return.

    Returns:
      A list of len(self.activation_names) arrays of training_embedding indices
      of shape (n_clusters, k).
    """
    # list of len(activation_names) numpy arrays of shape
    # (n_clusters, embedding_dim)
    centroids_list = self.get_centroids_list()

    # make this into a dict to use self._distances
    test_embeddings = dict(zip(self.activation_names, centroids_list))

    distances = self._distances(test_embeddings)

    # for each activation, get the top k nearest training example indices.
    result = []
    for dis in distances:
      ind = np.argsort(dis, axis=-1)
      result.append(ind[:, :k])

    return result

