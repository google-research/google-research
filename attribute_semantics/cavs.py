# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Cav class that stores Cav classifiers with easy weight vector access."""

import itertools
import math
import os
import random

import numpy as np
from six.moves import cPickle
import sklearn.linear_model
import sklearn.metrics
from tensorflow.io import gfile


class Cav:
  """Utility class for storing models and their properties."""

  def __init__(self, model):
    self.unnormalized_weight = model.coef_
    self.model_is_linear = True
    self.model = model

    self.normalized_weight = self.unnormalized_weight / np.linalg.norm(
        self.unnormalized_weight
    )


def gamma_correlation(
    num_aggrements,
    num_disagreements,
    num_strong_agreements,
    num_strong_disagreements,
):
  return (
      num_aggrements
      - num_disagreements
      + 2.0 * (num_strong_agreements - num_strong_disagreements)
  ) / (
      num_aggrements
      + num_disagreements
      + 2.0 * (num_strong_agreements + num_strong_disagreements)
  )


def evaluate_subjective_cavs(cavs, preferences, attribute, title_to_embs):
  """."""
  agreements = 0.0
  disagreements = 0.0
  strong_agreements = 0.0
  strong_disagreements = 0.0
  num_users = np.max(list(preferences.keys()))
  user_senses = np.zeros(num_users)
  for u in range(num_users):
    prefs = preferences[u + 1][attribute]
    num_agreements = 0.0
    num_strong_agreements = 0.0
    num_disagreements = 0.0
    num_strong_disagreements = 0.0
    best_gamma = -2.0
    for s, cav_model in enumerate(cavs):
      if cav_model is None:
        continue
      n_prefs = 0.0
      n_agreements = 0.0
      n_strong_agreements = 0.0
      n_disagreements = 0.0
      n_strong_disagreements = 0.0
      for pref in prefs:
        # Did not train embeddings for those movies.
        if pref.larger_item not in title_to_embs:
          print(pref.larger_item)
          continue
        if pref.smaller_item not in title_to_embs:
          print(pref.smaller_item)
          continue
        if pref.preference_strength == 0:
          # Skip pairs with similar degree for now
          continue
        n_prefs += 1.0
        larger_item = title_to_embs[pref.larger_item]
        smaller_item = title_to_embs[pref.smaller_item]
        cav = cav_model.normalized_weight.flatten()
        # if np.dot(larger_item, cav) < np.dot(smaller_item, cav):
        if np.dot(larger_item, cav) / np.linalg.norm(larger_item) < np.dot(
            smaller_item, cav
        ) / np.linalg.norm(smaller_item):
          if pref.preference_strength == 1:
            n_disagreements += 1.0
          if pref.preference_strength == 2:
            n_strong_disagreements += 1.0
        else:
          if pref.preference_strength == 1:
            n_agreements += 1.0
          if pref.preference_strength == 2:
            n_strong_agreements += 1.0
      if n_prefs == 0:
        continue
      gamma = gamma_correlation(
          n_agreements,
          n_disagreements,
          n_strong_agreements,
          n_strong_disagreements,
      )
      if gamma > best_gamma:
        user_senses[u] = s
        num_agreements = n_agreements
        num_disagreements = n_disagreements
        num_strong_agreements = n_strong_agreements
        num_strong_disagreements = n_strong_disagreements
        best_gamma = gamma
    agreements += num_agreements
    strong_agreements += num_strong_agreements
    disagreements += num_disagreements
    strong_disagreements += num_strong_disagreements

  return user_senses, gamma_correlation(
      agreements, disagreements, strong_agreements, strong_disagreements
  )


def train_single_subjective_cav(
    preferences,
    title_item_emb_dict_map,
    concept,
    do_ndcg_correction,
    max_num_senses=10,
    min_num_senses=1,
    num_em_iterations=20,
):
  """Trains cavs for subjective concepts."""
  num_users = np.max(list(preferences.keys()))
  num_users_tagged = 0
  for u in range(num_users):
    if preferences[u + 1][concept]:
      num_users_tagged += 1
  print(f'{concept} used by {num_users_tagged} users.')
  # Hardly require more than 5 senses for MovieLens tags.
  max_num_senses = min(max_num_senses, num_users_tagged)
  prev_gamma = -2.0
  result = None
  print(f'num_senses=0, accuracy={prev_gamma}')
  for num_senses in range(min_num_senses, max_num_senses + 1):
    senses = np.random.choice(num_senses, num_users)
    best_gamma = -2.0
    cavs = None
    for i in range(num_em_iterations):
      cavs = []
      # M-step
      for s in range(num_senses):
        preferences_attribute = set()
        for u in range(num_users):
          if senses[u] == s:
            preferences_attribute.update(preferences[u + 1][concept])
        if preferences_attribute:
          cav, _ = train_single_cav_ranking(
              preferences_attribute, title_item_emb_dict_map, do_ndcg_correction
          )
        else:
          # Generate a random CAV for degenerating cases.
          model = sklearn.linear_model.LogisticRegression(fit_intercept=False)
          model.fit(
              np.random.randn(
                  2, list(title_item_emb_dict_map.values())[0].shape[-1]
              ),
              np.array([0, 1]),
          )
          cav = Cav(model)
        cavs.append(cav)
      prev_sense = senses
      # E-step
      user_senses, gamma = evaluate_subjective_cavs(
          cavs, preferences, concept, title_item_emb_dict_map
      )
      if gamma > best_gamma:
        best_gamma = gamma
        senses = user_senses
        diff = sklearn.metrics.accuracy_score(prev_sense, senses)
      else:
        break
      print(i, diff, best_gamma)
      if diff >= 1 and i > 1:
        # Early termination.
        break
    print(f'num_senses={num_senses}, gamma={best_gamma}')
    if best_gamma >= 1.01 * prev_gamma:
      prev_gamma = best_gamma
      result = cavs
    else:
      break
  if len(result) == 1:
    result = result[0]
  return result, prev_gamma


NUM_FOLDS = 5


def ranking_training_data_for_softattributes(
    preferences_attribute,
    title_item_emb_dict_map,
    cv,
):
  """."""
  all_agg_examples = []
  all_agg_labels = []
  all_agg_weights = []
  for pref in preferences_attribute:
    if (
        pref.larger_item not in title_item_emb_dict_map
        or pref.smaller_item not in title_item_emb_dict_map
    ):
      continue
    if pref.preference_strength == 0:
      continue
    if pref.rater_id % NUM_FOLDS == cv:
      continue
    larger_item_emb = title_item_emb_dict_map[pref.larger_item]
    smaller_item_emb = title_item_emb_dict_map[pref.smaller_item]
    all_agg_examples.append(larger_item_emb - smaller_item_emb)
    all_agg_labels.append(1)
    all_agg_weights.append(pref.preference_strength)
    all_agg_examples.append(smaller_item_emb - larger_item_emb)
    all_agg_labels.append(-1)
    all_agg_weights.append(pref.preference_strength)

  # the ideal dcg of a 2-tuple list
  # with rel_score=1 in pos 1 and and rel_score=0. in pos 2
  max_dcg = (2.0**1.0 - 1.0) / np.log2(1.0 + 1.0) + (
      2.0**0.0 - 1.0
  ) / np.log2(1.0 + 2.0)
  # See reference for lambda-rank:
  # https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1e34e05e5e4bf2d12f41eb9ff29ac3da9fdb4de3.pdf

  delta_ndcg_weight = {
      'LESS': np.abs(
          1.0 / np.log2(1.0 + 1.0) - 1.0 / np.log2(1.0 + 2.0)
      ) * np.abs((2.0**0.0 - 1.0) / max_dcg - (2.0**1.0 - 1.0) / max_dcg),
      'MORE': np.abs(
          1.0 / np.log2(1.0 + 1.0) - 1.0 / np.log2(1.0 + 2.0)
      ) * np.abs((2.0**1.0 - 1.0) / max_dcg - (2.0**0.0 - 1.0) / max_dcg),
  }
  delta_ndcg_weight['MUCH MORE'] = delta_ndcg_weight['MORE']

  if len(all_agg_examples) > 0:  # pylint: disable=g-explicit-length-test
    return (
        np.stack(all_agg_examples, axis=0),
        np.stack(all_agg_labels, axis=0),
        np.stack(all_agg_weights, axis=0),
        delta_ndcg_weight,
    )
  else:
    return None, None, None, None


def train_single_cav_ranking(
    preferences_attribute,
    title_item_emb_dict_map,
    do_ndcg_correction=False,
    cv=-1,
):
  """."""
  (data, labels, weights, delta_ndcg_weight) = (
      ranking_training_data_for_softattributes(
          preferences_attribute,
          title_item_emb_dict_map,
          cv,
      )
  )
  if data is None:
    return None, None

  if not do_ndcg_correction:
    if np.count_nonzero(labels) >= 5:
      model = sklearn.linear_model.LogisticRegressionCV(
          max_iter=5000,
          fit_intercept=True,
          tol=1e-4,
      )
    else:
      model = sklearn.linear_model.LogisticRegression(
          max_iter=5000,
          fit_intercept=True,
          tol=1e-4,
      )
  else:
    if np.count_nonzero(labels) >= 5:
      model = sklearn.linear_model.LogisticRegressionCV(
          max_iter=5000,
          fit_intercept=True,
          tol=1e-4,
          class_weight={
              -1: delta_ndcg_weight['LESS'],
              1: delta_ndcg_weight['MORE'],
          },
      )
    else:
      model = sklearn.linear_model.LogisticRegression(
          max_iter=5000,
          fit_intercept=True,
          tol=1e-4,
          class_weight={
              -1: delta_ndcg_weight['LESS'],
              1: delta_ndcg_weight['MORE'],
          },
      )

  model.fit(data, labels, weights)
  log_loss = sklearn.metrics.log_loss(
      labels, model.predict_proba(data), normalize=False
  )

  return Cav(model), log_loss


def train_subjective_cavs_softattributes(
    all_attributes,
    preferences,
    title_item_emb_dict_map,
    do_ndcg_correction,
    max_num_senses,
    min_num_senses,
):
  """."""
  cav_dict = dict()
  for itr, concept in enumerate(all_attributes):
    cav_dict[concept], gamma = train_single_subjective_cav(
        preferences,
        title_item_emb_dict_map,
        concept,
        do_ndcg_correction=do_ndcg_correction,
        max_num_senses=max_num_senses,
        min_num_senses=min_num_senses,
    )
    print('No. {}. Training cavs for {}: {}'.format(itr, concept, gamma))
  return cav_dict


def training_data_for_concept(
    item_embs,
    concept,
    df,
    num_negative_multiplier=1,
    use_clean=False,
    mlp_model_item=None,
):
  """Generate positive and negative examples for concept from item-tag pairs in df.

  Args:
    item_embs (np.array): Vector embeddings of items, shape (n_items,
      embedding_dim).
    concept (string): tag to generate training data for.
    df (pandas df): dataframe with 'item_index' and 'tag' columns. 'item_index'
      in df corresponds to row in item_embs matrix.
    num_negative_multiplier (int): Factor by which the set of negative examples
      is larger than the set of positive examples.
    use_clean (bool): Flag to use clean data generation. Compared to non-clean,
      this option only takes negative samples from same users that gave positive
      concepts, potentially exlcudes negative example, to hopefully generate
      cleaner data.
    mlp_model_item:

  Returns:
    data (np.array): Matrix of positive and negative examples, shape
    (n_examples, embedding_dim).
    labels (np.array): Labelings for examples.
  """

  pos_examples = set(df.query('tag == @concept')['item_index'])
  num_examples = len(pos_examples)
  if use_clean:
    rel_users = df.query('tag == @concept')['user_index']
    all_tagged_by_rel_users = set(
        df.query('user_index in @rel_users')['item_index']
    )
    neg_examples = random.sample(
        list(all_tagged_by_rel_users - pos_examples),
        min(
            num_negative_multiplier * num_examples,
            len(all_tagged_by_rel_users - pos_examples),
        ),
    )
  else:
    all_tagged = set(df['item_index'])
    neg_examples = random.sample(
        list(all_tagged - pos_examples),
        min(
            len(all_tagged - pos_examples),
            num_negative_multiplier * num_examples,
        ),
    )

  item_index = np.array(list(pos_examples) + list(neg_examples))
  if mlp_model_item is not None:
    outp = mlp_model_item(item_index)
    if isinstance(outp, list):
      data = np.hstack([ele.numpy() for ele in outp])
    else:
      data = outp.numpy()
  else:
    data = item_embs[list(pos_examples) + list(neg_examples)]
  labels = np.zeros(data.shape[0])
  labels[: len(pos_examples)] = 1

  return (data, labels)


def ranking_training_data_for_concept(
    item_embs,
    concept,
    df,
    use_clean=False,
    mlp_model_item=None,
    max_data_limit=500,
):
  """Generate ranking examples for concept from item-tag pairs in df.

  Examples are generated by calculating differences between positive and
  negative examples for the concept.

  Args:
    item_embs (np.array): Vector embeddings of items, shape (n_items,
      embedding_dim).
    concept (string): tag to generate training data for.
    df (pandas df): dataframe with 'item_index' and 'tag' columns. 'item_index'
      in df corresponds to row in item_embs matrix.
    use_clean (bool): Flag to use clean data generation
    mlp_model_item:
    max_data_limit (int): Maximum data for pos/neg samples (before
      cross-product).

  Returns:
    agg_examples (np.array): Matrix of positive and negative examples, shape
    (n_examples, embedding_dim).
    agg_labels (np.array): Labelings for examples.
    delta_ndcg_weight (dict): weight for lambda_rank loss.
  """
  users = set(df['user_index'])
  all_agg_examples = []
  all_agg_labels = []
  for user in users:
    (data, labels) = training_data_for_concept(
        item_embs,
        concept,
        df.query('user_index == @user'),
        num_negative_multiplier=1,
        use_clean=use_clean,
        mlp_model_item=mlp_model_item,
    )

    pos_examples = data[labels == 1]

    if len(pos_examples) > max_data_limit:
      pos_examples = pos_examples[
          np.random.choice(
              pos_examples.shape[0], max_data_limit, replace=False
          ),
          :,
      ]
    neg_examples = data[labels == 0]
    if len(neg_examples) > max_data_limit:
      neg_examples = neg_examples[
          np.random.choice(
              neg_examples.shape[0], max_data_limit, replace=False
          ),
          :,
      ]
    product = itertools.product(pos_examples, neg_examples)

    agg_examples = np.array([p[0] - p[1] for p in product])
    # We need at least 5 examples in each class for 5-fold cross validation.
    if agg_examples.shape[0] < 5:
      agg_examples = np.tile(agg_examples, (2, 1))
    agg_labels = np.ones((agg_examples.shape[0], 1))

    if agg_examples.size > 0 and agg_labels.size > 0:
      # Make half of the examples have +1 labels and the other half -1 labels
      half = math.ceil(agg_examples.shape[0] / 2)
      agg_examples[:half, :] *= -1
      # the second half corresponds to the 0-label in binary classifier
      agg_labels[:half, :] *= 0

      agg_labels = agg_labels.ravel()
      all_agg_examples.append(agg_examples)
      all_agg_labels.append(agg_labels)

  # the ideal dcg of a 2-tuple list
  # with rel_score=1 in pos 1 and and rel_score=0. in pos 2
  max_dcg = (2.0**1.0 - 1.0) / np.log2(1.0 + 1.0) + (
      2.0**0.0 - 1.0
  ) / np.log2(1.0 + 2.0)
  # See reference for lambda-rank:
  # https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1e34e05e5e4bf2d12f41eb9ff29ac3da9fdb4de3.pdf
  delta_ndcg_weight = {
      'i<j': np.abs(
          1.0 / np.log2(1.0 + 1.0) - 1.0 / np.log2(1.0 + 2.0)
      ) * np.abs((2.0**0.0 - 1.0) / max_dcg - (2.0**1.0 - 1.0) / max_dcg),
      'i>j': np.abs(
          1.0 / np.log2(1.0 + 1.0) - 1.0 / np.log2(1.0 + 2.0)
      ) * np.abs((2.0**1.0 - 1.0) / max_dcg - (2.0**0.0 - 1.0) / max_dcg),
  }

  if all_agg_labels:
    return (
        np.concatenate(all_agg_examples),
        np.concatenate(all_agg_labels),
    ), delta_ndcg_weight
  return ([], []), delta_ndcg_weight


def train_subjective_cavs(
    item_embs,
    cav_mode,
    df,
    concepts,
    mlp_model_item=None,
    num_em_iterations=20,
    num_negative_multiplier=3,
    num_sense_hint={},
):
  """Trains cavs for subjective concepts.

  Cav training uses item and, potentially, user embeddings and user-item-tag
  data from df.

  Args:
    item_embs (np.array): Vector embeddings of items, shape (n_items,
      embedding_dim).
    cav_mode (string): Determine data generation and classifier training
      process.
    df (pandas df): Dataframe to use for training data generation.
    concepts (list): List of strings representing concepts for which to train
      cavs.
    mlp_model_item (tf.Keras.Model): Item model.
    num_em_iterations: Number of EM iterations.
    num_negative_multiplier: Weight on negatives.
    num_sense_hint: A dictionary storing number of senses for each concept.

  Returns:
    cav_dict (dict): Dictionary with keys concepts, and a list of Cav objects as
      values for all senses of that concept.
  """
  assert not df.empty
  basic_cav_mode = cav_mode[3:]  # Remove the prefix "em_".
  cav_dict = dict()
  for concept in concepts:
    users_tagged = list(set(df.query('tag == @concept')['user_index']))
    random.shuffle(users_tagged)
    num_users_tagged = len(users_tagged)
    print(f'{concept} used by {num_users_tagged} users.')
    # Hardly require more than 5 senses for MovieLens tags.
    max_senses = min(5, num_users_tagged)
    max_num_senses = num_sense_hint.get(concept, max_senses)
    cav_dict[concept], _ = train_single_cav(
        item_embs,
        basic_cav_mode,
        df.copy(),
        concept,
        mlp_model_item,
        num_negative_multiplier,
    )
    _, prev_accuracy = get_best_sense(
        concept,
        cav_dict[concept],
        cav_mode,
        df,
        item_embs,
        mlp_model_item,
        return_accuracy=True,
    )
    print(f'num_senses=1, accuracy={prev_accuracy}')
    for num_senses in range(2, max_num_senses + 1):
      df_tag = df.copy()
      # Initialize the tag sense for each user.
      df_tag.loc[:, 'sense'] = df_tag['user_index'].to_numpy() % num_senses
      # Randomly assign sense to users tagged with the concept.
      for i, user in enumerate(users_tagged):
        df_tag.loc[df_tag['user_index'] == user, 'sense'] = i % num_senses
      accuracy = 0.0
      cavs = None
      for i in range(num_em_iterations):
        cavs = []
        # M-step
        for s in range(num_senses):
          df_sense = df_tag.query('sense == @s')
          if (
              df_sense.empty
              or df_sense.query('tag == @concept')['item_index'].empty
          ):
            # Generate a random CAV for degenerating cases.
            model = sklearn.linear_model.LogisticRegression(fit_intercept=False)
            model.fit(np.random.randn(2, item_embs.shape[-1]), np.array([0, 1]))
            cav = Cav(model)
          else:
            cav, _ = train_single_cav(
                item_embs,
                basic_cav_mode,
                df_sense,
                concept,
                mlp_model_item,
                num_negative_multiplier,
            )
          cavs.append(cav)
        prev_sense = df_tag['sense'].to_numpy()
        # E-step
        user_sense, acc = get_best_sense(
            concept,
            cavs,
            cav_mode,
            df_tag,
            item_embs,
            mlp_model_item,
            return_accuracy=True,
        )
        if acc > accuracy:
          accuracy = acc
          df_tag.loc[:, 'sense'] = user_sense[df_tag['user_index'].to_numpy()]
          diff = sklearn.metrics.accuracy_score(
              prev_sense, df_tag.sense.to_numpy()
          )
        else:
          break
        print(i, diff, accuracy)
        if diff >= 1 and i > 1:
          # Early termination.
          break
      del df_tag
      print(f'num_senses={num_senses}, accuracy={accuracy}')
      if accuracy >= 1.01 * prev_accuracy:
        prev_accuracy = accuracy
        cav_dict[concept] = cavs
      else:
        break
  return cav_dict


def train_single_cav(
    item_embs,
    cav_mode,
    df,
    concept,
    mlp_model_item=None,
    num_negative_multiplier=3,
):
  """Trains cav for one concept.

  Cav training uses item and, potentially, user embeddings and user-item-tag
  data from df.

  Args:
    item_embs (np.array): Vector embeddings of items, shape (n_items,
      embedding_dim).
    cav_mode (string): Determine data generation and classifier training
      process.
    df (pandas df): Dataframe to use for training data generation.
    concept (string): List of strings representing concepts for which to train
      cavs.
    mlp_model_item:
    num_negative_multiplier: Negative samples multiplier for logistic regression
      methods

  Returns:
    cav (Cav): a Cav object.
  """
  assert not df.empty
  cav_mode_list = cav_mode.split('_')
  if 'embedNN' in cav_mode:
    end_ind = cav_mode_list.index('embedNN')
    cav_mode = '_'.join(cav_mode_list[:end_ind])
  else:
    mlp_model_item = None
  print(concept, cav_mode)
  if cav_mode in ['binary', 'binary_clean']:
    # Train binary CAVs

    if cav_mode in ['binary']:
      (data, labels) = training_data_for_concept(
          item_embs,
          concept,
          df,
          num_negative_multiplier=num_negative_multiplier,
          mlp_model_item=mlp_model_item,
      )
    else:
      # cav_mode == 'binary_clean'
      raise ValueError(f'cav_mode {cav_mode} is not implemented.')

    if np.count_nonzero(labels) >= 5:
      model = sklearn.linear_model.LogisticRegressionCV(
          max_iter=5000, fit_intercept=True, class_weight='balanced'
      )
    else:
      model = sklearn.linear_model.LogisticRegression(
          max_iter=5000, fit_intercept=True, class_weight='balanced'
      )

  elif cav_mode in [
      'ranking_simple',
      'ranking_lambda',
      'ranking_simple_clean',
      'ranking_lambda_clean',
  ]:
    # Train simple/lambda ranking CAVs for binary list
    # Only works for linear model!
    # Reformulate linear ranking CAV as binary classification.

    # current the nonlinear CAV ranking model is not supported
    if cav_mode in ['ranking_simple', 'ranking_lambda']:
      (data, labels), delta_ndcg_weight = ranking_training_data_for_concept(
          item_embs, concept, df, mlp_model_item=mlp_model_item
      )
    else:
      # cav_mode in ['ranking_simple_clean', 'ranking_lambda_clean']
      raise ValueError(f'cav_mode {cav_mode} is not implemented.')
    if cav_mode in ['ranking_lambda', 'ranking_lambda_clean']:
      if np.count_nonzero(labels) >= 5:
        model = sklearn.linear_model.LogisticRegressionCV(
            max_iter=5000,
            fit_intercept=False,
            tol=1e-4,
            class_weight={
                0: delta_ndcg_weight['i<j'],
                1: delta_ndcg_weight['i>j'],
            },
        )
      else:
        model = sklearn.linear_model.LogisticRegression(
            max_iter=5000,
            fit_intercept=False,
            tol=1e-4,
            class_weight={
                0: delta_ndcg_weight['i<j'],
                1: delta_ndcg_weight['i>j'],
            },
        )
    else:
      # cav_mode in ['ranking_simple', 'ranking_simple_clean']
      if np.count_nonzero(labels) >= 5:
        model = sklearn.linear_model.LogisticRegressionCV(
            max_iter=5000, fit_intercept=False, class_weight='balanced'
        )
      else:
        model = sklearn.linear_model.LogisticRegression(
            max_iter=5000, fit_intercept=False, class_weight='balanced'
        )

  else:
    raise ValueError(f'cav_mode {cav_mode} is not implemented.')

  assert data.size > 0
  assert labels.size > 0

  model.fit(data, labels)
  log_loss = sklearn.metrics.log_loss(
      labels, model.predict_proba(data), normalize=False
  )
  # print(model.n_iter_)
  print(model.score(data, labels))
  # just a cav class
  return Cav(model), log_loss


def train_cavs(
    item_embs,
    cav_mode,
    df,
    concepts,
    mlp_model_item=None,
    num_negative_multiplier=3,
):
  """Trains cavs for concepts.

  Cav training uses item and, potentially, user embeddings and user-item-tag
  data from df.

  Args:
    item_embs (np.array): Vector embeddings of items, shape (n_items,
      embedding_dim).
    cav_mode (string): Determine data generation and classifier training
      process.
    df (pandas df): Dataframe to use for training data generation.
    concepts (list): List of strings representing concepts for which to train
      cavs.
    mlp_model_item:
    num_negative_multiplier: Negative samples multiplier for logistic regression
      methods

  Returns:
    cav_dict (dict): Dictionary with keys concepts, and Cav objects as values.
  """
  cav_dict = dict()
  for concept in concepts:
    cav_dict[concept], log_loss = train_single_cav(
        item_embs,
        cav_mode,
        df,
        concept,
        mlp_model_item,
        num_negative_multiplier=num_negative_multiplier,
    )
    print(log_loss)
  return cav_dict


def get_best_sense(
    concept,
    cav,
    cav_mode,
    df,
    item_embs,
    mlp_model_item=None,
    return_accuracy=False,
):
  """Evaluates each user and picks the best sense.

  Args:
    concept (string): concept to be evaluated.
    cav (Cav): Cav of concept.
    cav_mode (string): Determine data generation process
    df (pandas df): Dataframe to use for evaluation data generation.
    item_embs (np.array): Vector embeddings of items, shape (n_items,
      embedding_dim).
    mlp_model_item:
    return_accuracy (bool):

  Returns:
    user_sense (np.array): A NumPy array storing the best sense
      for each user index.
  """
  assert not df.empty
  assert 'em_' in cav_mode
  basic_cav_mode = cav_mode[3:]  # Remove the prefix "em_".
  all_users = set(df['user_index'])
  max_user_id = np.amax(df['user_index'].to_numpy())
  if not isinstance(cav, list):
    if return_accuracy:
      accuracy, num_examples = get_cav_accuracy_simple(
          concept, cav, basic_cav_mode, df, item_embs, mlp_model_item
      )
      return np.zeros(max_user_id + 1), accuracy / num_examples
    return np.zeros(max_user_id + 1)
  num_senses = len(cav)
  assert num_senses > 1
  user_sense = -np.ones(max_user_id + 1)
  accuracy = 0.0
  num_examples = 0.0
  weight = 0.0
  for u in all_users:
    accs_s = np.zeros(num_senses)
    sub_df = df.query('user_index == @u')
    for s in range(num_senses):
      accs_s[s], weight = get_cav_accuracy_simple(
          concept, cav[s], basic_cav_mode, sub_df, item_embs, mlp_model_item
      )
    user_sense[u] = np.argmax(accs_s)
    accuracy += np.max(accs_s)
    num_examples += weight
  if return_accuracy:
    return user_sense, accuracy / num_examples
  return user_sense


def get_cav_accuracy_simple(
    concept, cav, cav_mode, df, item_embs, mlp_model_item=None
):
  """Evaluates one concept's cav on data generated according to cav_mode.

  For linear model, use sklearn functions model.score for accuracy and
  metrics.precision_recall_fscore_support for precision and recall.

  For keras model, use XXX for accuracy and YYY for precision and recall.

  Args:
    concept (string): concept to be evaluated.
    cav (Cav): Cav of concept.
    cav_mode (string): Determine data generation process
    df (pandas df): Dataframe to use for evaluation data generation.
    item_embs (np.array): Vector embeddings of items, shape (n_items,
      embedding_dim).
    mlp_model_item:

  Returns:
    acc: Accuracy of concept with cav.
  """
  cav_mode_list = cav_mode.split('_')
  if 'embedNN' in cav_mode:
    end_ind = cav_mode_list.index('embedNN')
    cav_mode = '_'.join(cav_mode_list[:end_ind])
  else:
    mlp_model_item = None

  if cav_mode in ['binary', 'binary_clean']:
    if cav_mode in ['binary']:
      (data, labels) = training_data_for_concept(
          item_embs, concept, df, mlp_model_item=mlp_model_item
      )
    else:
      # cav_mode == 'binary_clean'
      raise ValueError(f'cav_mode {cav_mode} is not implemented.')
  elif cav_mode in [
      'ranking_simple',
      'ranking_lambda',
      'ranking_simple_clean',
      'ranking_lambda_clean',
  ]:
    if cav_mode in ['ranking_simple', 'ranking_lambda']:
      (data, labels), _ = ranking_training_data_for_concept(
          item_embs, concept, df, mlp_model_item=mlp_model_item
      )
    else:
      # cav_mode in ['ranking_simple_clean', 'ranking_lambda_clean']
      raise ValueError(f'cav_mode {cav_mode} is not implemented.')
  else:
    raise ValueError(f'cav_mode {cav_mode} is not implemented.')

  if len(data) >= 1:
    return len(data) * cav.model.score(data, labels), len(data)
  return 0.0, 0.0


def save_model_pickle(model, savedir, file_name):
  if not gfile.isdir(savedir):
    gfile.makedirs(savedir)
  file_name = file_name.replace('.', '_')
  path = os.path.join(savedir, '{}.pkl'.format(file_name))
  print('Save model %s to %s' % (file_name, path))
  with gfile.GFile(path, 'wb') as f:
    cPickle.dump(model, f)
