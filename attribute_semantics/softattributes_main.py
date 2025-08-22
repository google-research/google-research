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

r"""Runner file for using WALS movie embeddings and softattributes dataset."""

import collections
import os

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from tensorflow.io import gfile

from attribute_semantics import cavs as cav_utils
from attribute_semantics import soft_attribute

_SOFTATTRIBUTES_DATA = flags.DEFINE_string(
    'softattributes_data', '', 'Path to softattributes data.'
)
_SAVE_DIR = flags.DEFINE_string(
    'save_dir',
    '',
    'Save directory.',
)
_CAV_MODES = flags.DEFINE_list(
    'cav_modes',
    [
        'em_ranking_simple',
        'em_ranking_lambda',
    ],
    'List of mode options for CAV training.',
)
_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    '',
    'Path to WALS model embeddings used to train CAVs',
)
_CROSS_VALIDATION = flags.DEFINE_integer(
    'cross_validation', 1, 'Number of folds used in cross validation.'
)
_MAX_NUM_SENSES = flags.DEFINE_integer(
    'max_num_senses', 10, 'Maximum number of senses.'
)
_MIN_NUM_SENSES = flags.DEFINE_integer(
    'min_num_senses', 1, 'Minimum number of senses.'
)

NUM_RATERS = 100


def evaluate_all_subjective_cavs(cav_dict, preferences, title_to_embs):
  """Evaluates subjective CAVs for all attributes.

  Args:
    cav_dict:
    preferences:
    title_to_embs:

  Returns:
    a dictionary keyed by attributes to hold results
  """
  agreements = collections.defaultdict(float)
  disagreements = collections.defaultdict(float)
  strong_agreements = collections.defaultdict(float)
  strong_disagreements = collections.defaultdict(float)
  total_prefs = collections.defaultdict(float)
  num_users = np.max(list(preferences.keys()))
  for u in range(num_users):
    for attribute, prefs in preferences[u + 1].items():
      # Did not train CAV for those attributes.
      if attribute not in cav_dict:
        print(f'{attribute} not found.')
        continue
      if not prefs:
        print('empty prefs:', attribute, u + 1)
        continue
      num_prefs = 0.0
      num_agreements = 0.0
      num_strong_agreements = 0.0
      num_disagreements = 0.0
      num_strong_disagreements = 0.0
      best_gamma = -2.0
      cavs = [cav_dict[attribute]]
      if isinstance(cav_dict[attribute], list):
        cavs = cav_dict[attribute]
      for cav_model in cavs:
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
        gamma = cav_utils.gamma_correlation(
            n_agreements,
            n_disagreements,
            n_strong_agreements,
            n_strong_disagreements,
        )
        if gamma > best_gamma:
          num_prefs = n_prefs
          num_agreements = n_agreements
          num_disagreements = n_disagreements
          num_strong_agreements = n_strong_agreements
          num_strong_disagreements = n_strong_disagreements
          best_gamma = gamma
      total_prefs[attribute] += num_prefs
      agreements[attribute] += num_agreements
      strong_agreements[attribute] += num_strong_agreements
      disagreements[attribute] += num_disagreements
      strong_disagreements[attribute] += num_strong_disagreements

  cav_results = {}
  for a in total_prefs:
    cav_results[a] = np.array([
        total_prefs[a],
        agreements[a],
        strong_agreements[a],
        disagreements[a],
        strong_disagreements[a],
        len(cav_dict[a]) if isinstance(cav_dict[a], list) else 1,
    ])
  return cav_results


def results_to_dataframe(
    all_cav_results, cav_mode
):
  """...

  Args:
    all_cav_results:
    cav_mode:

  Returns:

  """
  res = []
  for a in all_cav_results:
    gamma_a = cav_utils.gamma_correlation(
        np.sum(all_cav_results[a][1]),
        np.sum(all_cav_results[a][3]),
        np.sum(all_cav_results[a][2]),
        np.sum(all_cav_results[a][4]),
    )
    res.append([a] + [gamma_a] + list(all_cav_results[a]))
  res_df = pd.DataFrame(
      res,
      columns=[
          'attribute',
          'gamma_corr',
          'num_prefs',
          'agree',
          'strong_agree',
          'disagree',
          'strong_disagree',
          'num_senses',
      ],
  )
  res_df['cav_mode'] = cav_mode
  res_df.num_senses = res_df.num_senses / float(_CROSS_VALIDATION.value)
  print(
      cav_utils.gamma_correlation(
          np.sum(res_df.agree),
          np.sum(res_df.disagree),
          np.sum(res_df.strong_agree),
          np.sum(res_df.strong_disagree),
      )
  )
  print(res_df)
  return res_df


def main(_):
  np.random.seed(0)

  all_judgments = soft_attribute.load_judgments(_SOFTATTRIBUTES_DATA.value)
  all_attributes = list(all_judgments.keys())
  preferences = {}
  for u in range(NUM_RATERS):
    preferences[u + 1] = collections.defaultdict(set)
  for attribute, judgments in all_judgments.items():
    for j in judgments:
      preferences[j.rater_id][attribute].update(
          soft_attribute.convert_to_pairwise_preferences(j)
      )

  with gfile.GFile(_MODEL_PATH.value, 'rb') as f:
    item_embs = np.load(f)
    title_item_emb_dict_map = {
        soft_attribute.make_title_presentable(key, False): val
        for key, val in zip(item_embs.f.titles, item_embs.f.embeddings)
    }

  respath = os.path.join(_SAVE_DIR.value, 'le_False')
  gfile.makedirs(os.path.join(respath, 'CAVs'))

  for cav_mode in _CAV_MODES.value:
    assert cav_mode in ('em_ranking_simple', 'em_ranking_lambda')
    print('Now training CAVs for {}'.format(cav_mode))
    if _CROSS_VALIDATION.value > 1:
      cv = _CROSS_VALIDATION.value
      all_cav_results_train = {}
      all_cav_results = {}
      # [num_prefs, agree, strong_agree, disagree, strong_disagree, num_senses]
      for a in all_attributes:
        all_cav_results_train[a] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        all_cav_results[a] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      for i in range(cv):
        print('Cross Validation: ', i)
        preferences_train = {}
        preferences_test = {}
        num_train = 0
        num_test = 0
        for u in range(NUM_RATERS):
          if u % cv != i:
            num_train += 1
            preferences_train[num_train] = preferences[u + 1]
          else:
            num_test += 1
            preferences_test[num_test] = preferences[u + 1]
        cav_dict = cav_utils.train_subjective_cavs_softattributes(
            all_attributes,
            preferences_train,
            title_item_emb_dict_map,
            do_ndcg_correction=(cav_mode == 'em_ranking_lambda'),
            max_num_senses=_MAX_NUM_SENSES.value,
            min_num_senses=_MIN_NUM_SENSES.value,
        )
        cav_utils.save_model_pickle(
            cav_dict, os.path.join(respath, 'CAVs' + str(i)), cav_mode
        )
        cav_results = evaluate_all_subjective_cavs(
            cav_dict, preferences_train, title_item_emb_dict_map
        )
        for a in all_attributes:
          all_cav_results_train[a] += cav_results[a]
        cav_results = evaluate_all_subjective_cavs(
            cav_dict, preferences_test, title_item_emb_dict_map
        )
        for a in all_attributes:
          all_cav_results[a] += cav_results[a]
      res_df = results_to_dataframe(all_cav_results_train, cav_mode)
      pd.DataFrame.to_csv(
          res_df, os.path.join(respath, cav_mode + '_train.csv')
      )
      res_df = results_to_dataframe(all_cav_results, cav_mode)
      pd.DataFrame.to_csv(res_df, os.path.join(respath, cav_mode + '.csv'))
    else:
      cav_dict = cav_utils.train_subjective_cavs_softattributes(
          all_attributes,
          preferences,
          title_item_emb_dict_map,
          do_ndcg_correction=(cav_mode == 'em_ranking_lambda'),
          max_num_senses=_MAX_NUM_SENSES.value,
          min_num_senses=_MIN_NUM_SENSES.value,
      )
      cav_results = evaluate_all_subjective_cavs(
          cav_dict, preferences, title_item_emb_dict_map
      )
      res_df = results_to_dataframe(cav_results, cav_mode)
      pd.DataFrame.to_csv(res_df, os.path.join(respath, cav_mode + '.csv'))

      # save cav model
      cav_utils.save_model_pickle(
          cav_dict, os.path.join(respath, 'CAVs'), cav_mode
      )


if __name__ == '__main__':
  app.run(main)
