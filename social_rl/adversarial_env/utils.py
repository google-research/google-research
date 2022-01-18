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

"""Utility functions, primarily used for finding the best hyperparameters.
"""
import datetime
import os

import numpy as np
import pandas as pd

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

HPARAM_COLUMNS = ['xm_adv_conv_filters', 'xm_adv_entropy_regularization',
                  'xm_adversary_population_size',
                  'xm_antagonist_population_size', 'xm_non_negative_regret',
                  'xm_percent_random_episodes', 'xm_protagonist_episode_length',
                  'xm_protagonist_population_size', 'xm_agents_use_regret',
                  'xm_combined_population', 'xm_flexible_protagonist',
                  'xm_block_budget_weight']


def save_best_work_units_csv(experiments, csv_path=None, metrics=None,
                             last_x_percent=.2, step_limit_1agent=None):
  """Collects XM work unit IDs corresponding to best hparams and saves to csv.

  Args:
    experiments: A list of Experiment objects.
    csv_path: Location where the resulting csv should be saved.
    metrics: Metrics used to select best hparams (e.g. reward).
    last_x_percent: Select hparams that led to the best performance over the
      last X% of the run.
    step_limit_1agent: Restrict dataframes to this many steps before selecting
      best hparams for last X%. This is the step limit if only 1 agent is being
      trained, so will need to be adjusted for units with multiple agents.
  Returns:
    A pandas dataframe with the best work units.
  """
  if metrics is None:
    metrics = ['SolvedPathLength', 'adversary_env_AdversaryReward']
  best_seeds_df = pd.DataFrame()

  for exp in experiments:
    if (not exp.hparam_sweep or exp.metrics_df is None
        or 'work_unit' not in exp.metrics_df.columns.values):
      continue

    print_header(exp, exp.metrics_df)

    if ('combined' in exp.name.lower() and
        'xm_combined_population' not in exp.metrics_df.columns.values):
      exp.metrics_df['xm_combined_population'] = True

    metrics_df = exp.metrics_df
    for metric in metrics:
      if metric not in exp.metrics_df.columns.values:
        continue

      print('\nLooking for highest', metric)
      best_setting = find_best_setting_for_metric(
          metrics_df, metric, run='eval', step_limit_1agent=step_limit_1agent,
          last_x_percent=last_x_percent, num_agents=exp.num_agents)
      setting_df = restrict_to_setting(metrics_df, best_setting)
      units = setting_df['work_unit'].unique().tolist()

      # Check for combined population and calculate number of agents in pop
      combined_population = False
      if 'xm_combined_population' in setting_df.columns.values:
        assert len(setting_df['xm_combined_population'].unique()) == 1
        combined_population = setting_df['xm_combined_population'].unique()[0]
      num_agents = calculate_num_agents_based_on_population(
          setting_df, exp.num_agents, combined_population)

      # Adjust step limit for number of agents
      if step_limit_1agent:
        step_limit = step_limit_1agent * num_agents
      else:
        step_limit = None

      scores = get_score_for_setting(
          metrics_df, exp.metrics, best_setting, step_limit=step_limit,
          last_x_percent=last_x_percent, run='eval')

      m_dict = {
          'exp_name': exp.name,
          'xm_id': exp.xm_id,
          'settings': best_setting,
          'best_seeds': [str(u) for u in units],
          'metric': metric + '_last20%',
          'score': scores[metric],
          'work_units_tested': len(metrics_df['work_unit'].unique()),
          'max_steps': metrics_df['step'].max()
      }
      best_seeds_df = best_seeds_df.append(m_dict, ignore_index=True)

      if metric == 'SolvedPathLength':
        single_best = metrics_df.loc[metrics_df[metric].idxmax()]
        search_params = get_search_params(metrics_df)
        settings = {}
        for s in search_params:
          settings[s] = single_best[s]

        m_dict = {
            'exp_name': exp.name,
            'xm_id': exp.xm_id,
            'settings': settings,
            'best_seeds': single_best['work_unit'],
            'metric': metric + '_best_ever',
            'score': single_best[metric],
            'work_units_tested': len(metrics_df['work_unit'].unique()),
            'max_steps': metrics_df['step'].max()
        }
        best_seeds_df = best_seeds_df.append(m_dict, ignore_index=True)

  if csv_path is not None:
    with tf.io.gfile.GFile(csv_path, 'wb') as f:
      best_seeds_df.to_csv(f)
    print('Saved best seeds csv to:', csv_path)

  return best_seeds_df


def combine_existing_transfer_data(transfer_dir, after_date=None,
                                   filter_n_trials=10.):
  """Combine all transfer files after a certain date, dropping duplicates."""
  files = tf.io.gfile.listdir(transfer_dir)

  # This will sort files, and trim any files pre-dating the after_date
  sorted_files = sort_files_by_date(files, after_date=after_date)

  df = pd.DataFrame()
  for transfer_file in reversed(sorted_files):
    transfer_df_path = os.path.join(transfer_dir, transfer_file)
    if tf.io.gfile.stat(transfer_df_path).length == 0:
      print('File', transfer_df_path, 'has length 0, skipping')
      continue

    with tf.gfile.GFile(transfer_df_path, 'rb') as f:
      file_df = pd.read_csv(f)
    print('\nLoaded file', transfer_file, 'of length', len(file_df))

    if file_df.empty:
      continue

    # Remove previous rows which used a different number of trials
    if filter_n_trials is not None:
      prev_len = len(file_df)
      file_df = file_df[file_df['n'] == filter_n_trials]
      if len(file_df) != prev_len:
        print('Removed', prev_len - len(file_df),
              'rows where n !=', filter_n_trials, '... New length is:',
              len(file_df))
      if file_df.empty:
        continue

    # Remove extra unnecessary index columns
    bad_cols = [c for c in file_df.columns.values if 'Unnamed' in c]
    file_df = file_df.drop(columns=bad_cols)

    if 'metric' not in file_df.columns.values:
      file_df['metric'] = ''

    print('\tExperiments/metrics found in this file:',
          get_unique_combos_in_df(file_df, ['name', 'metric']))

    key_names = ['name', 'xm_id', 'seed', 'env', 'checkpoint', 'agent_id', 'n',
                 'domain_rand_comparable_checkpoint', 'metric']

    # Merge in new rows
    deduped_file_df = drop_duplicates_but_alert(
        file_df, key_names, transfer_file)
    deduped_df = drop_duplicates_but_alert(
        df, key_names, 'main df')

    prev_len = len(deduped_df)
    df = pd.concat([deduped_df, deduped_file_df],
                   sort=True).reset_index(drop=True)
    df.drop_duplicates(subset=key_names, inplace=True, keep='first')
    print('Added', len(df) - prev_len, 'new rows to the main df. It now has',
          len(df), 'rows')

    if len(df) == prev_len:
      continue

    assert prev_len < len(df), 'Merging should not remove rows'

    # Analyze which rows were added by this file
    new_rows = df[prev_len:]
    print('\t', len(new_rows) / float(len(file_df)) * 100.,
          '% of the rows in file', transfer_file, 'were new.')
    print('\tNew rows involve these experiments/metrics:',
          get_unique_combos_in_df(new_rows, ['name', 'metric']))

  return df


def sort_files_by_date(files, after_date=None, check_str='transfer'):
  """Sorts files by date, assuming the date is the last part of the filename.

  Will discard files with a date before the after_date.
  Args:
    files: A list of string filenames with the date as the last part of the
      string before the extension.
    after_date: A date such that any file dating after this date should be kept.
    check_str: Each file must contain this string or it will be skipped.
  Returns:
    A list of filenames in sorted order and with dates that are too early
    discarded.
  """
  after_dt = None
  if after_date is not None:
    after_dt = datetime.datetime.strptime(after_date, '%d.%m.%Y.%H:%M:%S')

  trimmed_files = []
  datetimes = []
  for f in files:
    if f == 'transfer_results.csv':
      continue

    # Skip files not containing check_str
    if check_str not in f:
      continue

    end_idx = f.find('.csv')
    start_idx = end_idx - len('02.06.2020.07:58:30')
    date_str = f[start_idx:end_idx]
    dt = datetime.datetime.strptime(date_str, '%d.%m.%Y.%H:%M:%S')

    if after_dt is not None and dt < after_dt:
      # Ignore dates before the after_date
      continue

    datetimes.append(dt)
    trimmed_files.append(f)

  zipped_pairs = zip(datetimes, trimmed_files)
  sorted_files = [x for _, x in sorted(zipped_pairs)]
  return sorted_files


def drop_duplicates_but_alert(df, key_names, df_name):
  prev_len = len(df)
  deduped_df = df.drop_duplicates(key_names)
  if len(deduped_df) != prev_len:
    print('Dropped', prev_len - len(deduped_df), 'duplicates from', df_name)
    return deduped_df
  return deduped_df


def get_unique_combos_in_df(df, keys):
  for k in keys:
    df[k] = df[k].fillna('')
  return np.unique(['/'.join(k) for k in df[keys].values])


def calculate_num_agents_based_on_population(
    settings, exp_num_agents, combined_population=False, is_dict=False):
  """Calculate how much to adjust steps based on number of trained agents."""
  pop_sizes = {}

  # Get population sizes from a dictionary
  if is_dict:
    for pop_type in ['xm_protagonist_population_size',
                     'xm_antagonist_population_size',
                     'xm_adversary_population_size']:
      if pop_type in settings:
        pop_sizes[pop_type] = settings[pop_type]
      else:
        pop_sizes[pop_type] = 1

  #  Get population sizes from a dataframe
  else:
    for pop_type in ['xm_protagonist_population_size',
                     'xm_antagonist_population_size',
                     'xm_adversary_population_size']:
      if pop_type in settings.columns.values:
        assert len(settings[pop_type].unique()) == 1
        pop_sizes[pop_type] = settings[pop_type].unique()[0]
      else:
        pop_sizes[pop_type] = 1

  if combined_population:
    num_agents = pop_sizes['xm_protagonist_population_size']  + \
          pop_sizes['xm_adversary_population_size']
  elif exp_num_agents == 3:
    num_agents = pop_sizes['xm_protagonist_population_size'] + \
        pop_sizes['xm_antagonist_population_size'] + \
        pop_sizes['xm_adversary_population_size']
  elif exp_num_agents == 2:
    num_agents = pop_sizes['xm_protagonist_population_size'] + \
        pop_sizes['xm_adversary_population_size']
  else:
    num_agents = 1

  return num_agents


def print_header(exp, df, last_x_percent=.2):
  """Print information about a hyperparameter sweep experiment."""
  print('HPARAM SWEEP =', exp.name)
  print('Looking at last', last_x_percent*100, '% of data')
  print('Considering', df['run'].unique())
  print('Model has been trained for', df['step'].max(), 'steps')
  print(len(df['work_unit'].unique()), 'work units reporting in\n')


def get_search_params(df, hparam_columns=None):
  """Find all different hyperparameter combinations present in an XM exp df."""
  if hparam_columns is None:
    hparam_columns = HPARAM_COLUMNS
  search_hparams = [h for h in hparam_columns if h in df.columns.values]
  to_remove = []
  for h in search_hparams:
    if len(df[h].unique()) < 2:
      to_remove.append(h)
  return [h for h in search_hparams if h not in to_remove]


def restrict_to_setting(df, setting, run='eval'):
  """Restrict an experiment dataframe to one hyperparameter setting."""
  setting_df = df[df['run'] == run]
  for k, v in setting.items():
    if k in df.columns.values:
      setting_df = setting_df[setting_df[k] == v]
  return setting_df


def get_score_for_setting(df, metrics, setting, step_limit=None,
                          last_x_percent=.2, run='eval', verbose=True,
                          ignore_metrics=None):
  """Find the average score across several metrics for an hparam setting."""
  if ignore_metrics is None:
    ignore_metrics = ['NumEnvEpisodes', 'GoalX', 'GoalY']

  if verbose:
    print('Testing hparameter settings:')
    print(setting)
  setting_df = restrict_to_setting(df, setting, run)
  if verbose:
    print('There are', len(setting_df['work_unit'].unique()),
          'work units with these settings')
    print('\twhich are:', setting_df['work_unit'].unique())

  setting_df = setting_df.sort_values('step')

  if step_limit is not None:
    prev_len = len(setting_df)
    setting_df = setting_df[setting_df['step'] <= step_limit]
    if verbose:
      print('After restricting to step limit of', step_limit,
            'the dataframe went from', prev_len, 'rows to', len(setting_df))

  start_step = int(len(setting_df) * (1-last_x_percent))
  scores = {}
  for metric in metrics:
    if metric not in setting_df.columns.values or metric in ignore_metrics:
      continue
    scores[metric] = setting_df[metric][start_step:].mean()
    if verbose: print('Mean', metric, scores[metric])
  return scores


def find_best_settings(df, metrics, verbose=True, step_limit_1agent=None,
                       last_x_percent=.2, run='eval', hparam_columns=None,
                       num_agents=1):
  """Find the hparam settings that led to the highest score on each metric."""
  if hparam_columns is None:
    hparam_columns = HPARAM_COLUMNS

  search_hparams = [h for h in hparam_columns if h in df.columns.values]
  to_remove = []
  for h in search_hparams:
    if h != 'xm_combined_population' and len(df[h].unique()) < 2:
      to_remove.append(h)
  search_hparams = [h for h in search_hparams if h not in to_remove]
  if verbose: print('Searching for combos of', search_hparams)
  hparam_combos = df[search_hparams].drop_duplicates()
  if verbose:
    print('Investigating', len(hparam_combos), 'hparam settings')

  scores_list = []
  settings_list = []
  for k, row in hparam_combos.iterrows():
    row_dict = row.to_dict()
    settings_list.append(row_dict)

    # Check for combined population. If True the number of agents varies per
    # hparam setting.
    combined_population = (
        'xm_combined_population' in row_dict
        and row_dict['xm_combined_population']) or (
            'xm_combined_population' in df.columns.values and
            df['xm_combined_population'].unique()[0])
    num_agents = calculate_num_agents_based_on_population(
        row_dict, num_agents, combined_population, is_dict=True)

    # Recompute step limit based on number of agents
    if step_limit_1agent is not None:
      step_limit = step_limit_1agent * num_agents
    else:
      step_limit = None

    scores_list.append(get_score_for_setting(
        df, metrics, row_dict, step_limit=step_limit,
        last_x_percent=last_x_percent, run=run, verbose=False))

  scores_dict = {k: [dic[k] for dic in scores_list] for k in scores_list[0]}
  return scores_dict, settings_list


def find_best_setting_for_metric(df, metric, run='eval', step_limit_1agent=None,
                                 last_x_percent=.2, num_agents=1):
  """Find the hparam setting that led to the highest score on metric."""
  scores_dict, settings_list = find_best_settings(
      df,
      [metric],
      run=run,
      step_limit_1agent=step_limit_1agent,
      last_x_percent=last_x_percent,
      num_agents=num_agents)

  scores = scores_dict[metric]
  max_idx = scores.index(max(scores))
  return settings_list[max_idx]


def restrict_to_best_setting_for_metric(df, metric, run='eval',
                                        last_x_percent=.2, num_agents=1,
                                        step_limit_1agent=None):
  """Restrict df to hparam settings with highest score on metric."""
  best_setting = find_best_setting_for_metric(
      df, metric, run=run, last_x_percent=last_x_percent, num_agents=num_agents,
      step_limit_1agent=step_limit_1agent)
  print('Found best setting', best_setting)
  return restrict_to_setting(df, best_setting)


def copy_recursively(source, destination):
  """Copies a directory and its content.

  Args:
    source: Source directory.
    destination: Destination directory.
  """
  for src_dir, _, src_files in tf.io.gfile.walk(source):
    dst_dir = os.path.join(destination, os.path.relpath(src_dir, source))
    if not tf.io.gfile.exists(dst_dir):
      tf.io.gfile.makedirs(dst_dir)
    for src_file in src_files:
      tf.io.gfile.copy(
          os.path.join(src_dir, src_file),
          os.path.join(dst_dir, src_file),
          overwrite=True)
