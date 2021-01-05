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

"""Workflows for the IPAGNN model evaluation and analysis."""

import collections
import datetime
import json
import os
import time

from absl import logging  # pylint: disable=unused-import
import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from ipagnn.lib import checkpoint_utils
from ipagnn.lib import dataset_utils
from ipagnn.lib import workflows

gfile = tf.io.gfile


def run(run_configuration):
  if run_configuration.mode == 'eval_multi_dataset':
    analyze_once(run_configuration)
  elif run_configuration.mode == 'aggregate':
    run_aggregate(run_configuration)
  else:
    workflows.run(run_configuration)


def analyze_once(run_configuration):
  """Analyzes the existing model checkpoint.

  Runs inference for the model for each of the evaluation datasets, and writes
  the metrics to disk as JSON.

  Args:
    run_configuration: The setup.RunConfiguration for the run.
  """
  analysis_run_dir = run_configuration.run_dir
  original_checkpoint_path = run_configuration.original_checkpoint_path
  config = run_configuration.config
  data_dir = run_configuration.data_dir
  adapter = run_configuration.adapter
  optimizer = adapter.create_optimizer(run_configuration)

  original_run_dir = checkpoint_utils.get_run_dir(original_checkpoint_path)
  original_run_name = os.path.basename(original_run_dir)
  original_config_path = os.path.join(original_run_dir, 'config.json')
  checkpoint_basename = os.path.basename(original_checkpoint_path)
  analysis_dir = os.path.join(analysis_run_dir, checkpoint_basename)
  analysis_file = os.path.join(analysis_dir, 'data.json')
  checkpoint_dir = checkpoint_utils.build_checkpoint_dir(analysis_dir)
  checkpoint_path = os.path.join(checkpoint_dir, checkpoint_basename)
  gfile.makedirs(checkpoint_dir)
  gfile.makedirs(analysis_dir)

  # Save a copy of the original checkpoint.
  gfile.copy(original_checkpoint_path, checkpoint_path)
  # Save analysis metadata.
  metadata = {
      'name': original_run_name,
      'run_dir': original_run_dir,
      'timestamp': datetime.datetime.now().timestamp(),
  }
  metadata_path = os.path.join(analysis_dir, 'metadata.json')
  with gfile.GFile(metadata_path, 'w') as f:
    f.write(json.dumps(metadata))

  # Save a copy of the original config.
  new_config_path = os.path.join(analysis_run_dir, 'config.json')
  # We set overwrite=True to handle preemption.
  gfile.copy(original_config_path, new_config_path, overwrite=True)
  logging.info('Saving results to %s', analysis_file)

  # Load the datasets to analyze.
  analysis_results = []
  eval_dataset_names = config.launcher.eval_dataset_names.split(',')
  for dataset_name in eval_dataset_names:
    logging.info('Evaluating with checkpoint_path: %s', checkpoint_path)
    logging.info('Evaluating on dataset: %s', dataset_name)
    dataset_info = dataset_utils.get_dataset(data_dir, config, dataset_name)
    run_configuration = dataclasses.replace(
        run_configuration,
        run_dir=analysis_dir,
        original_checkpoint_path=checkpoint_path,
        info=dataset_info.info,
        dataset_info=dataset_info,
    )
    metrics = workflows.predict_once(run_configuration, optimizer)
    logging.info('Done evaluating on dataset: %s', dataset_name)

    results = {
        'dataset_name': dataset_name,
        'accuracy': metrics['accuracy'].tolist(),
        'denominator': metrics['denominator'].tolist(),
    }
    analysis_results.append(results)

    with gfile.GFile(analysis_file, 'wb') as f:
      json.dump(analysis_results, f)


def _model_name(config):
  name = config['model']['name']
  if name in ('IPAGNN', 'IPAGNNInterpolant'):
    # These model implementations are used for multiple model variants that we
    # care about separately. So, we use the more specific model name here.
    return config['model']['interpolant']['name']
  if name == 'StackedLSTMModel':
    name += f"-{config['dataset']['representation']}"
  return name


def run_aggregate(run_configuration):
  """Aggregates the analyses of the trained models.

  Looks in the experiment directory for analysis run directories, and
  aggregates them to perform model selection and produce plots.

  Args:
    run_configuration: The setup.RunConfiguration for the run.
  """
  sns.set_style('darkgrid')
  sns.set_context('paper')

  run_dir = run_configuration.run_dir
  config = run_configuration.config
  xid = config.analysis.xid
  all_results, configs = load_analysis_data(run_configuration)

  names_map = {
      'rnn': 'Line-by-Line RNN',
      'StackedLSTMModel': 'Line-by-Line RNN',
      'StackedLSTMModel-code': 'Line-by-Line RNN',
      'StackedLSTMModel-trace': 'Trace RNN (Oracle)',
      'IPAGNN': 'IPA-GNN',
      'GAT': 'R-GAT',
  }
  colors = {
      'IPA-GNN': 'blue',
      'NoControl': 'brown',
      'NoExecute': 'green',
      'GGNN': 'orange',
      'R-GAT': 'gray',
      'Line-by-Line RNN': 'red',
      'Trace RNN (Oracle)': 'purple',
  }
  markers = {
      'IPA-GNN': 'X',
      'NoControl': 'o',
      'NoExecute': 'p',
      'GGNN': 's',
      'R-GAT': 's',
      'Line-by-Line RNN': '<',
      'Trace RNN (Oracle)': '^',
  }
  linestyles = {
      'Trace RNN (Oracle)': '--',
  }

  # Model selection
  if 'decimal-large-state-L10-partial' in config.launcher.eval_dataset_names:
    validation_dataset_name = 'control_flow_programs/decimal-large-state-L10-partial'
  elif 'decimal-large-state-L10' in config.launcher.eval_dataset_names:
    validation_dataset_name = 'control_flow_programs/decimal-large-state-L10'
  elif 'multivar-templates-train-L10-partial' in config.launcher.eval_dataset_names:
    validation_dataset_name = (
        'control_flow_programs/decimal-multivar-templates-train-L10-partial')
  elif 'multivar-templates-train-L10' in config.launcher.eval_dataset_names:
    validation_dataset_name = (
        'control_flow_programs/decimal-multivar-templates-train-L10')
  best_analysis_run_dir_by_name = {}
  best_results_by_name = {}
  best_accuracy_by_name = collections.defaultdict(float)
  for analysis_run_dir in all_results:
    results = all_results[analysis_run_dir]
    results_by_dataset_name = {
        result['dataset_name']: result for result in results}
    config = configs[analysis_run_dir]
    name = _model_name(config)
    selection_results = results_by_dataset_name[validation_dataset_name]
    selection_accuracy = (
        selection_results['accuracy'] / selection_results['denominator'])
    if selection_accuracy > best_accuracy_by_name[name]:
      best_accuracy_by_name[name] = selection_accuracy
      best_analysis_run_dir_by_name[name] = analysis_run_dir
      best_results_by_name[name] = results

  print(best_analysis_run_dir_by_name)

  # Plot the results
  plt.figure(0)
  plt.clf()
  handles = []
  all_lengths = set()
  for name in [
      'IPAGNN',
      'NoControl',
      'NoExecute',
      'GGNN',
      'GAT',
      'StackedLSTMModel-code',
      'StackedLSTMModel-trace',
  ]:
    if name not in best_results_by_name:
      continue
    results = best_results_by_name[name]
    name = names_map.get(name, name)
    if 'partial' in validation_dataset_name and 'Oracle' in name:
      continue

    lengths = []
    accuracies = []
    for result in results:
      length = int(result['dataset_name'].split('L')[-1].split('-')[0])
      all_lengths.add(length)
      lengths.append(length)
      accuracies.append(result['accuracy'] / result['denominator'])
    accuracies = np.array(accuracies)

    se = np.sqrt(accuracies * (1 - accuracies) / result['denominator'])  # pylint: disable=undefined-loop-variable
    linestyle = linestyles.get(name, None)
    ax = sns.lineplot(
        x=lengths,
        y=accuracies,
        label=name,
        marker=markers.get(name),
        markerfacecolor=colors.get(name),
        color=colors.get(name),
        linestyle=linestyle,
    )
    if linestyle is not None:
      ax.lines[-1].set_linestyle(linestyle)
    ax.fill_between(x=lengths,
                    y1=accuracies - se,
                    y2=accuracies + se,
                    color=colors.get(name),
                    alpha=0.2)
    handle, unused_label = ax.get_legend_handles_labels()
    handles.append(handle)

  plt.xlabel('Program Length')
  plt.xticks(sorted(all_lengths))
  plt.ylabel('Average Accuracy')
  plt.ylim(0, 1)
  # plt.legend(handles=handles, loc='upper right')
  date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  image_path = os.path.join(run_dir, f'{date}-length-generalization-{xid}.png')
  with gfile.GFile(image_path, 'wb') as f:
    plt.savefig(f)


def load_analysis_data(run_configuration):
  """Loads the analysis data already computed in the given experiment_dir."""
  config = run_configuration.config
  experiment_dir = config.analysis.experiment_dir
  # Example path and path format:
  # 17515048/4-rd=4bs=32,hs=256,n=IPAGNN,lr=0.1/ckpt_875000/data.json
  # experiment_dir/analysis_run_dir/analysis_dir/data.json
  # TODO(dbieber): Switch to path_utils.get_run_dirs(experiment_dir).
  analysis_run_dirs = [os.path.join(experiment_dir, name)
                       for name in gfile.listdir(experiment_dir)]
  analysis_run_dirs = [d for d in analysis_run_dirs
                       if gfile.exists(os.path.join(d, 'config.json'))]

  all_results = {}
  configs = {}
  for analysis_run_dir in analysis_run_dirs:
    # Load the config.
    config_filepath = os.path.join(analysis_run_dir, 'config.json')
    with gfile.GFile(config_filepath, 'rb') as f:
      config = json.load(f)

    # Load the analysis results for each checkpoint analyzed.
    checkpoint_basenames = [
        name for name in gfile.listdir(analysis_run_dir)
        if gfile.isdir(os.path.join(analysis_run_dir, name))
    ]
    for checkpoint_basename in checkpoint_basenames:
      analysis_dir = os.path.join(analysis_run_dir, checkpoint_basename)
      analysis_file = os.path.join(analysis_dir, 'data.json')
      if gfile.exists(analysis_file):
        with gfile.GFile(analysis_file, 'rb') as f:
          analysis_results = json.load(f)
          all_results[analysis_run_dir] = analysis_results
    configs[analysis_run_dir] = config
  return all_results, configs


def run_eval_multi_dataset(run_configuration):
  """Evaluates on checkpoints as they become available."""
  config = run_configuration.config
  run_dir = run_configuration.run_dir
  data_dir = run_configuration.data_dir
  adapter = run_configuration.adapter
  optimizer = adapter.create_optimizer(run_configuration)

  eval_dataset_names = config.launcher.eval_dataset_names.split(',')
  dataset_infos = [dataset_utils.get_dataset(data_dir, config, name)
                   for name in eval_dataset_names]
  all_dataset_ids = set(range(len(dataset_infos)))
  dataset_ids_evaluated = set()

  last_dataset_id = -1
  last_checkpoint_path = None
  last_checkpoint_time = time.time()
  checkpoint_dir = checkpoint_utils.build_checkpoint_dir(run_dir)
  success_path = checkpoint_utils.build_success_path(run_dir)
  error_count = 0
  while True:
    success = gfile.exists(success_path)
    # Always evaluate at the latest checkpoint.
    checkpoint_path = checkpoint_utils.latest_checkpoint(checkpoint_dir)
    # Choose the dataset to evaluate.

    if checkpoint_path is not None and checkpoint_path != last_checkpoint_path:
      # The dataset ids evaluated at the latest checkpoint.
      # Our goal is to evaluate all the datasets at the final checkpoint as soon
      # as possible, while providing best effort progress updates along the way.
      dataset_ids_evaluated = set()

    dataset_id = (last_dataset_id + 1) % len(dataset_infos)
    if (dataset_id in dataset_ids_evaluated
        and dataset_ids_evaluated != all_dataset_ids):
      dataset_id = next(iter(all_dataset_ids - dataset_ids_evaluated))

    if ((checkpoint_path, dataset_id) != (last_checkpoint_path, last_dataset_id)
        and dataset_ids_evaluated != all_dataset_ids
        and checkpoint_path is not None):
      logging.info('Evaluating with checkpoint_path: %s', checkpoint_path)
      logging.info('Evaluating on dataset id: %d', dataset_id)
      run_configuration.dataset_info = dataset_infos[dataset_id]
      try:
        workflows.eval_once(run_configuration, checkpoint_path, optimizer)
      except:  # pylint: disable=bare-except
        logging.info('Could not evaluate %s on dataset %d', checkpoint_path,
                     dataset_id)
        error_count += 1
        if error_count >= 10 or config.debug:
          raise
      last_dataset_id = dataset_id
      dataset_ids_evaluated.add(dataset_id)
      last_checkpoint_path = checkpoint_path
      last_checkpoint_time = time.time()
    else:
      if success:
        logging.info('SUCCESS file found. Stopping.')
        break
      if time.time() - last_checkpoint_time > config.eval_timeout:
        logging.info('Timed out waiting for checkpoint. Stopping.')
        break
      logging.info('Waiting for checkpoint.')
      time.sleep(15)
