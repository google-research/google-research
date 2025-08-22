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

"""BERT model experiments."""
import pickle
from typing import Sequence

from absl import app
import matplotlib.pyplot as plt

from fast_gradient_clipping.src import bert_experiment_tools


def get_dp_bert_model_experiment_results(repeats=1):
  """Runs BERT experiments over batch size."""
  batch_sizes = [100, 200, 400, 800, 1600]
  vocab_size = 100
  query_size = 5
  num_epochs = 5
  num_steps = 10
  params = []
  for batch_size in batch_sizes:
    params.append((batch_size, vocab_size, query_size, num_epochs, num_steps))

  runtimes, peak_memories = (
      bert_experiment_tools.get_dp_bert_model_compute_profile(
          params, repeats=repeats
      )
  )
  return {
      'params': params,
      'runtimes': runtimes,
      'peak_memories': peak_memories,
  }


def parse_dp_bert_model_experiment_results(results):
  r, pm, params = (
      results['runtimes'],
      results['peak_memories'],
      results['params'],
  )
  naive_mem_change, fast_mem_change = pm['naive_dp_model'], pm['fast_dp_model']
  naive_time_change, fast_time_change = r['naive_dp_model'], r['fast_dp_model']
  batch_sizes = []
  for p in params:
    batch_sizes.append(p[0])
  return {
      'batch_sizes': batch_sizes,
      'naive_mem_change': naive_mem_change,
      'fast_mem_change': fast_mem_change,
      'naive_time_change': naive_time_change,
      'fast_time_change': fast_time_change,
  }


def plot_dp_bert_model_combined_data(plot_data, fname=None):
  """Converts plot data into actualized plots."""
  fig = plt.figure(figsize=(9, 2.5))
  # Bar plots (peak memory)
  width = 50
  ax1 = fig.add_subplot(111)
  left_batch_sizes = [b - width / 2 for b in plot_data['batch_sizes']]
  ax1.bar(
      left_batch_sizes,
      plot_data['naive_mem_change'],
      label='Naive (peak memory)',
      width=width,
      hatch='//',
  )
  right_batch_sizes = [b + width / 2 for b in plot_data['batch_sizes']]
  ax1.bar(
      right_batch_sizes,
      plot_data['fast_mem_change'],
      label='Adjoint (peak memory)',
      width=width,
  )
  ax1.grid(linestyle='dotted')
  ax1.set_xlabel('Batch size |B|')
  ax1.set_ylabel('Peak memory (MB)')
  lines1, labels1 = ax1.get_legend_handles_labels()
  # Line plots (runtime)
  ax2 = plt.twinx()
  ax2.plot(
      plot_data['batch_sizes'],
      plot_data['naive_time_change'],
      label='Naive (runtime)',
      linestyle='dashed',
  )
  ax2.plot(
      plot_data['batch_sizes'],
      plot_data['fast_time_change'],
      label='Adjoint (runtime)',
  )
  ax2.set_ylabel('Runtime (seconds)')
  lines2, labels2 = ax2.get_legend_handles_labels()
  # Other annotations
  plt.title('Effect of Batch Size on Runtime and Peak Memory')
  lgd = ax2.legend(
      lines1 + lines2,
      labels1 + labels2,
      loc='upper center',
      bbox_to_anchor=(0.5, -0.25),
      ncol=2,
  )
  if fname is not None:
    plt.savefig(
        fname, format='svg', bbox_extra_artists=(lgd,), bbox_inches='tight'
    )


def main(_):
  bert_results = get_dp_bert_model_experiment_results(repeats=1)
  with open('/tmp/bert_results.pkl', 'wb') as handle:
    pickle.dump(bert_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
  bert_plot_data = parse_dp_bert_model_experiment_results(bert_results)
  plot_dp_bert_model_combined_data(bert_plot_data, '/tmp/bert_combined.svg')


if __name__ == '__main__':
  app.run(main)
