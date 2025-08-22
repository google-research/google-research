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

"""Fully-connected rebuttal experiments."""
import pickle
from typing import Sequence

from absl import app
import matplotlib.pyplot as plt

from fast_gradient_clipping.src import fc_experiment_tools


def get_fully_connected_experiment_results_v2(repeats=20):
  """Version for the rebuttal."""
  results = []
  num_runs = 8
  # num_runs = 2
  base = 2
  q = 3
  batch_sizes = [250, 500, 1000]
  # batch_sizes = [10]
  for batch_size in batch_sizes:
    p, r = 2, base**num_runs
    params = []
    for i in range(num_runs - 1):
      params.append((p, q, r, base ** (i + 1), batch_size))
    runtimes, peak_memories = (
        fc_experiment_tools.get_fully_connected_compute_profile(
            params, repeats=repeats
        )
    )
    single_result = {
        'params': params,
        'runtimes': runtimes,
        'peak_memories': peak_memories,
    }
    results.append(single_result)
  return results


def parse_fully_connected_experiment_results_v2(results):
  """Experiment parser for rebuttal version."""
  plot_data = []
  for result in results:
    r, pm, params = (
        result['runtimes'],
        result['peak_memories'],
        result['params'],
    )
    im_mem_change, dm_mem_change = pm['indirect_model'], pm['direct_model']
    im_time_change, dm_time_change = r['indirect_model'], r['direct_model']
    batch_size = params[0][4]
    m_values = []
    for p in params:
      m_values.append(p[3])
    plot_data.append({
        'batch_size': batch_size,
        'm_values': m_values,
        'indirect_mem_change': im_mem_change,
        'direct_mem_change': dm_mem_change,
        'indirect_time_change': im_time_change,
        'direct_time_change': dm_time_change,
    })
  return plot_data


def plot_fully_connected_runtime_data_v2(plot_data, fname=None):
  """Plotter for runtime data."""
  plt.figure(figsize=(6, 2.5))
  plt.grid(linestyle='dotted')
  for d in plot_data:
    plt.plot(
        d['m_values'],
        d['indirect_time_change'],
        label=f"GhostClip, |B|={d['batch_size']}",
        linestyle='dashed',
    )
  for d in plot_data:
    plt.plot(
        d['m_values'],
        d['direct_time_change'],
        label=f"Adjoint, |B|={d['batch_size']}",
    )
  plt.xlabel('Bias Dimension')
  plt.ylabel('Runtime (seconds)')
  plt.title('Effect of Batch Size on Runtime')
  lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
  if fname is not None:
    plt.savefig(
        fname, format='svg', bbox_extra_artists=(lgd,), bbox_inches='tight'
    )


def plot_fully_connected_memory_data_v2(plot_data, fname=None):
  """Plotter for memory usage data."""
  plt.figure(figsize=(6, 2.5))
  plt.grid(linestyle='dotted')
  for d in plot_data:
    plt.plot(
        d['m_values'],
        d['indirect_mem_change'],
        label=f"GhostClip, |B|={d['batch_size']}",
        linestyle='dashed',
    )
  for d in plot_data:
    plt.plot(
        d['m_values'],
        d['direct_mem_change'],
        label=f"Adjoint, |B|={d['batch_size']}",
    )
  plt.xlabel('Bias Dimension')
  plt.ylabel('Peak Heap Memory (MB)')
  plt.title('Effect of Batch Size on Memory')
  lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
  if fname is not None:
    plt.savefig(
        fname, format='svg', bbox_extra_artists=(lgd,), bbox_inches='tight'
    )


def main(_):
  fc_results_v2 = get_fully_connected_experiment_results_v2(repeats=20)
  with open('/tmp/fc2_results.pkl', 'wb') as handle:
    pickle.dump(fc_results_v2, handle, protocol=pickle.HIGHEST_PROTOCOL)
  fc_plot_data_v2 = parse_fully_connected_experiment_results_v2(fc_results_v2)
  plot_fully_connected_runtime_data_v2(
      fc_plot_data_v2, '/tmp/fc_runtimes_v2.svg'
  )


if __name__ == '__main__':
  app.run(main)
