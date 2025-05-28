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

"""Fully-connected model experiments."""
import pickle
from typing import Sequence

from absl import app
import matplotlib.pyplot as plt

from fast_gradient_clipping.src import fc_experiment_tools


def plot_fully_connected_runtime_data(plot_data, fname=None):
  """Plotter for runtime data."""
  plt.figure(figsize=(6, 2.5))
  plt.grid(linestyle='dotted')
  for d in plot_data:
    plt.plot(
        d['m_values'],
        d['indirect_time_change'],
        label=f"GhostClip, q={d['q']}",
        linestyle='dashed',
    )
  for d in plot_data:
    plt.plot(
        d['m_values'], d['direct_time_change'], label=f"Adjoint, q={d['q']}"
    )
  plt.xlabel('Bias Dimension')
  plt.ylabel('Runtime (seconds)')
  plt.title('Effect of Bias Dimension on Runtime')
  lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
  if fname is not None:
    plt.savefig(
        fname, format='svg', bbox_extra_artists=(lgd,), bbox_inches='tight'
    )


def plot_fully_connected_memory_data(plot_data, fname=None):
  """Plotter for memory usage data."""
  plt.figure(figsize=(6, 2.5))
  plt.grid(linestyle='dotted')
  for d in plot_data:
    plt.plot(
        d['m_values'],
        d['indirect_mem_change'],
        label=f"GhostClip, q={d['q']}",
        linestyle='dashed',
    )
  for d in plot_data:
    plt.plot(
        d['m_values'], d['direct_mem_change'], label=f"Adjoint, q={d['q']}"
    )
  plt.xlabel('Bias Dimension')
  plt.ylabel('Peak Heap Memory (MB)')
  plt.title('Effect of Bias Dimension on Memory')
  lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
  if fname is not None:
    plt.savefig(
        fname, format='svg', bbox_extra_artists=(lgd,), bbox_inches='tight'
    )


def get_fully_connected_experiment_results(repeats=20):
  """Initial version."""
  results = []
  batch_size = 1
  # num_runs = 2
  num_runs = 12
  # q_vals = [3]
  q_vals = [3, 4, 5]
  base = 2
  for q in q_vals:
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


def parse_fully_connected_experiment_results(results):
  """Experiment parser for initial version."""
  plot_data = []
  for result in results:
    r, pm, params = (
        result['runtimes'],
        result['peak_memories'],
        result['params'],
    )
    im_mem_change, dm_mem_change = pm['indirect_model'], pm['direct_model']
    im_time_change, dm_time_change = r['indirect_model'], r['direct_model']
    q = params[0][1]
    m_values = []
    for p in params:
      m_values.append(p[3])
    plot_data.append({
        'q': q,
        'm_values': m_values,
        'indirect_mem_change': im_mem_change,
        'direct_mem_change': dm_mem_change,
        'indirect_time_change': im_time_change,
        'direct_time_change': dm_time_change,
    })
  return plot_data


def main(_):
  fc_results = get_fully_connected_experiment_results(repeats=20)
  with open('/tmp/fc1_results.pkl', 'wb') as handle:
    pickle.dump(fc_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
  fc_plot_data = parse_fully_connected_experiment_results(fc_results)
  plot_fully_connected_runtime_data(fc_plot_data, '/tmp/fc_runtimes.svg')


if __name__ == '__main__':
  app.run(main)
