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

"""Embedding model experiments."""
import pickle
from typing import Sequence

from absl import app
import matplotlib.pyplot as plt

from fast_gradient_clipping.src import emb_experiment_tools


def get_embedding_experiment_results(repeats=20):
  """Runs multiple embedding experiments."""
  results = []
  output_dim = 10
  # num_runs = 2
  num_runs = 10
  # vocab_sizes = [500]
  vocab_sizes = [5000, 7500, 10000]
  for vocab_size in vocab_sizes:
    params = []
    for i in range(num_runs):
      params.append((vocab_size, 1000 * (i + 1), output_dim))
    runtimes, peak_memories = (
        emb_experiment_tools.get_embedding_compute_profile(
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


def parse_embedding_experiment_results(results):
  """Parses a set of experiment results."""
  plot_data = []
  for result in results:
    r, pm, params = (
        result['runtimes'],
        result['peak_memories'],
        result['params'],
    )
    im_mem_change, dm_mem_change = pm['indirect_model'], pm['direct_model']
    im_time_change, dm_time_change = r['indirect_model'], r['direct_model']
    vocab_size = params[0][0]
    q_values = []
    for p in params:
      q_values.append(p[1])
    plot_data.append({
        'vocab_size': vocab_size,
        'query_size': q_values,
        'indirect_mem_change': im_mem_change,
        'direct_mem_change': dm_mem_change,
        'indirect_time_change': im_time_change,
        'direct_time_change': dm_time_change,
    })
  return plot_data


def plot_embedding_runtime_data(plot_data, fname=None):
  """Plots runtime results."""
  plt.figure(figsize=(6, 2.5))
  plt.grid(linestyle='dotted')
  for d in plot_data:
    plt.plot(
        d['query_size'],
        d['indirect_time_change'],
        label=f"GhostClip, r={d['vocab_size']}",
        linestyle='dashed',
    )
  for d in plot_data:
    plt.plot(
        d['query_size'],
        d['direct_time_change'],
        label=f"Adjoint, r={d['vocab_size']}",
    )
  plt.xlabel('Query Size')
  plt.ylabel('Runtime (seconds)')
  plt.title('Effect of Query Size on Runtime')
  lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
  if fname is not None:
    plt.savefig(
        fname, format='svg', bbox_extra_artists=(lgd,), bbox_inches='tight'
    )


def plot_embedding_memory_data(plot_data, fname=None):
  """Plots memory results."""
  plt.figure(figsize=(6, 2.5))
  plt.grid(linestyle='dotted')
  for d in plot_data:
    plt.plot(
        d['query_size'],
        d['indirect_mem_change'],
        label=f"GhostClip, r={d['vocab_size']}",
        linestyle='dashed',
    )
  for d in plot_data:
    plt.plot(
        d['query_size'],
        d['direct_mem_change'],
        label=f"Adjoint, r={d['vocab_size']}",
    )
  plt.xlabel('Query Size')
  plt.ylabel('Peak Heap Memory (MB)')
  plt.title('Effect of Query Size on Memory')
  lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
  if fname is not None:
    plt.savefig(
        fname, format='svg', bbox_extra_artists=(lgd,), bbox_inches='tight'
    )


def main(_):
  embedding_results = get_embedding_experiment_results(repeats=20)
  with open('/tmp/embedding_results.pkl', 'wb') as handle:
    pickle.dump(embedding_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
  embedding_plot_data = parse_embedding_experiment_results(embedding_results)
  plot_embedding_runtime_data(
      embedding_plot_data, '/tmp/embedding_runtimes.svg'
  )


if __name__ == '__main__':
  app.run(main)
