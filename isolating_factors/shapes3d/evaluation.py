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

"""Evaluate the mutual info between representations and generative factors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

GENERATIVE_FACTORS = [('wall_hue', 10),
                      ('object_hue', 10),
                      ('floor_hue', 10),
                      ('scale', 8),
                      ('shape', 4),
                      ('orientation', 15)]


def compute_mutual_info(model, dataset_full):
  """Computes the mutual info between generative factors and representations.

  Approximates the distribution over all representations p(y) by histogram, over
  the entire 480k dataset.  Then does the same with the conditional
  distributions for each of the 6 generative factors, across all of their
  possible values, in order to compute the conditional entropy. Only valid for
  two-dimensional embeddings.

  Args:
    model: Model which embeds an image into a two-dimensional point.
    dataset_full: tf.data.Dataset object where each element is a tuple
      (image, generative_factors).

  Returns:
    entropy_y: The entropy of the distribution p(y) of all embeddings.
    mutual_infos_all: A list of the 6 values of mutual information, one for each
      generative factor.
  """

  embeddings_all, gen_factors_all = [[], []]
  eval_stack_size = 256
  for img_stack, gen_factors in dataset_full.batch(eval_stack_size):
    embeddings_all.append(model(img_stack, training=False))
    gen_factors_all.append(gen_factors)

  embeddings_all = np.concatenate(embeddings_all, axis=0)
  gen_factors_all = np.concatenate(gen_factors_all, axis=0)

  entire_dataset_size = embeddings_all.shape[0]

  # Histogram the full distribution p(y)
  # The ideal total number of bins is entire_dataset_size**0.5, so the number of
  # bins per dimension is entire_dataset_size**0.25
  num_bins = int(np.power(entire_dataset_size, 0.25))
  hist_y, binsy_i, binsy_j = np.histogram2d(
      embeddings_all[:, 0], embeddings_all[:, 1], bins=num_bins, density=True)
  delta_i = binsy_i[1]-binsy_i[0]
  delta_j = binsy_j[1]-binsy_j[0]

  p_y = hist_y * delta_i * delta_j
  entropy_y = -np.sum(p_y[p_y > 0] * np.log(p_y[p_y > 0]))

  # Histogram the conditional distributions p(y|x) using the same bins
  mutual_infos_all = []
  for gen_factor_id, (_, num_vals) in enumerate(GENERATIVE_FACTORS):
    entropy_conditional_all = []
    for x_sample in range(num_vals):
      embeddings_subset = embeddings_all[gen_factors_all[:, gen_factor_id] ==
                                         x_sample]
      hist_conditional = np.histogram2d(
          embeddings_subset[:, 0],
          embeddings_subset[:, 1],
          bins=[binsy_i, binsy_j],
          density=True)[0]
      p_conditional = hist_conditional * delta_i * delta_j
      entropy_conditional = -np.sum(p_conditional[p_conditional > 0] *
                                    np.log(p_conditional[p_conditional > 0]))
      entropy_conditional_all.append(entropy_conditional)

    entropy_conditional = np.mean(entropy_conditional_all)
    mutual_info = entropy_y - entropy_conditional
    mutual_infos_all.append(mutual_info)

  return entropy_y, mutual_infos_all


def visualize_embeddings(mutual_infos_all, visualization_embeddings,
                         visualization_gen_factors, out_filename):
  """Saves a figure displaying sample embeddings and the mutual information.

  Args:
    mutual_infos_all: A list of the 6 values of mutual information between each
      of the generative factors and the embedding space.
    visualization_embeddings: [N, 2] tensor with a sampling of N embeddings
      to display.
    visualization_gen_factors: [N, 6] tensor with the integer values for each of
      the generative factors of the corresponding image.
    out_filename: A filename for saving the figure (in PNG format).
  """

  fig = plt.figure(figsize=(12, 6))
  gs = fig.add_gridspec(2, 4)
  for gen_factor_id, (gen_factor_name,
                      num_vals) in enumerate(GENERATIVE_FACTORS):
    fig.add_subplot(gs[gen_factor_id // 3, gen_factor_id % 3])
    plt.scatter(
        visualization_embeddings[:, 0],
        visualization_embeddings[:, 1],
        s=30.,
        c=np.float32(visualization_gen_factors[:, gen_factor_id]) /
        float(num_vals),
        cmap='jet')
    if not gen_factor_id:
      plt.title('Colored by {}'.format(gen_factor_name), fontsize=14.)
    else:
      plt.title('{}'.format(gen_factor_name), fontsize=14.)
    # To declutter the figure, since all six of these subplots are over the same
    # space, only include axis labels and ticks for the bottom left subplot.
    if gen_factor_id == 3:
      plt.xlabel('X')
      plt.ylabel('Y')
    else:
      plt.xticks([])
      plt.yticks([])

  # Display the mutual information values
  fig.add_subplot(gs[:, 3])
  plt.bar(np.arange(len(mutual_infos_all)), mutual_infos_all)
  plt.ylim([0., 2.5])
  plt.xticks(range(len(GENERATIVE_FACTORS)),
             [gen_factor_name for gen_factor_name, _ in GENERATIVE_FACTORS],
             rotation=90)
  plt.title('Mutual info (nats)', fontsize=16.)

  plt.savefig(out_filename, format='png')
  return
