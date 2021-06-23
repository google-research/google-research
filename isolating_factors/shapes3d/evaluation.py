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
from sklearn.decomposition import PCA
import tensorflow as tf

GENERATIVE_FACTORS = [('wall_hue', 'WH', 10),
                      ('object_hue', 'OH', 10),
                      ('floor_hue', 'FH', 10),
                      ('scale', 'Sc', 8),
                      ('shape', 'Sh', 4),
                      ('orientation', 'Or', 15)]


def compute_mutual_info(model, dataset_full, noise=1.0, num_latent_dims=64):
  """Computes the mutual info between generative factors and representations.

  Measures the mutual information using MINE (Belghazi et al., 2018). A
  statistics network is trained to maximize the Donsker-Varadhan form of the KL-
  divergence between the joint and product of marginals.

  Args:
    model: Model which embeds an image into a two-dimensional point.
    dataset_full: tf.data.Dataset object where each element is a tuple
      (image, generative_factors).
    noise: The magnitude of the Gaussian noise to add to the embeddings when
      calculating the mutual information.
    num_latent_dims: The dimensionality of the latent space.

  Returns:
    mutual_infos_all: A list of the 6 values of mutual information, one for each
      generative factor.
  """

  embeddings_all, gen_factors_all = [[], []]
  eval_stack_size = 256
  eval_size = 250_000
  for img_stack, gen_factors in dataset_full.batch(eval_stack_size).take(
      eval_size // eval_stack_size):
    embeddings_all.append(model(img_stack, training=False))
    gen_factors_all.append(gen_factors)

  embeddings_all = np.concatenate(embeddings_all, axis=0)
  gen_factors_all = np.concatenate(gen_factors_all, axis=0)

  dset_embeddings = tf.data.Dataset.from_tensor_slices(embeddings_all)
  dset_gen_factors = tf.data.Dataset.from_tensor_slices(gen_factors_all)
  dset_combined = tf.data.Dataset.zip((dset_embeddings, dset_gen_factors))

  num_opt_steps = 20000
  bs = 256
  lr = 3e-4
  mutual_infos_all = []
  for gen_factor_id in range(len(GENERATIVE_FACTORS)):
    opt = tf.keras.optimizers.Adam(lr)
    t_losses = []
    statistics_network = tf.keras.Sequential([
        tf.keras.layers.Input((num_latent_dims + 1,)),
        tf.keras.layers.Dense(128, 'relu'),
        tf.keras.layers.Dense(128, 'relu'),
        tf.keras.layers.Dense(128, 'relu'),
        tf.keras.layers.Dense(1),
    ])
    # Train the statistics network
    for batch in dset_combined.repeat().shuffle(10000).batch(bs).take(
        num_opt_steps):
      embeddings, labels = batch
      embeddings += np.random.normal(scale=noise, size=tf.shape(embeddings))
      labels = tf.cast(labels[:, gen_factor_id], tf.float32)[:, tf.newaxis]
      shuffled_labels = tf.random.shuffle(labels)
      with tf.GradientTape() as tape:
        t_joint = statistics_network(
            tf.concat([embeddings, labels], -1), training=True)
        t_marginals = statistics_network(
            tf.concat([embeddings, shuffled_labels], -1), training=True)
        loss = tf.math.log(tf.reduce_mean(
            tf.exp(t_marginals))) - tf.reduce_mean(t_joint)
      grads = tape.gradient(loss, statistics_network.trainable_variables)
      opt.apply_gradients(zip(grads, statistics_network.trainable_variables))
      t_losses.append(loss.numpy())
    t_losses = np.float32(t_losses)
    mutual_infos_all.append(-np.mean(t_losses[-1000:]))

  return mutual_infos_all


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
  pca_embeddings = PCA(n_components=2)
  pca_embeddings.fit(visualization_embeddings)
  explained_variance = pca_embeddings.explained_variance_ratio_
  transformed_embeddings = pca_embeddings.transform(visualization_embeddings)

  fig = plt.figure(figsize=(12, 6))
  gs = fig.add_gridspec(2, 4)
  for gen_factor_id, (gen_factor_name, _,
                      num_vals) in enumerate(GENERATIVE_FACTORS):
    fig.add_subplot(gs[gen_factor_id // 3, gen_factor_id % 3])
    plt.scatter(
        transformed_embeddings[:, 0],
        transformed_embeddings[:, 1],
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
      plt.xlabel(f'PC0, explained variance = {explained_variance[0]:.3f}')
      plt.ylabel(f'PC1, explained variance = {explained_variance[1]:.3f}')
    else:
      plt.xticks([])
      plt.yticks([])

  # Display the mutual information values
  fig.add_subplot(gs[:, 3])
  plt.bar(np.arange(len(mutual_infos_all)), mutual_infos_all, color='#394f56')
  plt.xticks(
      range(len(GENERATIVE_FACTORS)),
      [gen_factor_abbrev for _, gen_factor_abbrev, _ in GENERATIVE_FACTORS],
      fontsize=14)
  plt.title('Mutual info (nats)', fontsize=16.)
  plt.savefig(out_filename, format='png')
  return
