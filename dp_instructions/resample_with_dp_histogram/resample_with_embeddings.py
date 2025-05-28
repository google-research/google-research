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

"""Resample with DP histogram."""

import argparse
import math
import pickle

import faiss
import numpy as np
import sklearn


def clustering_with_features(
    features, num_clusters=100, seed=0, kmeans_num_redo=5, kmeans_max_iter=500  # pylint: disable=redefined-outer-name
):
  """Clustering with features."""
  features = sklearn.preprocessing.normalize(features, norm='l2', axis=1)
  pca = sklearn.decomposition.PCA(
      n_components=None, whiten=False, random_state=seed + 1
  )
  pca.fit(features)
  s = np.cumsum(pca.explained_variance_ratio_)
  idx = np.argmax(s >= 0.9)  # pylint: disable=redefined-outer-name
  print(f'performing clustering in lower dimension = {idx}')
  features = pca.transform(features)[:, : idx + 1]
  features = features.astype(np.float32)

  # Cluster
  kmeans = faiss.Kmeans(  # pylint: disable=redefined-outer-name
      features.shape[1],
      num_clusters,
      niter=kmeans_max_iter,
      verbose=True,
      nredo=kmeans_num_redo,
      update_index=True,
      seed=seed + 2,
  )
  kmeans.train(features)
  return kmeans, pca


def get_pca_index(pca, explained_variance_ratio=0.9):  # pylint: disable=redefined-outer-name
  s = np.cumsum(pca.explained_variance_ratio_)
  idx = np.argmax(s >= explained_variance_ratio)  # pylint: disable=redefined-outer-name
  return idx


def build_histogram(
    features, kmeans, pca, num_clusters=100, return_projected_embeddings=False  # pylint: disable=redefined-outer-name
):
  """Build histogram."""
  features = sklearn.preprocessing.normalize(features, norm='l2', axis=1)
  s = np.cumsum(pca.explained_variance_ratio_)
  idx = np.argmax(s >= 0.9)  # pylint: disable=redefined-outer-name
  print(f'performing clustering in lower dimension = {idx}')

  batch_size = 100000
  projected_features = []
  steps = math.ceil(features.shape[0] / batch_size)
  for i in range(steps):
    print(f'processing batch {i+1}/{steps}')
    batch = features[i * batch_size : (i + 1) * batch_size]
    projected_embeddings = pca.transform(batch)[:, : idx + 1]
    projected_features.append(projected_embeddings.astype(np.float32))
  projected_features = np.vstack(projected_features)
  print('projected_features_shape: ', projected_features.shape)
  features = projected_features

  _, labels = kmeans.index.search(features, 1)
  labels = labels.reshape(-1)
  bins = np.histogram(
      labels, bins=num_clusters, range=[0, num_clusters], density=True
  )[0]

  if not return_projected_embeddings:
    return labels, bins / bins.sum()
  else:
    return labels, bins / bins.sum(), features


def subsample(arr, num_targets, seed):  # pylint: disable=redefined-outer-name
  np.random.seed(seed)
  if arr.shape[0] > num_targets:
    idx = np.random.choice(arr.shape[0], size=num_targets, replace=False)  # pylint: disable=redefined-outer-name
    arr = arr[idx]
  return arr


parser = argparse.ArgumentParser()
parser.add_argument('--real_embeddings_path', type=str, required=True)
parser.add_argument('--syn_embeddings_path', type=str, required=True)
parser.add_argument('--syn_instructions_path', type=str, required=True)
parser.add_argument('--output_name', type=str, required=True)
parser.add_argument('--num_buckets', type=int, required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_samples', type=int, default=180000)
parser.add_argument('--histogram_sigma', type=float, default=10)
parser.add_argument('--sampling_method', type=str, default='uniform')


args = parser.parse_args()


assert args.sampling_method in ['uniform', 'distance']

num_buckets = args.num_buckets
seed = args.seed  # pylint: disable=redefined-outer-name
num_samples = args.num_samples  # number of real instructions to use

real_embeddings_path = args.real_embeddings_path
syn_embeddings_path = args.syn_embeddings_path
syn_instructions_path = args.syn_instructions_path

real_instruction_embeddings = np.load(real_embeddings_path)  # size = (N, 768)
syn_instruction_embeddings = np.load(syn_embeddings_path)  # size = (M, 768)

full_syn_instructions = np.array(pickle.load(open(syn_instructions_path, 'rb')))
# Take the first M instructions. We may generate more than M synthetic
# instructions but only compute embeddings for the first M synthetic samples.
full_syn_instructions = full_syn_instructions[
    0 : syn_instruction_embeddings.shape[0]
]

# we use these embeddings for resampling.
full_syn_instruction_embeddings = syn_instruction_embeddings
# we use a subset of synthetic embeddings for clustering to reduce computational
# cost.
syn_instruction_embeddings = subsample(
    syn_instruction_embeddings, num_samples, seed
)

# Cluster synthetic embeddings into target number of buckets. To save time, we
# use PCA to reduce the dimensionality of the embeddings before clustering,
# following previous work.
kmeans, pca_params = clustering_with_features(
    syn_instruction_embeddings,
    num_clusters=num_buckets,
    seed=seed,
    kmeans_num_redo=5,
    kmeans_max_iter=500,
)
# Use the clustering result to build histogram for both real and synthetic
# embeddings.
p_labels, p_bins = build_histogram(
    real_instruction_embeddings, kmeans, pca_params, num_clusters=num_buckets
)
q_labels, q_bins = build_histogram(
    syn_instruction_embeddings, kmeans, pca_params, num_clusters=num_buckets
)
# pylint: disable=unbalanced-tuple-unpacking
full_q_labels, full_q_bins, full_projected_embeddings = build_histogram(
    kmeans,
    pca_params,
    num_clusters=num_buckets,
    return_projected_embeddings=True,
)


def privitize_p_bins(p_bins, n, sigma=10):  # pylint: disable=redefined-outer-name
  noise = np.random.normal(0, sigma, size=p_bins.shape[0])
  noisy_bins = p_bins * n + noise
  noisy_bins /= n
  noisy_bins[noisy_bins < 0] = 0
  return noisy_bins


# DP-fy the real histogram
noisy_p_bins = privitize_p_bins(p_bins, num_samples, sigma=args.histogram_sigma)

n_full = full_q_labels.shape[0]
ratio = noisy_p_bins / q_bins  # you can also use full_q_bins instead of q_bins.
subsampled_instruction_embeddings = []
subsampled_instructions = []

max_ratio = max(ratio)

# Number of target samples after resampling. Here we simply set it to be the
# same as the number of real samples. You can also set it to be a different
# number.
num_target_samples = num_samples

sorted_index = np.argsort(ratio)[::-1]


for idx in sorted_index:  # pylint: disable=redefined-outer-name
  current_bin_embeddings = full_syn_instruction_embeddings[full_q_labels == idx]
  current_bin_instructions = full_syn_instructions[full_q_labels == idx]
  current_bin_projected_embeddings = full_projected_embeddings[
      full_q_labels == idx
  ]

  num_samples_from_current_bin = int(num_target_samples * noisy_p_bins[idx])
  if num_samples_from_current_bin > current_bin_embeddings.shape[0]:
    print(
        f'Warning: number of samples from bin {idx} is larger than the number'
        ' of samples in the bin. We will use all samples from this bin.'
        ' Generate more initial synthetic samples to avoid this.'
    )
    num_samples_from_current_bin = current_bin_embeddings.shape[0]

  random_idx = np.random.choice(
      current_bin_embeddings.shape[0],
      size=num_samples_from_current_bin,
      replace=False,
  )
  subsampled_instruction_embeddings.append(current_bin_embeddings[random_idx])
  subsampled_instructions.extend(current_bin_instructions[random_idx].tolist())

subsampled_instruction_embeddings = np.vstack(subsampled_instruction_embeddings)

n_after_subsample = subsampled_instruction_embeddings.shape[0]

print('Number of instructions after resampling: ', len(subsampled_instructions))
print('One intruction from the first cluster: ', subsampled_instructions[0])

pickle.dump(
    subsampled_instructions,
    open(
        f'max{max_ratio}_subsampled_instructions_n{n_after_subsample}_{args.output_name}_seed{seed}.pkl',
        'wb',
    ),
)
np.save(
    f'max{max_ratio}_subsampled_instruction_embeddings_n{n_after_subsample}_{args.output_name}_seed{seed}.npy',
    subsampled_instruction_embeddings,
)
