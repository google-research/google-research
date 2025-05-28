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

"""A library for computing unsupervised embedding quality metrics."""
from typing import Mapping, Optional

import numpy as np


def report_all_metrics(tensor):
  """Computes all metric values given a tensor and its SVD.

  Args:
    tensor (dense matrix): Input embeddings.

  Returns:
    Mapping[str, float]: All metric values.
  """
  # Pre-compute SVD for metric computations.
  u, s, _ = np.linalg.svd(tensor, compute_uv=True, full_matrices=False)
  fns = [
      rankme,
      coherence,
      pseudo_condition_number,
      alpha_req,
      stable_rank,
      ne_sum,
      self_clustering,
  ]
  return dict((fn.__name__, fn(tensor, u=u, s=s)) for fn in fns)


def pseudo_condition_number(
    tensor,
    s = None,
    epsilon = 1e-12,
    **_
):
  """Implementation of the pseudo-condition number metric.

  Args:
    tensor (dense matrix): Input embeddings.
    s (optional, dense vector): Singular values of `tensor`.
    epsilon (float): Numerical epsilon.

  Returns:
    float: Pseudo-condition number metric value.
  """
  if s is None:
    s = np.linalg.svd(tensor, compute_uv=False)
  return s[-1] / (s[0] + epsilon)


def coherence(tensor, u = None, **_):
  """Implementation of the coherence metric.

  Args:
    tensor (dense matrix): Input embeddings.
    u (optional, dense matrix): Left singular vectors of `tensor`.

  Returns:
    float: Coherence metric value.
  """
  if u is None:
    u, _, _ = np.linalg.svd(tensor, compute_uv=True, full_matrices=False)
  maxu = np.linalg.norm(u, axis=1).max() ** 2
  return maxu * u.shape[0] / u.shape[1]


def stable_rank(
    tensor,
    s = None,
    epsilon = 1e-12,
    **_
):
  """Implementation of the stable rank metric.

  Args:
    tensor (dense matrix): Input embeddings.
    s (optional, dense vector): Singular values of `tensor`.
    epsilon (float): Numerical epsilon.

  Returns:
    float: Stable rank metric value.
  """
  if s is None:
    s = np.linalg.svd(tensor, compute_uv=False)
  trace = np.square(tensor).sum()
  denominator = s[0] * s[0] + epsilon
  return trace / denominator


def self_clustering(tensor, epsilon = 1e-12, **_):
  """Implementation of the SelfCluster metric.

  Args:
    tensor (dense matrix): Input embeddings.
    epsilon (float): Numerical epsilon.

  Returns:
    float: SelfCluster metric value.
  """
  tensor = tensor + epsilon
  tensor /= np.linalg.norm(tensor, axis=1)[:, np.newaxis]
  n, d = tensor.shape
  expected = n + n * (n - 1) / d
  actual = np.sum(np.square(tensor @ tensor.T))
  return (actual - expected) / (n * n - expected)


def rankme(
    tensor,
    s = None,
    epsilon = 1e-12,
    **_
):
  """Implementation of the RankMe metric.

  This metric is defined in "RankMe: Assessing the Downstream Performance of
  Pretrained Self-Supervised Representations by Their Rank". Garrido et al.
  arXiv:2210.02885.

  Args:
    tensor (dense matrix): Input embeddings.
    s (optional, dense vector): Singular values of `tensor`.
    epsilon (float): Numerical epsilon.

  Returns:
    float: RankMe metric value.
  """
  if s is None:
    s = np.linalg.svd(tensor, compute_uv=False)
  p_ks = s / np.sum(s + epsilon) + epsilon
  return np.exp(-np.sum(p_ks * np.log(p_ks)))


def ne_sum(tensor, epsilon = 1e-12, **_):
  """Implementation of the NESum metric.

  This metric is defined in "Exploring the Gap between Collapsed & Whitened
  Features in Self-Supervised Learning". He & Ozay, ICML 2022. See Definition
  4.1 from the paper for more details.

  Args:
    tensor (dense matrix): Input embeddings.
    epsilon (float): Numerical epsilon.

  Returns:
    float: NESum metric value.
  """
  cov_t = np.cov(tensor.T)
  ei_t = np.linalg.eigvalsh(cov_t) + epsilon
  return (ei_t / ei_t[-1]).sum()


def alpha_req(
    tensor,
    s = None,
    epsilon = 1e-12,
    **_
):
  """Implementation of the Alpha-ReQ metric.

  This metric is defined in "α-ReQ: Assessing representation quality in
  self-supervised learning by measuring eigenspectrum decay". Agrawal et al.,
  NeurIPS 2022.

  Args:
    tensor (dense matrix): Input embeddings.
    s (optional, dense vector): Singular values of `tensor`.
    epsilon (float): Numerical epsilon.

  Returns:
    float: Alpha-ReQ metric value.
  """
  if s is None:
    s = np.linalg.svd(tensor, compute_uv=False)
  n = s.shape[0]
  s = s + epsilon
  features = np.vstack([np.linspace(1, 0, n), np.ones(n)]).T
  a, _, _, _ = np.linalg.lstsq(features, np.log(s), rcond=None)
  return a[0]
