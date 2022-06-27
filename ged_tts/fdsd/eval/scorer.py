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

"""No-Op / placeholder scorer."""

from absl import logging
import numpy as np
import scipy


def compute_frechet_inception_distance(mu1, sigma1, mu2, sigma2, epsilon=1e-6):
  """Compute the Frechet Inception Distance (FID) score.

  Args:
    mu1: 1D np.ndarray. Mean of the first sample.
    sigma1: 2D np.ndarray. Covariance matrix of the first sample.
    mu2: 1D np.ndarray. Mean of the second sample.
    sigma2: 2D np.ndarray. Covariance matrix of the first sample.
    epsilon: Float. Small number added for numerical stability.

  Returns:
    FID score as a float.
  """
  assert mu1.shape == mu2.shape, 'mu1 and mu2 shapes differ.'
  assert sigma1.shape == sigma2.shape, 'sigma1 and simga2 shapes differ.'

  assert len(mu1.shape) == 1, 'mu1 and mu2 must be 1D vectors.'
  assert len(sigma1.shape) == 2, 'sigma1 and sigma2 must be 2D matrices..'

  diff = mu1 - mu2

  # Product might be almost singular.
  covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    logging.warning('FID calculated produced singular product -> adding epsilon'
                    '(%g) to the diagonal of the covariances.', epsilon)
    offset = np.eye(sigma1.shape[0]) * epsilon
    covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # Numerical error might give slight imaginary component.
  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError(
          'FID calculation lead to non-negligible imaginary component (%g)' % m)
    covmean = covmean.real

  tr_covmean = np.trace(covmean)
  return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_mean_and_covariance(samples):
  """Compute mean and covariance for a given set of samples.

  Args:
    samples: np.ndarray. Samples to compute the mean and covariance matrix on.

  Returns:
    Mean and covariance matrix as 1D and 2D numpy arrays respectively.
  """
  samples = np.array(samples)
  samples = np.reshape(samples, [-1, np.prod(samples.shape[1:])])
  mu = np.mean(samples, axis=0)
  sigma = np.cov(samples, rowvar=False)
  return mu, sigma


class NoOpScorer(object):
  """NoOp Scorer."""

  def __init__(self):
    """Constructor."""
    self.infer = lambda x: {}
    self.is_restored = True
    self.collect_names = []

  def restore(self):
    """Restore the model from a checkpoint."""
    pass

  def compute_scores(self, **unused_kwargs):
    """Compute scores.

    Returns:
      Empty dictionary.
    """
    if unused_kwargs:
      logging.warning('Arguments (%s) passed to NoOpScorer and will be '
                      'ignored!', str(unused_kwargs))
    return {}
