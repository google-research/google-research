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

# Lint as: python3
"""utils classes and functions for the experiment script."""

import numpy as np
import scipy
from six.moves import range
import sklearn.metrics


def compute_map_auc_dprime(pred, gt, prefix=''):
  """Computes mAP using scikit learn."""
  assert len(gt.shape) == len(pred.shape) == 2, (
      'Expecting B x N_classes both in prediction and gt (one-hot encoding)')
  ap = [
      sklearn.metrics.average_precision_score(gt[:, c], pred[:, c])
      for c in range(gt.shape[1])
  ]
  auc = [
      sklearn.metrics.roc_auc_score(gt[:, c], pred[:, c])
      for c in range(gt.shape[1])
  ]
  dprime = [
      np.sqrt(2) * scipy.stats.norm.ppf(ras)
      for ras in auc
  ]
  return {prefix + 'mAP': sum(ap) / len(ap),
          prefix + 'AUC': sum(auc) / len(auc),
          prefix + 'd-prime': sum(dprime) / len(dprime)}


def compute_accuracy_metrics(pred, gt, prefix=''):
  order_pred = np.argsort(pred, axis=1)
  assert len(gt.shape) == len(order_pred.shape) == 2
  top1_pred = order_pred[:, -1:]
  top5_pred = order_pred[:, -5:]
  top1_acc = np.mean(top1_pred == gt)
  top5_acc = np.mean(np.max(top5_pred == gt, 1))
  return {prefix + 'top1': top1_acc,
          prefix + 'top5': top5_acc}


def compute_retrieval_metrics(x, prefix=''):
  sx = np.argsort(-x, axis=1)
  gt = np.arange(len(x))
  ind = np.where(sx == gt[:, None])[1]
  return {
      prefix + 'R1': float(np.sum(ind == 0)) / len(ind),
      prefix + 'R5': float(np.sum(ind < 5)) / len(ind),
      prefix + 'R10': float(np.sum(ind < 10)) / len(ind),
      prefix + 'MedianRank': np.median(ind) + 1,
  }


def normalize_fn(x, axis=-1, order=2):
  l2 = np.atleast_1d(np.linalg.norm(x, ord=order, axis=axis))
  l2[l2 == 0] = 1
  return x / np.expand_dims(l2, axis=axis)


def compute_similarity_eval(embd,
                            video_embd,
                            n_windows=1,
                            normalize=False,
                            average_similarities=False,
                            average_embeddings=True):
  """Get the similarity between the text embeddings and the video embeddings.

  Args:
    embd: Tensor of embeddings of shape [B, D] where B is the number of
      clips.
    video_embd: Tensor of  embeddings of shape [B * N_windows, D] where B is
      the number of clips, N_windows the number of evaluating windows and D is
      the embedding dimension.
    n_windows: Number of video windows used for evaluation.
    normalize: whether to normalize vectors to unit norm before computing sim.
    average_similarities: whether to average similarities over windows.
    average_embeddings: whether to average embeddings over windows before
      calculating similarity.


  Returns:
    similarity: a [B, B] (or [B, B*N_windows]) tensor, with the similarity of
    each sentence and the different windows in the video.
  """

  if n_windows != 1:
    assert average_embeddings or average_similarities, (
        'for n_windows > 1 at least one of embeddings or '
        'similarities should be averaged'
        )
    assert not (average_embeddings and average_similarities), (
        'either embeddings or similarities could be averaged, not both'
        )

  if normalize and not average_embeddings:
    embd = normalize_fn(embd)  # (b, d)
    video_embd = normalize_fn(video_embd)  # (b*n, d)

  if n_windows == 1:
    similarity = np.matmul(embd, video_embd.T)  # (b, b)
    return similarity

  elif average_similarities:
    similarity = np.matmul(embd, video_embd.T)  # (b, b*n)
    similarity = np.reshape(similarity,
                            [similarity.shape[0], -1, n_windows])  # (b, b, n)
    similarity = similarity.mean(axis=2)  # (b, b)
    return similarity

  elif average_embeddings:
    # (b, n, d)
    video_embd = np.reshape(video_embd, [embd.shape[0], n_windows, -1])
    video_embd = video_embd.mean(axis=1)  # (b, d)
    if normalize:
      embd = normalize_fn(embd)  # (b, d)
      video_embd = normalize_fn(video_embd)  # (b, d)
    similarity = np.matmul(embd, video_embd.T)  # (b, b)
    return similarity
