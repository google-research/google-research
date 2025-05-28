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

"""Helper functions for model evaluation."""

import numpy as np
import pandas as pd
import sklearn.metrics as skm

import torch


def predict_on_set(algorithm, loader, device):
  """Inference a model on a dataloader and returns predictions, targets, attributes, and groups."""

  num_labels = loader.dataset.num_labels

  ys, atts, gs, ps = [], [], [], []

  algorithm.eval()
  with torch.no_grad():
    for _, x, y, a in loader:
      p = algorithm.predict(x.to(device))
      if p.squeeze().ndim == 1:
        p = torch.sigmoid(p).detach().cpu().numpy()
      else:
        p = torch.softmax(p, dim=-1).detach().cpu().numpy()
        if num_labels == 2:
          p = p[:, 1]

      ps.append(p)
      ys.append(y)
      atts.append(a)
      gs.append([f'y={yi},a={gi}' for _, (yi, gi) in enumerate(zip(y, a))])

  return (
      np.concatenate(ys, axis=0),
      np.concatenate(atts, axis=0),
      np.concatenate(ps, axis=0),
      np.concatenate(gs),
  )


def eval_metrics(algorithm, loader, device, thres=0.5):
  """Evaluates a model on a dataloader and returns a dictionary of metrics."""
  targets, attributes, preds, gs = predict_on_set(algorithm, loader, device)
  preds_rounded = (
      preds >= thres if preds.squeeze().ndim == 1 else preds.argmax(1)
  )
  label_set = np.unique(targets)

  res = {}
  res['overall'] = {
      **binary_metrics(targets, preds_rounded, label_set),
      **prob_metrics(targets, preds, label_set),
  }
  res['per_attribute'] = {}
  res['per_class'] = {}
  res['per_group'] = {}

  for a in np.unique(attributes):
    mask = attributes == a
    res['per_attribute'][int(a)] = {
        **binary_metrics(targets[mask], preds_rounded[mask], label_set),
        **prob_metrics(targets[mask], preds[mask], label_set),
    }

  classes_report = skm.classification_report(
      targets, preds_rounded, output_dict=True, zero_division=0.0
  )
  res['overall']['macro_avg'] = classes_report['macro avg']
  res['overall']['weighted_avg'] = classes_report['weighted avg']
  for y in np.unique(targets):
    res['per_class'][int(y)] = classes_report[str(y)]

  for g in np.unique(gs):
    mask = gs == g
    res['per_group'][g] = {
        **binary_metrics(targets[mask], preds_rounded[mask], label_set)
    }

  res['overall']['adjusted_accuracy'] = sum(
      [res['per_group'][g]['accuracy'] for g in np.unique(gs)]
  ) / len(np.unique(gs))
  res['min_attr'] = pd.DataFrame(res['per_attribute']).min(axis=1).to_dict()
  res['max_attr'] = pd.DataFrame(res['per_attribute']).max(axis=1).to_dict()
  res['min_group'] = pd.DataFrame(res['per_group']).min(axis=1).to_dict()
  res['max_group'] = pd.DataFrame(res['per_group']).max(axis=1).to_dict()

  return res


def binary_metrics(targets, preds, label_set=None, return_arrays=False):
  """Binary classification metrics."""
  if not targets:
    return {}
  if label_set is None:
    label_set = [0, 1]

  res = {
      'accuracy': skm.accuracy_score(targets, preds),
      'n_samples': len(targets),
  }

  if len(label_set) == 2:
    conf_matrix = skm.confusion_matrix(targets, preds, labels=label_set)

    res['TN'] = conf_matrix[0][0].item()
    res['FN'] = conf_matrix[1][0].item()
    res['TP'] = conf_matrix[1][1].item()
    res['FP'] = conf_matrix[0][1].item()
    res['error'] = res['FN'] + res['FP']

    if res['TP'] + res['FN'] == 0:
      res['TPR'] = 0
      res['FNR'] = 1
    else:
      res['TPR'] = res['TP'] / (res['TP'] + res['FN'])
      res['FNR'] = res['FN'] / (res['TP'] + res['FN'])

    if res['FP'] + res['TN'] == 0:
      res['FPR'] = 1
      res['TNR'] = 0
    else:
      res['FPR'] = res['FP'] / (res['FP'] + res['TN'])
      res['TNR'] = res['TN'] / (res['FP'] + res['TN'])

    res['pred_prevalence'] = (res['TP'] + res['FP']) / res['n_samples']
    res['prevalence'] = (res['TP'] + res['FN']) / res['n_samples']
  else:
    # conf_matrix = skm.confusion_matrix(targets, preds, labels=label_set)
    res['TPR'] = skm.recall_score(
        targets, preds, labels=label_set, average='macro', zero_division=0.0
    )

  if len(np.unique(targets)) > 1:
    res['balanced_acc'] = skm.balanced_accuracy_score(targets, preds)

  if return_arrays:
    res['targets'] = targets
    res['preds'] = preds

  return res


def prob_metrics(targets, preds, label_set, return_arrays=False):
  """Probability metrics."""
  if not targets:
    return {}

  res = {
      'BCE': skm.log_loss(targets, preds, eps=1e-6, labels=label_set),
  }

  if return_arrays:
    res['targets'] = targets
    res['preds'] = preds

  return res
