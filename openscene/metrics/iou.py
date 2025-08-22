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
"""Evaluation metric."""

from dataset.scannet200_constants import CLASS_LABELS_20
from dataset.scannet200_constants import MATTERPORT_LABELS_21
from dataset.scannet200_constants import MATTERPORT_LABELS_NYU160
from dataset.scannet200_constants import MATTERPORT_LABELS_NYU40
from dataset.scannet200_constants import MATTERPORT_LABELS_NYU80
from dataset.scannet200_constants import NUSCENES_LABELS_16
import numpy as np

UNKNOWN_ID = 255
NO_FEATURE_ID = 256


def confusion_matrix(pred_ids, gt_ids, num_classes):
  """Calculate confusion matrix."""
  assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
  idxs = gt_ids != UNKNOWN_ID
  # some points have no feature assigned for prediction
  if NO_FEATURE_ID in pred_ids:
    pred_ids[pred_ids == NO_FEATURE_ID] = num_classes
    confusion = np.bincount(
        pred_ids[idxs] * (num_classes + 1) + gt_ids[idxs],
        minlength=(num_classes + 1)**2).reshape(
            (num_classes + 1, num_classes + 1)).astype(np.ulonglong)
    return confusion
  else:
    return np.bincount(
        pred_ids[idxs] * num_classes + gt_ids[idxs],
        minlength=num_classes**2).reshape(
            (num_classes, num_classes)).astype(np.ulonglong)


def get_iou(label_id, confusion):
  """Calculate IoU."""
  # true positives
  tp = np.longlong(confusion[label_id, label_id])
  # false positives
  fp = np.longlong(confusion[label_id, :].sum()) - tp
  # false negatives
  fn = np.longlong(confusion[:, label_id].sum()) - tp

  denom = (tp + fp + fn)
  if denom == 0:
    return float('nan')
  return float(tp) / denom, tp, denom


def evaluate(pred_ids, gt_ids, stdout=False, dataset='scannet_3d'):
  """Evaluation metric calculation."""
  if stdout:
    print('evaluating', gt_ids.size, 'points...')
  if 'scannet_3d' in dataset:
    class_labels = CLASS_LABELS_20
  elif 'matterport_3d' in dataset:
    class_labels = MATTERPORT_LABELS_21
  elif 'matterport_nyu40_3d' in dataset:
    class_labels = MATTERPORT_LABELS_NYU40
  elif 'matterport_nyu80_3d' in dataset:
    class_labels = MATTERPORT_LABELS_NYU80
  elif 'matterport_nyu160_3d' in dataset:
    class_labels = MATTERPORT_LABELS_NYU160
  elif 'nuscenes_3d' in dataset:
    class_labels = NUSCENES_LABELS_16
  else:
    raise NotImplementedError

  n_classes = len(class_labels)
  confusion = confusion_matrix(pred_ids, gt_ids, n_classes)
  class_ious = {}
  mean_iou = 0
  class_accs = {}
  mean_acc = 0
  count = 0
  for i in range(n_classes):
    label_name = class_labels[i]

    if (gt_ids == i).sum(
    ) == 0:  # at least 1 point needs to be in the evaluation for this class
      continue

    class_ious[label_name] = get_iou(i, confusion)
    class_accs[label_name] = class_ious[label_name][1] / (gt_ids == i).sum()
    count += 1

    mean_iou += class_ious[label_name][0]
    mean_acc += class_accs[label_name]

  mean_iou /= n_classes
  mean_acc /= n_classes
  if stdout:
    print('classes          IoU')
    print('----------------------------')
    for i in range(n_classes):
      label_name = class_labels[i]
      if 'matterport' in dataset:
        print('{0:<14s}: {1:>5.3f}'.format(label_name, class_accs[label_name]))
      else:
        print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(
            label_name, class_ious[label_name][0], class_ious[label_name][1],
            class_ious[label_name][2]))
        print(label_name + ' error!')
        continue
    print('Mean IoU', mean_iou)
    print('Mean Acc', mean_acc)
  return mean_iou
