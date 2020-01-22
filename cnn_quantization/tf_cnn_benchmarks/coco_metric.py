# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""COCO-style evaluation metrics.

Forked from reference model implementation.

COCO API: github.com/cocodataset/cocoapi/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import tempfile

from absl import flags

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import six

import tensorflow.compat.v1 as tf

from cnn_quantization.tf_cnn_benchmarks import mlperf
from cnn_quantization.tf_cnn_benchmarks import ssd_constants

FLAGS = flags.FLAGS


# https://github.com/cocodataset/cocoapi/issues/49
if six.PY3:
  import pycocotools.coco
  pycocotools.coco.unicode = str


def async_eval_runner(queue_predictions, queue_results, val_json_file):
  """Load intermediate eval results and get COCO metrics."""
  while True:
    message = queue_predictions.get()
    if message == 'STOP':  # poison pill
      break
    step, predictions = message
    results = compute_map(predictions, val_json_file)
    queue_results.put((step, results))


def compute_map(predictions, val_json_file):
  """Use model predictions to compute mAP.

  Args:
    predictions: a list of tuples returned by decoded_predictions function,
      each containing the following elements:
      image source_id, box coordinates in XYWH order, probability score, label
    val_json_file: path to COCO annotation file
  Returns:
    A dictionary that maps all COCO metrics (keys) to their values
  """

  if val_json_file.startswith("gs://"):
    _, local_val_json = tempfile.mkstemp(suffix=".json")
    tf.gfile.Remove(local_val_json)

    tf.gfile.Copy(val_json_file, local_val_json)
    atexit.register(tf.gfile.Remove, local_val_json)
  else:
    local_val_json = val_json_file

  cocoGt = COCO(local_val_json)
  cocoDt = cocoGt.loadRes(np.array(predictions))
  E = COCOeval(cocoGt, cocoDt, iouType='bbox')
  E.evaluate()
  E.accumulate()
  E.summarize()
  print("Current AP: {:.5f}".format(E.stats[0]))
  metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                  'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']

  # Prefix with "COCO" to group in TensorBoard.
  return {"COCO/" + key: value for key, value in zip(metric_names, E.stats)}


def calc_iou(target, candidates):
  target_tiled = np.tile(target[np.newaxis, :], (candidates.shape[0], 1))
  # Left Top & Right Bottom
  lt = np.maximum(target_tiled[:,:2], candidates[:,:2])

  rb = np.minimum(target_tiled[:,2:], candidates[:,2:])

  delta = np.maximum(rb - lt, 0)

  intersect = delta[:,0] * delta[:,1]

  delta1 = target_tiled[:,2:] - candidates[:,:2]
  area1 = delta1[:,0] * delta1[:,1]
  delta2 = target_tiled[:,2:] - candidates[:,:2]
  area2 = delta2[:,0] * delta2[:,1]

  iou = intersect/(area1 + area2 - intersect)
  return iou


# TODO(haoyuzhang): Rewrite this NumPy based implementation to TensorFlow based
# implementation under ssd_model.py accuracy_function.
def decode_predictions(labels_and_predictions):
  """Decode predictions and remove unused boxes and labels."""
  predictions = []
  for example in labels_and_predictions:
    source_id = int(example[ssd_constants.SOURCE_ID])
    pred_box = example[ssd_constants.PRED_BOXES]
    pred_scores = example[ssd_constants.PRED_SCORES]

    locs, labels, probs = decode_single(
        pred_box, pred_scores, ssd_constants.OVERLAP_CRITERIA,
        ssd_constants.MAX_NUM_EVAL_BOXES, ssd_constants.MAX_NUM_EVAL_BOXES)

    raw_height, raw_width, _ = example[ssd_constants.RAW_SHAPE]
    for loc, label, prob in zip(locs, labels, probs):
      # Ordering convention differs, hence [1], [0] rather than [0], [1]
      x, y = loc[1] * raw_width, loc[0] * raw_height
      w, h = (loc[3] - loc[1]) * raw_width, (loc[2] - loc[0]) * raw_height
      predictions.append(
          [source_id, x, y, w, h, prob, ssd_constants.CLASS_INV_MAP[label]])
  mlperf.logger.log(key=mlperf.tags.NMS_THRESHOLD,
                    value=ssd_constants.OVERLAP_CRITERIA)
  mlperf.logger.log(key=mlperf.tags.NMS_MAX_DETECTIONS,
                    value=ssd_constants.MAX_NUM_EVAL_BOXES)
  return predictions


def decode_single(bboxes_in, scores_in, criteria, max_output, max_num=200):
  # Reference to https://github.com/amdegroot/ssd.pytorch

  bboxes_out = []
  scores_out = []
  labels_out = []

  for i, score in enumerate(np.split(scores_in, scores_in.shape[1], 1)):
    score = np.squeeze(score, 1)

    # skip background
    if i == 0:
      continue

    mask = score > ssd_constants.MIN_SCORE
    if not np.any(mask):
      continue

    bboxes, score = bboxes_in[mask, :], score[mask]

    score_idx_sorted = np.argsort(score)
    score_sorted = score[score_idx_sorted]

    score_idx_sorted = score_idx_sorted[-max_num:]
    candidates = []

    # perform non-maximum suppression
    while len(score_idx_sorted):
      idx = score_idx_sorted[-1]
      bboxes_sorted = bboxes[score_idx_sorted, :]
      bboxes_idx = bboxes[idx, :]
      iou = calc_iou(bboxes_idx, bboxes_sorted)

      score_idx_sorted = score_idx_sorted[iou < criteria]
      candidates.append(idx)

    bboxes_out.append(bboxes[candidates, :])
    scores_out.append(score[candidates])
    labels_out.extend([i]*len(candidates))

  if len(scores_out) == 0:
    tf.logging.info("No objects detected. Returning dummy values.")
    return (
        np.zeros(shape=(1, 4), dtype=np.float32),
        np.zeros(shape=(1,), dtype=np.int32),
        np.ones(shape=(1,), dtype=np.float32) * ssd_constants.DUMMY_SCORE,
    )

  bboxes_out = np.concatenate(bboxes_out, axis=0)
  scores_out = np.concatenate(scores_out, axis=0)
  labels_out = np.array(labels_out)

  max_ids = np.argsort(scores_out)[-max_output:]

  return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]
