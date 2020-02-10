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

# Lint as: python3
"""Script to evaluate model predictions against the ground truth."""
import glob
import os

from absl import app
from absl import flags
import numpy as np
from PIL import Image
import tensorflow.compat.v2 as tf
from dual_pixels.eval import get_metrics

flags.DEFINE_string('test_dir', 'test/', 'Path to test dataset.')
flags.DEFINE_string('prediction_dir', 'model_prediction/',
                    'Path to model predictions.')

FLAGS = flags.FLAGS

# Crop over which we do evaluation.
CROP_HEIGHT = 512
CROP_WIDTH = 384


def get_captures():
  """Gets a list of captures."""
  depth_dir = os.path.join(FLAGS.test_dir, 'merged_depth')
  return [
      name for name in os.listdir(depth_dir)
      if os.path.isdir(os.path.join(depth_dir, name))
  ]


def load_capture(capture_name):
  """Loads the ground truth depth, confidence and prediction for a capture."""
  # Assume that we are loading the center capture.
  # Load GT Depth.
  depth_dir = os.path.join(FLAGS.test_dir, 'merged_depth')
  gt_depth_path = glob.glob(
      os.path.join(depth_dir, capture_name, '*_center.png'))[0]
  gt_depth = Image.open(gt_depth_path)
  gt_depth = np.asarray(gt_depth, dtype=np.float32) / 255.0
  # Load GT Depth confidence.
  depth_conf_dir = os.path.join(FLAGS.test_dir, 'merged_conf')
  gt_depth_conf_path = glob.glob(
      os.path.join(depth_conf_dir, capture_name, '*_center.npy'))[0]
  gt_depth_conf = np.load(gt_depth_conf_path)
  # Load prediction.
  prediction_path = glob.glob(
      os.path.join(FLAGS.prediction_dir, capture_name + '.npy'))[0]
  prediction = np.load(prediction_path)
  return prediction, gt_depth, gt_depth_conf


def main(argv):
  del argv  # Unused.
  tf.enable_v2_behavior()
  captures = get_captures()
  loss_dict = {'wmae': [], 'wrmse': [], 'spearman': []}
  for capture in captures:
    print(capture)
    pred, depth_gt, conf_gt = load_capture(capture)
    losses = get_metrics.metrics(pred, depth_gt, conf_gt, CROP_HEIGHT,
                                 CROP_WIDTH)
    for loss_name, loss in loss_dict.items():
      loss.append(losses[loss_name].numpy())
  for loss_name, loss in loss_dict.items():
    loss_dict[loss_name] = np.mean(loss)
  print(loss_dict)


if __name__ == '__main__':
  app.run(main)
