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
r"""Scrape class summaries from tensorboard event summaries and save as CSV.

This script scrapes class summaries and stores individual predictions
for all ImageNet shards given a single checkpoint.


"""

import time

from absl import app
from absl import flags

import pandas as pd
import tensorflow.compat.v1 as tf
from pruning_identified_exemplars.utils import model_utils

flags.DEFINE_integer(
    "target_step", 32000,
    "The last step of training. Target values at this step will be recorded.")
flags.DEFINE_string("event_files_pattern", "*tfevents*",
                    "Pattern to look for in event files file names.")
flags.DEFINE_string("output_path", "", "Path to save the csv data to.")
flags.DEFINE_string("ckpt_dir", "", "Checkpoint to extract predictions from.")
flags.DEFINE_string("data_directory", "",
                    "The location of the tfrecords used for training.")
# set this flag to true to do a test run of this code with synthetic data
flags.DEFINE_bool("test_small_sample", True,
                  "Boolean for whether to test internally.")
FLAGS = flags.FLAGS

imagenet_params = {
    "num_eval_images": 50000,
    "num_label_classes": 1000,
    "batch_size": 1,
    "mean_rgb": [0.485 * 255, 0.456 * 255, 0.406 * 255],
    "stddev_rgb": [0.229 * 255, 0.224 * 255, 0.225 * 255]
}


def tags_from_checkpoint_dir(ckpt_directory, params):
  """Get metrics from event file.

  Args:
    ckpt_directory: model checkpoint directory containing event file.
    params: dictionary of params for model training and eval.

  Returns:
    pd.DataFrame containing metrics from event file
  """

  split = ckpt_directory.split("/")
  eval_metrics = model_utils.initiate_task_helper(
      model_params=params, ckpt_directory=ckpt_directory)
  df = pd.DataFrame.from_dict(eval_metrics, orient="index").reset_index()

  df["exp"] = split[10]
  df["pruning_method"] = split[11]
  df["fraction_pruned"] = split[12]
  df["start_pruning_step"] = split[13]
  df["end_pruning_step"] = split[14]
  df["pruning_frequency"] = split[15]
  df["ckpt"] = ckpt_directory
  return df


def main(argv):
  del argv  # Unused.

  output_path = FLAGS.output_path + "_tags"
  params = imagenet_params
  split = FLAGS.ckpt_dir.split("/")
  params["task"] = "ckpt_prediction"
  # shuffle is set to false to prevent output ordering of images
  update_params = {
      "sloppy_shuffle": False,
      "data_dir": FLAGS.data_directory,
      "pruning_method": split[11],
      "num_cores": 8,
      "base_learning_rate": 0.1,
      "weight_decay": 1e-4,
      "lr_schedule": [(1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)],
      "momentum": 0.9,
      "data_format": "channels_last",
      "output_dir": output_path,
      "label_smoothing": 0.1,
  }
  params.update(update_params)

  if FLAGS.test_small_sample:
    update_params = {
        "num_eval_images": 10,
    }
    params["test_small_sample"] = True
    params.update(update_params)
  else:
    params["test_small_sample"] = False
  df = tags_from_checkpoint_dir(ckpt_directory=FLAGS.ckpt_dir, params=params)

  # this allows for unique naming of each output file
  timestamp = str(time.time())
  with tf.gfile.Open(output_path + "_" + timestamp + ".csv", "w") as f:
    df.to_csv(f)


if __name__ == "__main__":
  app.run(main)
