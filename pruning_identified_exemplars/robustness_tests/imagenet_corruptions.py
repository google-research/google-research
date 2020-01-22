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
r"""Evaluate robustness of sparse models on PIE, ImageNet-A and ImageNet-C.

run locally:

"""

import os
import time

from absl import app
from absl import flags

import pandas as pd
import tensorflow.compat.v1 as tf
from pruning_identified_exemplars.utils import model_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("master", "", "Master job.")
flags.DEFINE_integer(
    "target_step", 32000,
    "The last step of training. Target values at this step will be recorded.")

flags.DEFINE_string("ckpt", "", "Ckpt to extract predictions from.")
flags.DEFINE_string("mode", "eval", "Mode designated as train or eval.")
flags.DEFINE_integer(
    "num_label_classes", default=1000, help="Number of classes, at least 2")
flags.DEFINE_integer(
    "num_eval_images", default=100, help="Size of evaluation data set.")
flags.DEFINE_float("label_smoothing", 0.1,
                   "Relax confidence in the labels by (1-label_smoothing).")
flags.DEFINE_integer(
    "iterations_per_loop",
    default=1251,
    help=(
        "Number of steps to run on GPU before outfeeding metrics to the CPU."))
flags.DEFINE_string("dest_dir", "", "File to save the csv data to.")
flags.DEFINE_string("data_directory", "", "File to save the csv data to.")
flags.DEFINE_string("output_path", "", "File to save the csv data to.")
flags.DEFINE_string("corruption_type", "frosted_glass_blur_3",
                    "The location of the sstable used for training.")
flags.DEFINE_enum("dataset_name", "imagenet_a", ("imagenet_a", "imagenet_c"),
                  "What dataset_name is the model trained on.")

# set this flag to true to do a test run of this code with synthetic data
flags.DEFINE_bool("test_small_sample", False,
                  "Boolean for whether to test internally.")

imagenet_params = {
    "train_batch_size": 4096,
    "num_train_images": 1281167,
    "num_eval_images": 50000,
    "num_label_classes": 1000,
    "num_train_steps": 32000,
    "eval_batch_size": 1024,
    "mean_rgb": [0.485 * 255, 0.456 * 255, 0.406 * 255],
    "stddev_rgb": [0.229 * 255, 0.224 * 255, 0.225 * 255]
}


def read_all_eval_subdir(ckpt_dir, dataset_name, directory_path,
                         corruption_type, global_step, dest_dir, params):
  """Get metrics from many subdirectories."""

  split = ckpt_dir.split("/")
  pruning_method = split[11]

  # we pass the updated eval and train string to the params dictionary.

  if pruning_method == "baseline":
    params["pruning_method"] = None
  else:
    params["pruning_method"] = pruning_method

  if dataset_name == "imagenet_c":
    directory_path = os.path.join(directory_path, corruption_type, "3.0.1")
    params["data_dir"] = os.path.join(directory_path, "imagenet2012_*")
  elif dataset_name == "imagenet_a":
    params["data_dir"] = os.path.join(directory_path, "validation*")
  else:
    raise ValueError("dataset not found")

  print(params["data_dir"])
  params["dataset"] = dataset_name
  params["task"] = "robustness_" + dataset_name
  params["train_dir"] = ckpt_dir
  params["is_training"] = False
  params["sloppy_shuffle"] = True

  eval_metrics = model_utils.initiate_task_helper(
      ckpt_directory=ckpt_dir, model_params=params, pruning_params=None)
  print(eval_metrics)

  df = pd.DataFrame.from_dict(eval_metrics, orient="index").reset_index()
  df["exp"] = split[8]
  df["split"] = split[9]
  df["pruning_method"] = split[11]
  df["fraction_pruned"] = split[12]
  df["start_pruning_step"] = split[13]
  df["end_pruning_step"] = split[14]
  df["pruning_frequency"] = split[15]
  df["global_step"] = global_step
  timestamp = str(time.time())
  if dataset_name == "imagenet_c":
    df["corruption"] = corruption_type
    df["corruption_intensity"] = int(corruption_type[-1])
    filename = "{}_{}_{}.csv".format(corruption_type, str(split[8]), timestamp)
  else:
    filename = "{}_{}.csv".format(str(split[8]), timestamp)

  if not tf.gfile.IsDirectory(dest_dir):
    tf.gfile.MakeDirs(dest_dir)

  output_dir = os.path.join(dest_dir, filename)
  with tf.gfile.Open(output_dir, "w") as f:
    tf.logging.info("outputting to csv now.")
    df.to_csv(f)


def main(argv):
  del argv  # Unused.

  params = imagenet_params
  if FLAGS.mode == "eval":
    params["batch_size"] = params["eval_batch_size"]
  else:
    params["batch_size"] = params["train_batch_size"]
  params["num_eval_images"] = FLAGS.num_eval_images
  model_params = {
      "mode": "eval",
      "sloppy_shuffle": True,
      "num_cores": 8,
      "lr_schedule": [(1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)],
      "momentum": 0.9,
      "data_format": "channels_last",
      "output_dir": FLAGS.output_path,
      "label_smoothing": FLAGS.label_smoothing,
      "num_label_classes": FLAGS.num_label_classes,
      "iterations_per_loop": FLAGS.iterations_per_loop,
      "master": FLAGS.master,
      "base_learning_rate": 0.1,
      "weight_decay": 1e-4,
  }
  params.update(model_params)

  if FLAGS.test_small_sample:
    update_params = {"num_images": 2, "num_eval_images": 10, "batch_size": 2}
    params["test_small_sample"] = True
    params.update(update_params)
  else:
    params["test_small_sample"] = False

  global_step = int(os.path.basename(FLAGS.ckpt).split("-")[1])

  read_all_eval_subdir(
      directory_path=FLAGS.data_directory,
      dataset_name=FLAGS.dataset_name,
      ckpt_dir=FLAGS.ckpt,
      corruption_type=FLAGS.corruption_type,
      global_step=global_step,
      dest_dir=FLAGS.dest_dir,
      params=params)


if __name__ == "__main__":
  app.run(main)
