# coding=utf-8
# Copyright 2019 The Google Research Authors.
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
r"""Saves predictions for all ImageNet eval shards and ResNet-50 checkpoints.

This script stores individual predictions in a pandas dataframe and outputs
to csv.

"""

import os
import time

from absl import app
from absl import flags
from absl import logging

import pandas as pd
import tensorflow.compat.v1 as tf
from pruning_identified_exemplars.utils import model_utils

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_cores", default=8, help=("Number of cores."))
flags.DEFINE_string("data_directory", "",
                    "The location of the tfrecords used for training.")
flags.DEFINE_integer("batch_size", 1, "Batch size for creating new dataset.")
flags.DEFINE_string("output_path", "/tmp/output/",
                    "Directory path to save the csv data to.")
flags.DEFINE_string("ckpt_dir", "", "Ckpt to extract predictions from.")
flags.DEFINE_enum("mode", "eval", ("eval", "train"),
                  "Mode designated as train or eval.")
flags.DEFINE_float("label_smoothing", 0.1,
                   "Relax confidence in the labels by (1-label_smoothing).")
# set this flag to true to do a test run of this code with synthetic data
flags.DEFINE_bool("test_small_sample", True,
                  "Boolean for whether to test internally.")


imagenet_params = {
    "num_eval_images": 50000,
    "num_label_classes": 1000,
    "batch_size": 1,
    "mean_rgb": [0.485 * 255, 0.456 * 255, 0.406 * 255],
    "stddev_rgb": [0.229 * 255, 0.224 * 255, 0.225 * 255]
}


def predictions_from_checkpoint_dir(directory_path, filename, params,
                                    ckpt_directory, global_step):
  """Outputs predictions as a pandas dataframe.

  Args:
    directory_path: The path to the directory where dataset is stored.
    filename: The shard to retrieve predictions for.
    params: Dictionary of training and eval specific params.
    ckpt_directory: Path to the directory where checkpoint is stored.
    global_step: Training Step at which eval metrics were stored.

  Returns:
    When run on full dataset (test_small_sample=False) returns a pandas
    dataframe with predictions for all images on specified shard.

  Raises:
    ValueError when checkpoint is not stored in the correct format.
  """

  split = ckpt_directory.split("/")
  pruning_method = split[11]
  # Assert statement to catch if ckp path is saved in correct way.
  if pruning_method not in ["threshold", "baseline"]:
    raise ValueError("Pruning method is not known %s" % (pruning_method))

  # We pass the updated eval and train string to the params dictionary.
  params["output_dir"] = ckpt_directory
  if pruning_method == "baseline":
    params["pruning_method"] = None
  else:
    params["pruning_method"] = pruning_method
  params["train_dir"] = ckpt_directory
  params["data_dir"] = directory_path
  params["split"] = filename
  params["is_training"] = False
  params["task"] = "imagenet_predictions"
  if FLAGS.test_small_sample:
    update_params = {
        "num_eval_images": 10,
    }
    params["test_small_sample"] = True
    params.update(update_params)
  else:
    params["test_small_sample"] = False

  predictions = model_utils.initiate_task_helper(
      ckpt_directory=ckpt_directory, model_params=params)
  if not FLAGS.test_small_sample:
    df = pd.DataFrame.from_records(list(predictions))
    df["exp"] = split[8]
    df["split"] = split[9]
    df["filename"] = filename
    df["pruning_method"] = split[11]
    df["fraction_pruned"] = split[12]
    df["start_pruning_step"] = split[13]
    df["end_pruning_step"] = split[14]
    df["pruning_frequency"] = split[15]
    df["global_step"] = global_step
    return df


def main(argv):
  del argv  # Unused.

  if FLAGS.mode == "eval":
    file_path = "val*"
  else:
    file_path = "train*"

  if FLAGS.test_small_sample:
    filenames = []
    data_directory = FLAGS.data_directory
  else:
    data_directory = os.path.join(FLAGS.data_directory, file_path)


    filenames = tf.io.gfile.glob(data_directory)

  ckpt_directory = FLAGS.ckpt_dir

  params = imagenet_params
  # shuffle is set to false to prevent output ordering of images
  update_params = {
      "mode": FLAGS.mode,
      "sloppy_shuffle": False,
      "num_cores": 8,
      "base_learning_rate": 0.1,
      "weight_decay": 1e-4,
      "lr_schedule": [(1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)],
      "momentum": 0.9,
      "data_format": "channels_last",
      "label_smoothing": FLAGS.label_smoothing,
  }
  params.update(update_params)
  global_step = int(os.path.basename(ckpt_directory).split("-")[1])

  if FLAGS.test_small_sample:
    logging.info(
        "test_small_sample is set to True, ignoring data directory."
    )
    params["output_dir"] = FLAGS.output_path
    params["num_eval_images"] = 10
    df = predictions_from_checkpoint_dir(
        directory_path=data_directory,
        filename=filenames,
        params=params,
        ckpt_directory=ckpt_directory,
        global_step=global_step)
    logging.info("testing workflow complete")
  else:
    shard_count = 0
    for filename in sorted(filenames):
      shard = os.path.basename(filename)
      dest_dir = os.path.join(FLAGS.output_path, "imagenet",
                              "predictions_dataframe", FLAGS.mode, shard)
      if not tf.gfile.IsDirectory(dest_dir):
        tf.gfile.MkDir(dest_dir)

      params["output_dir"] = dest_dir

      df = predictions_from_checkpoint_dir(
          directory_path=data_directory,
          filename=filename,
          params=params,
          ckpt_directory=ckpt_directory,
          global_step=global_step)
      timestamp = str(time.time())
      output_path = os.path.join(
          dest_dir, "predictions_dataframe_{}.csv".format(timestamp))
      with tf.gfile.Open(output_path, "w") as f:
        df.to_csv(f)
      shard_count += 1
      print("number of shards processed: ", shard_count)


if __name__ == "__main__":
  app.run(main)
