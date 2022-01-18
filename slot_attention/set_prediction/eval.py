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

"""Script for evaluation of a trained set prediction model on CLEVR."""
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import slot_attention.data as data_utils
import slot_attention.model as model_utils
import slot_attention.utils as utils

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir", "/tmp/set_prediction/",
                    "Path to model checkpoint.")
flags.DEFINE_integer("batch_size", 500, "Batch size for the model.")
flags.DEFINE_integer("num_slots", 10, "Number of slots in Slot Attention.")
flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
flags.DEFINE_bool("full_eval", False,
                  "If True, use full evaluation set, otherwise a single batch.")


def load_model():
  """Load the latest checkpoint."""
  # Build the model.
  model = model_utils.build_model(
      resolution=(128, 128), batch_size=FLAGS.batch_size,
      num_slots=FLAGS.num_slots, num_iterations=FLAGS.num_iterations,
      model_type="set_prediction")
  # Load the weights.
  ckpt = tf.train.Checkpoint(network=model)
  ckpt_manager = tf.train.CheckpointManager(
      ckpt, directory=FLAGS.checkpoint_dir, max_to_keep=5)
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    logging.info("Restored from %s", ckpt_manager.latest_checkpoint)
  else:
    raise ValueError("Failed to load checkpoint.")
  return model


def run_eval(model, data_iterator):
  """Run evaluation."""

  if FLAGS.full_eval:  # Evaluate on the full validation set.
    num_eval_batches = 15000 // FLAGS.batch_size
  else:
    # By default, we only test on a single batch for faster evaluation.
    num_eval_batches = 1

  outs = None
  for _ in tf.range(num_eval_batches):
    batch = next(data_iterator)
    if outs is None:
      outs = model(batch["image"], training=False)
      target = batch["target"]
    else:
      new_outs = model(batch["image"], training=False)
      outs = tf.concat([outs, new_outs], axis=0)
      target = tf.concat([target, batch["target"]], axis=0)
  logging.info("Finished getting model predictions.")

  # Compute the AP score.
  ap = [
      utils.average_precision_clevr(outs, target, d)
      for d in [-1., 1., 0.5, 0.25, 0.125]
      ]

  return ap


def main(argv):
  del argv
  model = load_model()
  dataset = data_utils.build_clevr_iterator(
      batch_size=FLAGS.batch_size, split="validation", resolution=(128, 128))
  ap = run_eval(model, dataset)
  logging.info(
      "AP@inf: %.2f, AP@1: %.2f, AP@0.5: %.2f, AP@0.25: %.2f, AP@0.125: %.2f.",
      ap[0], ap[1], ap[2], ap[3], ap[4])

if __name__ == "__main__":
  app.run(main)
