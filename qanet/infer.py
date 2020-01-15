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

r"""Script to load a saved model and run inference on it for SQuAD.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from absl import flags
import tensorflow.compat.v1 as tf

from tensorflow.python.framework import errors
# Need to import these modules so the models are registered and accessible
from qanet.google.old import squad  # pylint: disable=unused-import
from qanet.util import configurable
from qanet.util import evaluator_util
from qanet.util import misc_util
from qanet.google.old import squad_prepro

flags.DEFINE_string("input_dir", "", "Directory to load data from")
flags.DEFINE_string("checkpoint_dir", "",
                    "Directory to load model from. "
                    "One of saved_model or checkpoint must be given")
flags.DEFINE_string("glove_dir", "", "Directory to find GLOVE embeddings")
flags.DEFINE_string("output_dir", "", "Path to save predictions")
flags.DEFINE_string("groundtruth_path", "", "Json path to groundtruths")
flags.DEFINE_string("split", "dev", "Split name in input_dir")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("num_gpus", 0, "number of gpus")
flags.DEFINE_boolean("include_probabilities", True,
                     "Whether to dump start/end probabilities")

# NOTE(thangluong): Google-internal info on TGQ
flags.DEFINE_string("data_format", "squad",
                    "\'squad\' (default), \'tgq\', or \'squad2\' (Squad 2.0)")

FLAGS = flags.FLAGS


def _get_data(split="dev", batch_size=1):
  """Load data."""
  print(FLAGS.input_dir)
  print(FLAGS.glove_dir)
  input_fn = squad_prepro.get_input_fn(
      root_data_dir=FLAGS.input_dir,
      glove_dir=FLAGS.glove_dir,
      data_type=split,
      batch_size=batch_size,
      # glove_size=300,
      shuffle_files=False,
      shuffle_examples=False,
      queue_capacity=1000,
      min_after_dequeue=1,
      num_epochs=1)
  features, _ = input_fn()
  return features


def infer(output_path):
  """Run inference."""
  tf.logging.info("Run inference")
  checkpoint = FLAGS.checkpoint_dir

  with tf.gfile.GFile(os.path.join(checkpoint, "config.json")) as f:
    config = json.load(f)
  if FLAGS.num_gpus > 1:
    config["num_gpus"] = FLAGS.num_gpus
  tf.logging.info("# infer config %s", config)

  # Load the data
  tf.logging.info("Loading data")
  features = _get_data(split=FLAGS.split, batch_size=FLAGS.batch_size)
  tf.logging.info("Loading model")
  model_class = configurable.Configurable.load(config["model"])
  model = model_class("eval", config=config["model"])
  outputs = model(features)
  tf.logging.info(outputs)

  sess = tf.Session(
      config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False))
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  sess.run(tf.tables_initializer())
  tf.logging.info(config["model"]["optimizer"])

  # Get trainable / frozen vars
  trainable_vars, frozen_vars, _ = misc_util.get_trainable_vars(
      all_vars=tf.global_variables(),
      exclude_pattern=config["model"]["optimizer"]["nograd_var"])

  # Make sure to load in the exponential moving average vars
  ema = tf.train.ExponentialMovingAverage(decay=0.9999)
  ema_vars = {}
  for var in trainable_vars:
    ema_vars[ema.average_name(var)] = var

  # Restoring EMA trainable vars
  tf.logging.info("Restoring ema.")
  saver = tf.train.Saver(ema_vars)
  ckpt_path = tf.train.latest_checkpoint(checkpoint)
  saver.restore(sess, ckpt_path)

  # Restoring frozen vars
  tf.logging.info("Restoring frozen.")
  saver_frozen = tf.train.Saver(frozen_vars)
  saver_frozen.restore(sess, ckpt_path)

  # Setup scaffolding and load in the saved model
  coord = tf.train.Coordinator()
  _ = tf.train.start_queue_runners(coord=coord, sess=sess)

  is_tgq = FLAGS.data_format == "tgq"
  result = {}
  try:
    i = 0
    while True:
      predictions = sess.run(outputs)
      for qid, answer, start, end in zip(predictions["id"], predictions["a"],
                                         predictions["p1"], predictions["p2"]):
        if FLAGS.include_probabilities or is_tgq:
          output = {"answer": answer}
          output["start_prob"] = list([float(x) for x in start])
          output["end_prob"] = list([float(x) for x in end])
          if is_tgq:
            start, end, _ = evaluator_util.get_start_end(
                output["start_prob"], output["end_prob"])
            output["start"] = start
            output["end"] = end
        else:
          output = answer

        result[qid] = output

      if i % 100 == 0:
        tf.logging.info(i)
      i += 1
  except errors.OutOfRangeError:
    pass

  # Dump results to a file
  with tf.gfile.GFile(output_path, "w") as f:
    if is_tgq:
      for qid in result:
        # NOTE(thangluong): from chrisalberti@'s observation,
        #   we need to subtract 1 to get good results.
        #   To investigate; we could have added bos (found no evidence yet).
        start = result[qid]["start"] - 1
        end = result[qid]["end"] - 1
        f.write("%s\t-1\t%d\t%d\n" % (qid, start, end))
    else:
      json.dump(result, f)


def main(_):
  is_tgq = FLAGS.data_format == "tgq"
  if is_tgq:
    output_path = os.path.join(FLAGS.output_dir, "infer.txt")
  else:
    output_path = os.path.join(FLAGS.output_dir, "infer.json")

  # Inference
  if not tf.gfile.Exists(output_path):
    infer(output_path)
  else:
    tf.logging.info("Output file %s exists" % output_path)

  # Evaluation
  if FLAGS.groundtruth_path and not is_tgq:
    scores = evaluator_util.evaluate(
        FLAGS.groundtruth_path, output_path, data_format=FLAGS.data_format)
    tf.logging.info(scores)


if __name__ == "__main__":
  tf.app.run(main)
