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

"""Train and eval for the seq2act estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import trainer_lib
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from seq2act.models import input as input_utils
from seq2act.models import seq2act_estimator

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("exp_mode", "train", "the running mode")
flags.DEFINE_string("eval_name", "", "the eval name")
flags.DEFINE_string("train_file_list", None, "the list of train files")
flags.DEFINE_string("train_source_list", None, "the list of train sources")
flags.DEFINE_string("train_batch_sizes", None, "the list of batch sizes")
flags.DEFINE_string("eval_data_source", "android_howto", "the data source")
flags.DEFINE_string("reference_checkpoint", "",
                    "the reference_checkpoint")
flags.DEFINE_string("hparam_file", "", "the hyper parameter file")
flags.DEFINE_string("experiment_dir", "/tmp",
                    "the directory for output checkpoints")
flags.DEFINE_integer("eval_steps", 150, "eval_steps")
flags.DEFINE_integer("decode_length", 20, "decode_length")
flags.DEFINE_integer("eval_batch_size", 2, "eval_batch_size")
flags.DEFINE_integer("shuffle_size", 2, "shuffle_size")
flags.DEFINE_boolean("boost_input", False, "boost_input")


def continuous_eval(experiment_dir):
  """Evaluate until checkpoints stop being produced."""
  for ckpt_path in trainer_lib.next_checkpoint(experiment_dir,
                                               timeout_mins=-1):
    hparams = seq2act_estimator.load_hparams(experiment_dir)
    hparams.set_hparam("batch_size", FLAGS.eval_batch_size)
    eval_input_fn = seq2act_estimator.create_input_fn(
        FLAGS.eval_files, hparams.batch_size,
        -1, 2,
        input_utils.DataSource.from_str(FLAGS.eval_data_source),
        max_range=hparams.max_span,
        max_dom_pos=hparams.max_dom_pos,
        max_pixel_pos=hparams.max_pixel_pos,
        mean_synthetic_length=hparams.mean_synthetic_length,
        stddev_synthetic_length=hparams.stddev_synthetic_length,
        load_extra=True,
        load_screen=hparams.load_screen,
        load_dom_dist=(hparams.screen_encoder == "gcn"))
    estimator = create_estimator(experiment_dir, hparams,
                                 decode_length=FLAGS.decode_length)
    estimator.evaluate(input_fn=eval_input_fn,
                       steps=FLAGS.eval_steps,
                       checkpoint_path=ckpt_path,
                       name=FLAGS.eval_name)


def create_estimator(experiment_dir, hparams, decode_length=20):
  """Creates an estimator with given hyper parameters."""
  if FLAGS.worker_gpu > 1:
    strategy = tf.distribute.MirroredStrategy()
  else:
    strategy = None
  config = tf_estimator.RunConfig(
      save_checkpoints_steps=1000, save_summary_steps=300,
      train_distribute=strategy)
  model_fn = seq2act_estimator.create_model_fn(
      hparams,
      seq2act_estimator.compute_additional_loss\
      if hparams.use_additional_loss else None,
      seq2act_estimator.compute_additional_metric\
      if hparams.use_additional_loss else None,
      compute_seq_accuracy=True,
      decode_length=decode_length)
  if FLAGS.reference_checkpoint:
    latest_checkpoint = tf.train.latest_checkpoint(
        FLAGS.reference_checkpoint)
    ws = tf_estimator.WarmStartSettings(
        ckpt_to_initialize_from=latest_checkpoint,
        vars_to_warm_start=["embed_tokens/task_embed_w", "encode_decode/.*",
                            "output_layer/.*"])
  else:
    ws = None
  estimator = tf_estimator.Estimator(
      model_fn=model_fn, model_dir=experiment_dir, config=config,
      warm_start_from=ws)
  return estimator


def train(experiment_dir):
  """Trains the model."""
  if FLAGS.hparam_file:
    hparams = seq2act_estimator.load_hparams(FLAGS.hparam_file)
  else:
    hparams = seq2act_estimator.create_hparams()

  estimator = create_estimator(experiment_dir, hparams)
  seq2act_estimator.save_hyperparams(hparams, experiment_dir)
  train_file_list = FLAGS.train_file_list.split(",")
  train_source_list = FLAGS.train_source_list.split(",")
  train_batch_sizes = FLAGS.train_batch_sizes.split(",")
  print("* xm_train", train_file_list, train_source_list, train_batch_sizes)
  if len(train_file_list) > 1:
    train_input_fn = seq2act_estimator.create_hybrid_input_fn(
        train_file_list,
        [input_utils.DataSource.from_str(s) for s in train_source_list],
        map(int, train_batch_sizes),
        max_range=hparams.max_span,
        max_dom_pos=hparams.max_dom_pos,
        max_pixel_pos=hparams.max_pixel_pos,
        mean_synthetic_length=hparams.mean_synthetic_length,
        stddev_synthetic_length=hparams.stddev_synthetic_length,
        batch_size=hparams.batch_size,
        boost_input=FLAGS.boost_input,
        load_screen=hparams.load_screen,
        buffer_size=FLAGS.shuffle_size,
        shuffle_size=FLAGS.shuffle_size,
        load_dom_dist=(hparams.screen_encoder == "gcn"))
  else:
    train_input_fn = seq2act_estimator.create_input_fn(
        train_file_list[0],
        hparams.batch_size,
        -1, -1, input_utils.DataSource.from_str(train_source_list[0]),
        max_range=hparams.max_span,
        max_dom_pos=hparams.max_dom_pos,
        max_pixel_pos=hparams.max_pixel_pos,
        mean_synthetic_length=hparams.mean_synthetic_length,
        stddev_synthetic_length=hparams.stddev_synthetic_length,
        load_extra=False,
        load_screen=hparams.load_screen,
        buffer_size=FLAGS.shuffle_size,
        shuffle_size=FLAGS.shuffle_size,
        load_dom_dist=(hparams.screen_encoder == "gcn"))
  estimator.train(input_fn=train_input_fn, steps=FLAGS.train_steps)


def main(_):
  """The main function."""
  if FLAGS.exp_mode == "train":
    train(FLAGS.experiment_dir)
  elif FLAGS.exp_mode == "eval":
    continuous_eval(FLAGS.experiment_dir)

if __name__ == "__main__":
  tf.app.run()
