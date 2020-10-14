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

"""Dual encoder SMITH models."""

import json
import os

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

from smith import constants
from smith import experiment_config_pb2
from smith import input_fns
from smith import modeling as smith_modeling
from smith import utils

flags.DEFINE_string("dual_encoder_config_file", None,
                    "The proto config file for dual encoder SMITH models.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_enum(
    "train_mode", None, ["finetune", "pretrain", "joint_train"],
    "Whether it is joint_train, pretrain or finetune. The difference is "
    "about total_loss calculation and input files for eval and training.")

flags.DEFINE_enum(
    "schedule", None, ["train", "continuous_eval", "predict", "export"],
    "The run schedule which can be any one of train, continuous_eval, "
    "predict or export.")

flags.DEFINE_bool("debugging", False,
                  "Print out some information for debugging.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", None, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", None, "Number of warmup steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

FLAGS = flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  train_mode = FLAGS.train_mode
  ############################################################################
  # Load the dual_encoder_config_file file.
  ############################################################################
  if tf.gfile.Exists(FLAGS.dual_encoder_config_file):
    exp_config = utils.load_config_from_file(
        FLAGS.dual_encoder_config_file,
        experiment_config_pb2.DualEncoderConfig())
  else:
    raise ValueError("dual_encoder_config: {} not found!".format(
        FLAGS.dual_encoder_config_file))
  tf.logging.info(">>>> final dual_encoder_config:\n {}".format(exp_config))
  tf.gfile.MakeDirs(FLAGS.output_dir)

  ############################################################################
  # Save/copy the configuration file.
  ############################################################################
  configs_dir = os.path.join(FLAGS.output_dir, "configs")
  tf.gfile.MakeDirs(configs_dir)
  tf.gfile.MakeDirs(FLAGS.output_dir)
  with tf.gfile.Open(
      os.path.join(configs_dir, "dual_encoder_config.pbtxt"), "w") as fout:
    print(exp_config, file=fout)

  # Write bert_config.json and doc_bert_config.json.
  tf.gfile.Copy(
      exp_config.encoder_config.bert_config_file,
      os.path.join(configs_dir, "bert_config.json"),
      overwrite=True)
  tf.gfile.Copy(
      exp_config.encoder_config.doc_bert_config_file,
      os.path.join(configs_dir, "doc_bert_config.json"),
      overwrite=True)

  # Write vocab file(s).
  tf.gfile.Copy(
      exp_config.encoder_config.vocab_file,
      os.path.join(configs_dir, "vocab.txt"),
      overwrite=True)

  # Save other important parameters as a json file.
  hparams = {
      "dual_encoder_config_file": FLAGS.dual_encoder_config_file,
      "output_dir": FLAGS.output_dir,
      "schedule": FLAGS.schedule,
      "debugging": FLAGS.debugging,
      "learning_rate": FLAGS.learning_rate,
      "num_warmup_steps": FLAGS.num_warmup_steps,
      "num_train_steps": FLAGS.num_train_steps,
      "num_tpu_cores": FLAGS.num_tpu_cores
  }
  with tf.gfile.Open(os.path.join(configs_dir, "hparams.json"), "w") as fout:
    json.dump(hparams, fout)
    tf.logging.info(">>>> saved hparams.json:\n {}".format(hparams))

  ############################################################################
  # Run the train/eval/predict/export process based on the schedule.
  ############################################################################
  max_seq_length_actual, max_predictions_per_seq_actual = \
        utils.get_actual_max_seq_len(exp_config.encoder_config.model_name,
                                     exp_config.encoder_config.max_doc_length_by_sentence,
                                     exp_config.encoder_config.max_sent_length_by_word,
                                     exp_config.encoder_config.max_predictions_per_seq)

  # Prepare input for train and eval.
  input_files = []
  for input_pattern in exp_config.train_eval_config.input_file_for_train.split(
      ","):
    input_files.extend(tf.gfile.Glob(input_pattern))
  input_file_num = 0
  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)
    input_file_num += 1
    if input_file_num > 10:
      break
  tf.logging.info("train input_files[0:10]: %s " % "\n".join(input_files[0:10]))
  eval_files = []
  if exp_config.train_eval_config.eval_with_eval_data:
    eval_files = []
    for input_pattern in exp_config.train_eval_config.input_file_for_eval.split(
        ","):
      eval_files.extend(tf.gfile.Glob(input_pattern))
  else:
    eval_files = input_files

  input_fn_builder = input_fns.input_fn_builder
  # Prepare the input functions.
  # Drop_remainder = True during training to maintain fixed batch size.
  train_input_fn = input_fn_builder(
      input_files=input_files,
      is_training=True,
      drop_remainder=True,
      max_seq_length=max_seq_length_actual,
      max_predictions_per_seq=max_predictions_per_seq_actual,
      num_cpu_threads=4,
      batch_size=exp_config.train_eval_config.train_batch_size,
  )
  eval_drop_remainder = True if FLAGS.use_tpu else False
  eval_input_fn = input_fn_builder(
      input_files=eval_files,
      max_seq_length=max_seq_length_actual,
      max_predictions_per_seq=max_predictions_per_seq_actual,
      is_training=False,
      drop_remainder=eval_drop_remainder,
      batch_size=exp_config.train_eval_config.eval_batch_size)
  predict_input_fn = input_fn_builder(
      input_files=eval_files,
      max_seq_length=max_seq_length_actual,
      max_predictions_per_seq=max_predictions_per_seq_actual,
      is_training=False,
      drop_remainder=eval_drop_remainder,
      batch_size=exp_config.train_eval_config.predict_batch_size,
      is_prediction=True)

  # Build and run the model.
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=exp_config.train_eval_config
      .save_checkpoints_steps,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=exp_config.train_eval_config.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = smith_modeling.model_fn_builder(
      dual_encoder_config=exp_config,
      train_mode=FLAGS.train_mode,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      debugging=FLAGS.debugging)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU. The batch size for eval and predict is the same.
  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=exp_config.train_eval_config.train_batch_size,
      eval_batch_size=exp_config.train_eval_config.eval_batch_size,
      predict_batch_size=exp_config.train_eval_config.predict_batch_size)

  if FLAGS.schedule == "train":
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d",
                    exp_config.train_eval_config.train_batch_size)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
  elif FLAGS.schedule == "continuous_eval":
    tf.logging.info("***** Running continuous evaluation *****")
    tf.logging.info("  Batch size = %d",
                    exp_config.train_eval_config.eval_batch_size)
    # checkpoints_iterator blocks until a new checkpoint appears.
    for ckpt in tf.train.checkpoints_iterator(estimator.model_dir):
      try:
        # Estimator automatically loads and evaluates the latest checkpoint.
        result = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=exp_config.train_eval_config.max_eval_steps)
        tf.logging.info("***** Eval results for %s *****", ckpt)
        for key, value in result.items():
          tf.logging.info("  %s = %s", key, str(value))

      except tf.errors.NotFoundError:
        # Checkpoint might get garbage collected before the eval can run.
        tf.logging.error("Checkpoint path '%s' no longer exists.", ckpt)
  elif FLAGS.schedule == "predict":
    # Load the model checkpoint and run the prediction process
    # to get the predicted scores and labels. The batch size is the same with
    # the eval batch size. For more options, refer to
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/estimator/tpu/TPUEstimator#predict
    tf.logging.info("***** Running prediction with ckpt {} *****".format(
        exp_config.encoder_config.predict_checkpoint))
    tf.logging.info("  Batch size = %d",
                    exp_config.train_eval_config.eval_batch_size)
    output_predict_file = os.path.join(FLAGS.output_dir,
                                       "prediction_results.json")
    # Output the prediction results in json format.
    pred_res_list = []
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      written_line_index = 0
      tf.logging.info("***** Predict results *****")
      for result in estimator.predict(
          input_fn=predict_input_fn,
          checkpoint_path=exp_config.encoder_config.predict_checkpoint,
          yield_single_examples=True):
        if (exp_config.encoder_config.model_name ==
            constants.MODEL_NAME_SMITH_DUAL_ENCODER):
          pred_item_dict = utils.get_pred_res_list_item_smith_de(result)
        else:
          raise ValueError("Unsupported model name: %s" %
                           exp_config.encoder_config.model_name)
        pred_res_list.append(pred_item_dict)
        written_line_index += 1
        if written_line_index % 500 == 0:
          tf.logging.info(
              "Current written_line_index: {} *****".format(written_line_index))
      tf.logging.info("***** Finished prediction for %d examples *****",
                      written_line_index)
      tf.logging.info("***** Output prediction results into %s *****",
                      output_predict_file)
      json.dump(pred_res_list, writer)

  elif FLAGS.schedule == "export":
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=exp_config.train_eval_config
        .save_checkpoints_steps)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
    export_dir_base = os.path.join(FLAGS.output_dir, "export/")
    tf.logging.info(
        "***** Export the prediction checkpoint to the folder {} *****".format(
            export_dir_base))
    tf.gfile.MakeDirs(export_dir_base)
    estimator.export_saved_model(
        export_dir_base=export_dir_base,
        assets_extra={"vocab.txt": exp_config.encoder_config.vocab_file},
        serving_input_receiver_fn=input_fns.make_serving_input_example_fn(
            max_seq_length=max_seq_length_actual,
            max_predictions_per_seq=max_predictions_per_seq_actual),
        checkpoint_path=exp_config.encoder_config.predict_checkpoint)
  else:
    raise ValueError("Unsupported schedule : %s" % FLAGS.schedule)


if __name__ == "__main__":
  flags.mark_flag_as_required("dual_encoder_config_file")
  flags.mark_flag_as_required("output_dir")
  app.run(main)
