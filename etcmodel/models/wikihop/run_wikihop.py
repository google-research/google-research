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

"""ETC finetuning runner for WikiHop evaluation.

1) Reference paper describing the construction and details of the dataset:
https://transacl.org/ojs/index.php/tacl/article/viewFile/1325/299

2) Dataset link: http://qangaroo.cs.ucl.ac.uk/

"""

import json
import os
import time

from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from etcmodel.models import input_utils
from etcmodel.models.wikihop import run_wikihop_lib

tf.compat.v1.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "source_model_config_file", None,
    "The source config json file corresponding to the ETC model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_tf_records_path", None,
    "The path to the TFRecords. If None, the data will be generated using "
    "the `input_file_path`. At least one of `input_file_path` or "
    "this should be specified. This flag is useful for optimization where in "
    "we don't need to generated train/dev tf_records during multiple "
    "iterations of modeling.")

flags.DEFINE_string(
    "predict_ckpt", None, "The path to the checkpoint to "
    "be used for in predict mode. If None, the latest checkpoint in the "
    "model dir would be used.")

flags.DEFINE_string(
    "predict_output_file_path", None, "The full path of the output file to "
    "write the test set results. The results would be in json format with key "
    "being the example id and value being the candidate answer.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_bool(
    "candidate_ignore_hard_g2l", False, "If True, all the "
    "candidate tokens in the global input attend to everything "
    "in the long input (except padding) even when "
    "`use_hard_g2l_mask` is enabled.")

flags.DEFINE_bool(
    "query_ignore_hard_g2l", False, "If True, all the "
    "query tokens in the global input attend to everything in "
    "the long input (except padding) even when "
    "`use_hard_g2l_mask` is enabled.")

flags.DEFINE_bool(
    "enable_l2g_linking", True, "If True, all the "
    "candidate mentions in the long will be linked to the "
    "candidate global token.")

flags.DEFINE_float(
    "hidden_dropout_prob", -1, "The dropout probability for "
    "all fully connected layers in the embeddings, encoder, and "
    "pooler.")

flags.DEFINE_float("attention_probs_dropout_prob", -1,
                   "The dropout ratio for the attention "
                   "probabilities.")

flags.DEFINE_float(
    "local_radius", -1, "The local radius (window size) for the long input "
    "attention.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained ETC model) to start "
    "fine-tuning.")

flags.DEFINE_integer("long_seq_len", 4096,
                     "The total input sequence length to pad to for training.")

flags.DEFINE_integer("global_seq_len", 430,
                     "The raw maximum global input sequence length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "do_export", False, "To export SavedModels for all the "
    "checkpoints within the model_dir.")

flags.DEFINE_string(
    "export_ckpts", None, "A space separated list of all the "
    "checkpoints to be exported. If None, exports all the "
    "checkpoints within the model_dir. Applicable only when "
    "`do_export` is set to True.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_enum("optimizer", "adamw", ["adamw", "lamb"],
                  "The optimizer for training.")

flags.DEFINE_float("learning_rate", 3e-05, "The initial learning rate for "
                   "Adam.")

flags.DEFINE_float("weight_decay_rate", 0.1, "The weight decay rate.")

flags.DEFINE_float("label_smoothing", 0.0, "The label smoothing param.")

flags.DEFINE_integer(
    "num_train_epochs", 15, "Number of train epochs. The total number of "
    "examples on the WikiHop dataset is ~44K.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer(
    "max_eval_steps", 600, "Maximum number of eval steps. "
    "Total number of dev examples in WikiHop is ~5K. "
    "This number has been set assuming a eval_batch_size of "
    "8.")

flags.DEFINE_enum(
    "learning_rate_schedule", "poly_decay", ["poly_decay", "inverse_sqrt"],
    "The learning rate schedule to use. The default of "
    "`poly_decay` uses tf.train.polynomial_decay, while "
    "`inverse_sqrt` uses inverse sqrt of time after the warmup.")

flags.DEFINE_float("poly_power", 1.0,
                   "The power of poly decay if using `poly_decay` schedule.")

flags.DEFINE_integer("start_warmup_step", 0, "The starting step of warmup.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer(
    "grad_checkpointing_period", None,
    "If specified, this overrides the corresponding `EtcConfig` value loaded "
    "from `source_model_config_file`.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("keep_checkpoint_max", 100,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "Whether to use one-hot multiplication instead of gather for embedding "
    "lookups.")

flags.DEFINE_bool(
    "add_final_layer", True,
    "If True, a ResNet block is applied on the global output before "
    "prediction.")

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

flags.DEFINE_string("tpu_job_name", None,
                    "Name of TPU worker, if anything other than 'tpu_worker'")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer("num_train_examples", None, "Number of train tf examples.")

flags.DEFINE_integer("num_dev_examples", None, "Number of dev tf examples.")


def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.compat.v1.enable_resource_variables()

  if (not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and
      not FLAGS.do_export):
    raise ValueError(
        "At least one of `do_train`, `do_eval`, `do_predict' or `do_export` "
        "must be True.")

  if not FLAGS.do_export and FLAGS.input_tf_records_path is None:
    raise ValueError(
        "input_tf_records_path` must be specified when not in export mode.")

  tf.gfile.MakeDirs(FLAGS.output_dir)

  model_config = input_utils.get_model_config(
      model_dir=FLAGS.output_dir,
      source_file=FLAGS.source_model_config_file,
      write_from_source=FLAGS.do_train)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf_estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf_estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      tpu_config=tf_estimator.tpu.TPUConfig(
          tpu_job_name=FLAGS.tpu_job_name,
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  num_train_steps = int(FLAGS.num_train_examples / FLAGS.train_batch_size *
                        FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = run_wikihop_lib.model_fn_builder(
      model_config=model_config,
      model_dir=FLAGS.output_dir,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings,
      optimizer=FLAGS.optimizer,
      poly_power=FLAGS.poly_power,
      start_warmup_step=FLAGS.start_warmup_step,
      learning_rate_schedule=FLAGS.learning_rate_schedule,
      add_final_layer=FLAGS.add_final_layer,
      weight_decay_rate=FLAGS.weight_decay_rate,
      label_smoothing=FLAGS.label_smoothing)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf_estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size,
      use_tpu=FLAGS.use_tpu,
      export_to_tpu=False)

  if FLAGS.do_export:
    tf.logging.info("***** Running export of models *****")
    run_wikihop_lib.run_export(
        estimator=estimator,
        model_dir=FLAGS.output_dir,
        model_config=model_config,
        export_ckpts=FLAGS.export_ckpts,
        long_seq_len=FLAGS.long_seq_len,
        global_seq_len=FLAGS.global_seq_len,
        candidate_ignore_hard_g2l=FLAGS.candidate_ignore_hard_g2l,
        query_ignore_hard_g2l=FLAGS.query_ignore_hard_g2l,
        enable_l2g_linking=FLAGS.enable_l2g_linking)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    assert FLAGS.weight_decay_rate is not None
    assert FLAGS.learning_rate is not None
    tf.logging.info("*** Model Hyperparams ****")
    tf.logging.info(
        "learning_rate: {}, weight_decay_rate: {}, label_smoothing:{}"
        .format(FLAGS.learning_rate, FLAGS.weight_decay_rate,
                FLAGS.label_smoothing))
    if FLAGS.hidden_dropout_prob >= 0.0:
      model_config.hidden_dropout_prob = FLAGS.hidden_dropout_prob
      tf.logging.info("Overwriting hidden_dropout_prob to: {}".format(
          model_config.hidden_dropout_prob))

    if FLAGS.attention_probs_dropout_prob >= 0.0:
      model_config.attention_probs_dropout_prob = (
          FLAGS.attention_probs_dropout_prob)
      tf.logging.info("Overwriting attention_probs_dropout_prob to: {}".format(
          model_config.attention_probs_dropout_prob))

    if FLAGS.grad_checkpointing_period is not None:
      model_config.grad_checkpointing_period = FLAGS.grad_checkpointing_period
      tf.logging.info("Overwriting grad_checkpointing_period to: {}".format(
          model_config.grad_checkpointing_period))

    if FLAGS.local_radius >= 0:
      model_config.local_radius = FLAGS.local_radius
      tf.logging.info("Overwriting local_radius to: {}".format(
          model_config.local_radius))

    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_tf_file = FLAGS.input_tf_records_path
    train_input_fn = run_wikihop_lib.input_fn_builder(
        input_file_pattern=train_tf_file,
        model_config=model_config,
        long_seq_len=FLAGS.long_seq_len,
        global_seq_len=FLAGS.global_seq_len,
        is_training=True,
        drop_remainder=True,
        candidate_ignore_hard_g2l=FLAGS.candidate_ignore_hard_g2l,
        query_ignore_hard_g2l=FLAGS.query_ignore_hard_g2l,
        enable_l2g_linking=FLAGS.enable_l2g_linking)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_tf_file = FLAGS.input_tf_records_path
    eval_input_fn = run_wikihop_lib.input_fn_builder(
        input_file_pattern=eval_tf_file,
        model_config=model_config,
        long_seq_len=FLAGS.long_seq_len,
        global_seq_len=FLAGS.global_seq_len,
        is_training=False,
        drop_remainder=eval_drop_remainder,
        candidate_ignore_hard_g2l=FLAGS.candidate_ignore_hard_g2l,
        query_ignore_hard_g2l=FLAGS.query_ignore_hard_g2l,
        enable_l2g_linking=FLAGS.enable_l2g_linking)

    # Run evaluation for each new checkpoint.
    for ckpt in tf.train.checkpoints_iterator(FLAGS.output_dir):
      tf.logging.info("Starting eval on new checkpoint: %s", ckpt)
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = estimator.evaluate(
            input_fn=eval_input_fn,
            checkpoint_path=ckpt,
            steps=FLAGS.max_eval_steps,
            name="metrics")
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info("Eval results: %s. Elapsed seconds: %d", eval_results,
                        elapsed_time)

        # Terminate eval job when final checkpoint is reached.
        current_step = int(os.path.basename(ckpt).split("-")[1])
        if current_step >= num_train_steps:
          tf.logging.info("Evaluation finished after training step %d",
                          current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info("Checkpoint %s no longer exists, skipping checkpoint",
                        ckpt)

  if FLAGS.do_predict:
    predict_tf_file = FLAGS.input_tf_records_path
    predict_input_fn = run_wikihop_lib.input_fn_builder(
        input_file_pattern=predict_tf_file,
        model_config=model_config,
        long_seq_len=FLAGS.long_seq_len,
        global_seq_len=FLAGS.global_seq_len,
        is_training=False,
        drop_remainder=False,
        candidate_ignore_hard_g2l=FLAGS.candidate_ignore_hard_g2l,
        query_ignore_hard_g2l=FLAGS.query_ignore_hard_g2l,
        enable_l2g_linking=FLAGS.enable_l2g_linking)

    tf.logging.info("***** Running prediction *****")
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    for ckpt in tf.train.checkpoints_iterator(FLAGS.output_dir):
      tf.logging.info("Starting prediction on new checkpoint: %s", ckpt)
      current_step = int(os.path.basename(ckpt).split("-")[1])

      try:
        result = estimator.predict(
            input_fn=predict_input_fn,
            checkpoint_path=ckpt,
            yield_single_examples=True)
      except tf.errors.NotFoundError:
        tf.logging.info("Checkpoint %s no longer exists, skipping checkpoint",
                        ckpt)
        continue

      tf.logging.info("***** Predict results for ckpt = %d *****", current_step)
      predict_output_file = os.path.join(
          FLAGS.output_dir, "predict-" + str(current_step) + ".json")
      predict_output = {}
      num_written_lines = 0
      num_correct_predictions = 0
      num_incorrect_predictions = 0

      for (i, prediction) in enumerate(result):
        if i >= FLAGS.num_dev_examples:
          break

        if i % 500 == 0:
          tf.logging.info("*** Done processing %d examples for ckpt %d ***", i,
                          current_step)
          tf.logging.info("*** num_total_predictions = %d ***",
                          num_written_lines)
          tf.logging.info("*** num_correct_predictions = %d ***",
                          num_correct_predictions)
          tf.logging.info("*** num_incorrect_predictions = %d ***",
                          num_incorrect_predictions)

        logits = prediction["logits"]
        assert len(logits) == FLAGS.global_seq_len
        predicted_index = np.argmax(logits)
        predict_output["WH_dev_" + str(i)] = str(predicted_index)
        if prediction["label_ids"] == predicted_index:
          num_correct_predictions += 1
        else:
          num_incorrect_predictions += 1
        num_written_lines += 1

      tf.logging.info("*** Prediction results for ckpt = %d ***", current_step)
      tf.logging.info("*** num_total_predictions = %d ***", num_written_lines)
      tf.logging.info("*** num_correct_predictions = %d ***",
                      num_correct_predictions)
      tf.logging.info("*** num_incorrect_predictions = %d ***",
                      num_incorrect_predictions)

      assert num_written_lines == FLAGS.num_dev_examples

      predict_output["num_total_predictions"] = num_written_lines
      predict_output["num_correct_predictions"] = num_correct_predictions
      predict_output["num_incorrect_predictions"] = num_incorrect_predictions
      predict_output["accuracy"] = (num_correct_predictions / num_written_lines)

      with tf.gfile.GFile(predict_output_file, "w") as writer:
        json.dump(predict_output, writer)

      if current_step >= num_train_steps:
        tf.logging.info("Prediction finished after training step %d",
                        current_step)
        break


if __name__ == "__main__":
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("source_model_config_file")
  tf.app.run()
