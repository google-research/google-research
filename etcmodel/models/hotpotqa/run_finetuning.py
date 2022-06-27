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

"""Binary to run training or evaluation of ETC HotpotQA model."""
import collections
import functools
import json
import os
from typing import Mapping, Sequence, Union

from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from etcmodel.models import input_utils
from etcmodel.models import modeling
from etcmodel.models import tokenization
from etcmodel.models.hotpotqa import eval_utils
from etcmodel.models.hotpotqa import hotpot_evaluate_v1_lib as hotpot_eval
from etcmodel.models.hotpotqa import run_finetuning_lib

tf.compat.v1.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "etc_config_file", None,
    "The config json file corresponding to the pre-trained ETC model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "output_dir", None,
    ("The output directory where the model checkpoints and prediction results"
     "will be saved."))

flags.DEFINE_string("train_tf_examples_filepattern", None,
                    "Training tf examples filepattern.")

flags.DEFINE_integer("num_train_tf_examples", None,
                     "Number of train tf examples.")

flags.DEFINE_string("predict_tf_examples_filepattern", None,
                    "Prediction tf examples filepattern.")

flags.DEFINE_string("predict_gold_json_file", None, "Prediction json filename.")

flags.DEFINE_string(
    "spm_model_file", "",
    ("The SentencePiece tokenizer model file that the ETC model was trained on."
     "If not None, the `vocab_file` is ignored."))
flags.DEFINE_string(
    "vocab_file", "",
    "The WordPiece tokenizer vocabulary file that the ETC model was trained on."
)

flags.DEFINE_integer(
    "max_long_seq_length", 4096,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_global_seq_length", 230, "The maximum total global sequence length. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_enum("run_mode", "train", ["train", "predict", "export"],
                  "The run mode of the program.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_enum(
    "learning_rate_schedule", "poly_decay", ["poly_decay", "inverse_sqrt"],
    "The learning rate schedule to use. The default of "
    "`poly_decay` uses tf.train.polynomial_decay, while "
    "`inverse_sqrt` uses inverse sqrt of time after the warmup.")

flags.DEFINE_float("poly_power", 1.0, "The power of poly decay.")

flags.DEFINE_enum("optimizer", "adamw", ["adamw", "lamb"],
                  "The optimizer for training.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    ("Proportion of training to perform linear learning rate warmup for. "
     "E.g., 0.1 = 10% of training."))

flags.DEFINE_integer("start_warmup_step", 0, "The starting step of warmup.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_bool(
    "flat_sequence", False,
    ("If True, the attention masks / relative attention ids would be computing"
     "assuming the default ETC setting where there is not any structure (except"
     "for having the notion of a 'sentence')."))

flags.DEFINE_enum("answer_encoding_method", "span", ["span", "bio"],
                  "The answer encoding method.")

flags.DEFINE_bool("use_tpu", True, "Whether to use tpu.")

flags.DEFINE_string("tpu_job_name", None,
                    "Name of TPU worker, if anything other than 'tpu_worker'")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer("save_checkpoints_steps", 500,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 500,
                     "How many steps to make in each estimator call.")

flags.DEFINE_float(
    "supporting_fact_threshold", 0.5,
    ("The threshold for whether a sentence is a supporting fact. If None search"
     "the threshold for best joint f1."))

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The max number of wordpiece toknes allowed for an answer.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_bool("save_raw_predictions", False,
                  "Whether to save raw predictions to file.")

flags.DEFINE_integer(
    "grad_checkpointing_period", None,
    "If specified, this overrides the corresponding `EtcConfig` value loaded "
    "from `etc_config_file`.")

flags.DEFINE_integer("random_seed", 0, "Random seed for random repeat runs.")

flags.DEFINE_string(
    "export_ckpts", None, "A space separated list of all the "
    "checkpoints to be exported. If None, exports all the "
    "checkpoints within the model_dir. Applicable only when "
    "`do_export` is set to True.")

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


def _get_tokenizer():
  """Gets tokenizer and whether WordPiece tokenizer is used."""
  if FLAGS.spm_model_file:
    use_wordpiece = False
    tokenizer = tokenization.FullTokenizer(None, None, FLAGS.spm_model_file)
  elif FLAGS.vocab_file:
    use_wordpiece = True
    tokenizer = tokenization.FullTokenizer(FLAGS.vocab_file)
  else:
    raise ValueError(
        "Either a 'sp_model' or a 'vocab_file' need to specified to create a"
        "tokenizer.")
  return tokenizer, use_wordpiece


def _add_extra_info(raw_predictions: Sequence[Mapping[str, np.ndarray]],
                    predict_tf_examples: Sequence[tf.train.Example],
                    use_wordpiece: bool) -> None:
  """Adds the extra info in tf examples to raw predictions."""
  if len(raw_predictions) != len(predict_tf_examples):
    raise ValueError(
        f"Num of raw predictions {len(raw_predictions)} doesn't equal to"
        f"num of predict tf examples {len(predict_tf_examples)}.")
  for raw_prediction, predict_tf_example in zip(raw_predictions,
                                                predict_tf_examples):
    for meta_info_name in ["unique_ids", "type", "level"]:
      raw_prediction[meta_info_name] = input_utils.get_repeated_values(
          meta_info_name, predict_tf_example)[0]
    raw_prediction["global_sentence_ids"] = np.array(
        input_utils.get_repeated_values("global_sentence_ids",
                                        predict_tf_example))
    raw_prediction["global_paragraph_ids"] = np.array(
        input_utils.get_repeated_values("global_paragraph_ids",
                                        predict_tf_example))
    if use_wordpiece:
      raw_prediction["long_tokens_to_unigrams"] = np.array(
          input_utils.get_repeated_values("long_tokens_to_unigrams",
                                          predict_tf_example))


def _save_raw_predictions(checkpoint: str,
                          raw_predictions: Sequence[Mapping[str, np.ndarray]],
                          use_wordpiece: bool) -> None:
  """Save raw prediction to file as tf.Examples."""
  output_file = f"{checkpoint}.predicted-tfrecords"
  with tf.python_io.TFRecordWriter(output_file) as writer:
    for raw_prediction in raw_predictions:
      features = collections.OrderedDict()
      for output_name in ["unique_ids", "type", "level"]:
        features[output_name] = input_utils.create_bytes_feature(
            [raw_prediction[output_name]])
      for output_name in [
          "long_token_ids", "long_sentence_ids", "long_token_type_ids",
          "global_token_ids", "global_sentence_ids", "global_paragraph_ids",
          "answer_begin_top_indices", "answer_end_top_indices", "answer_types"
      ]:
        features[output_name] = input_utils.create_int_feature(
            raw_prediction[output_name])
      for output_name in [
          "supporting_facts_probs",
          "answer_begin_top_probs",
          "answer_end_top_probs",
      ]:
        features[output_name] = input_utils.create_float_feature(
            raw_prediction[output_name])
      if use_wordpiece:
        features["long_tokens_to_unigrams"] = input_utils.create_int_feature(
            raw_prediction["long_tokens_to_unigrams"])
      writer.write(
          tf.train.Example(features=tf.train.Features(
              feature=features)).SerializeToString())


def _get_predictions_and_scores(raw_predictions, gold_json_data, tokenizer,
                                use_wordpiece, sp_threshold):
  prediction_json = eval_utils.generate_prediction_json(
      raw_predictions, gold_json_data, tokenizer, sp_threshold,
      FLAGS.max_answer_length, use_wordpiece, FLAGS.answer_encoding_method)
  scores = hotpot_eval.evaluate(prediction_json, gold_json_data)
  scores.update(hotpot_eval.get_em_counts(prediction_json, gold_json_data))
  return prediction_json, scores


def _search_sp_threshold(raw_predictions, gold_json_data, tokenizer,
                         use_wordpiece):
  """Search supporting facts thresholds giving the best joint f1."""
  best_joint_f1 = -1.0
  best_result = None
  sp_thresholds = np.linspace(0, 1, 11)
  for sp_threshold in sp_thresholds:
    prediction_json, scores = _get_predictions_and_scores(
        raw_predictions, gold_json_data, tokenizer, use_wordpiece, sp_threshold)
    if scores["joint_f1"] > best_joint_f1:
      best_joint_f1 = scores["joint_f1"]
      best_result = (sp_threshold, prediction_json, scores)
  assert best_result is not None, "No best result."
  return best_result


def _write_predictions_json(prediction_json, filename: str):
  with tf.gfile.Open(filename, "w") as f:
    json.dump(prediction_json, f)


def _write_scores_to_summary(scores: Mapping[str, Union[float, int]],
                             summary_writer: tf.summary.FileWriter,
                             current_step: int) -> None:
  """Writes eval scores to tf summary file."""
  for metric_name, score in scores.items():
    if metric_name.startswith("sp"):
      metric_name = f"sp/{metric_name}"
    elif metric_name.startswith("joint"):
      metric_name = f"joint/{metric_name}"
    else:
      metric_name = f"ans/{metric_name}"
    summary_writer.add_summary(
        tf.Summary(value=[
            tf.Summary.Value(tag=metric_name, simple_value=score),
        ]),
        global_step=current_step)
  summary_writer.flush()


def _write_scores_to_text(scores: Mapping[str, Union[float, int]],
                          filename: str) -> None:
  """Writes eval scores to text file."""
  lb_metrics = ["em", "f1", "sp_em", "sp_f1", "joint_em", "joint_f1"]
  lb_scores = np.array([scores[k] for k in lb_metrics])
  with tf.gfile.Open(filename, "w") as f:
    print("leaderboard metrics:", file=f)
    print("ans, sup, joint", file=f)
    print("EM, F1, EM, F1, EM, F1", file=f)
    print(", ".join(["{:.2f}"] * 6).format(*(lb_scores * 100)), file=f)
    print(", ".join(["{}"] * 6).format(*lb_scores), file=f)
    print("all metrics:", file=f)
    for metric_name, score in sorted(scores.items()):
      print(f"{metric_name}: {score}", file=f)


def _serving_input_receiver_fn(
    long_seq_length: int,
    global_seq_length: int,
) -> tf_estimator.export.ServingInputReceiver:
  """Creates an input function to parse input features for inference.

  This function defines format of the inputs to the exported HotpotQA model.
  at inference time.

  Args:
    long_seq_length: The long input len.
    global_seq_length: The global input len.

  Returns:
    The ServingInputReceiver fn.
  """

  # An input receiver that expects a vector of serialized `tf.Example`s.
  serialized_tf_example = tf.placeholder(
      dtype=tf.string, shape=[None], name="serialized_tf_example")
  receiver_tensors = {"serialized_tf_example": serialized_tf_example}
  schema = run_finetuning_lib.get_inference_name_to_features(
      long_seq_length, global_seq_length)
  features = tf.parse_example(serialized_tf_example, schema)
  return tf_estimator.export.ServingInputReceiver(features, receiver_tensors)


def run_export(estimator, model_dir, export_ckpts, long_seq_length,
               global_seq_length):
  """Exports a `tf.SavedModel` for each checkpoint in the model_dir.

  Args:
    estimator: The TPUEstimator.
    model_dir: The model directory to be used for finding the checkpoints to be
      exported.
    export_ckpts: A space separated list of all the checkpoints to be exported.
      If None, exports all the checkpoints within the `model_dir`
    long_seq_length: The long input len.
    global_seq_length: The global input len.
  """

  if export_ckpts is None:
    # Export all the checkpoints within the `model_dir`.
    ckpts = [
        f[0:f.rfind(".")]
        for f in tf.gfile.ListDir(model_dir)
        if f.startswith("model.ckpt-")
    ]
    ckpts = set(ckpts)
  else:
    ckpts = [ckpt.strip() for ckpt in export_ckpts.split(" ")]

  for ckpt_name in ckpts:
    ckpt_path = os.path.join(model_dir, ckpt_name)
    export_path = estimator.export_saved_model(
        export_dir_base=os.path.join(model_dir, "saved_models", ckpt_name),
        serving_input_receiver_fn=functools.partial(
            _serving_input_receiver_fn,
            long_seq_length=long_seq_length,
            global_seq_length=global_seq_length),
        checkpoint_path=ckpt_path)

    tf.logging.info("HotpotQA ETC Model exported to %s for checkpoint %s ",
                    export_path, ckpt_path)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  with tf.gfile.Open(FLAGS.predict_gold_json_file, "r") as f:
    gold_json_data = json.load(f)

  predict_tf_examples = []
  for tf_record_path in tf.gfile.Glob(FLAGS.predict_tf_examples_filepattern):
    for tf_record in tf.compat.v1.io.tf_record_iterator(tf_record_path):
      predict_tf_examples.append(tf.train.Example.FromString(tf_record))

  tokenizer, use_wordpiece = _get_tokenizer()

  etc_model_config = modeling.EtcConfig.from_json_file(FLAGS.etc_config_file)
  if FLAGS.grad_checkpointing_period is not None:
    etc_model_config.grad_checkpointing_period = FLAGS.grad_checkpointing_period

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  run_config = tf_estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf_estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=(
              tf_estimator.tpu.InputPipelineConfig.PER_HOST_V2),
          tpu_job_name=FLAGS.tpu_job_name))

  num_train_steps = int(FLAGS.num_train_tf_examples / FLAGS.train_batch_size *
                        FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
  model_fn = run_finetuning_lib.model_fn_builder(
      etc_model_config, FLAGS.learning_rate, num_train_steps, num_warmup_steps,
      FLAGS.flat_sequence, FLAGS.answer_encoding_method, FLAGS.use_tpu,
      use_wordpiece, FLAGS.optimizer, FLAGS.poly_power, FLAGS.start_warmup_step,
      FLAGS.learning_rate_schedule, FLAGS.init_checkpoint)

  estimator = tf_estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size,
      export_to_tpu=False)

  if FLAGS.run_mode == "train":
    tf.logging.info("***** Running train of models *****")
    input_fn = run_finetuning_lib.input_fn_builder(
        input_filepattern=FLAGS.train_tf_examples_filepattern,
        long_seq_length=FLAGS.max_long_seq_length,
        global_seq_length=FLAGS.max_global_seq_length,
        is_training=True,
        answer_encoding_method=FLAGS.answer_encoding_method,
        drop_remainder=True)
    estimator.train(input_fn=input_fn, max_steps=num_train_steps)

  elif FLAGS.run_mode == "predict":
    tf.logging.info("***** Running predict of models *****")
    summary_writer = tf.summary.FileWriter(
        logdir=os.path.join(FLAGS.output_dir))
    input_fn = run_finetuning_lib.input_fn_builder(
        input_filepattern=FLAGS.predict_tf_examples_filepattern,
        long_seq_length=FLAGS.max_long_seq_length,
        global_seq_length=FLAGS.max_global_seq_length,
        is_training=False,
        answer_encoding_method=FLAGS.answer_encoding_method,
        drop_remainder=False)
    for ckpt in tf.train.checkpoints_iterator(FLAGS.output_dir):
      try:
        raw_predictions = list(
            estimator.predict(
                input_fn=input_fn,
                checkpoint_path=ckpt,
                yield_single_examples=True))
      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info("Checkpoint %s no longer exists, skipping checkpoint",
                        ckpt)
        continue
      _add_extra_info(raw_predictions, predict_tf_examples, use_wordpiece)

      if FLAGS.save_raw_predictions:
        _save_raw_predictions(ckpt, raw_predictions, use_wordpiece)

      prediction_json, scores = _get_predictions_and_scores(
          raw_predictions, gold_json_data, tokenizer, use_wordpiece,
          FLAGS.supporting_fact_threshold)

      current_step = int(os.path.basename(ckpt).split("-")[1])
      _write_predictions_json(prediction_json, f"{ckpt}.predictions.json")
      _write_scores_to_summary(scores, summary_writer, current_step)
      _write_scores_to_text(scores, f"{ckpt}.scores.txt")

      # Terminate eval job when final checkpoint is reached
      if current_step >= num_train_steps:
        sp_threshold, prediction_json, scores = _search_sp_threshold(
            raw_predictions, gold_json_data, tokenizer, use_wordpiece)
        _write_predictions_json(
            prediction_json, f"{ckpt}.predictions_sp{sp_threshold:.2f}.json")
        _write_scores_to_text(scores, f"{ckpt}.scores_sp{sp_threshold:.2f}.txt")
        tf.logging.info(
            f"Prediction finished after training step {current_step}")
        break

  elif FLAGS.run_mode == "export":
    tf.logging.info("***** Running export of models *****")
    run_export(
        estimator=estimator,
        model_dir=FLAGS.output_dir,
        export_ckpts=FLAGS.export_ckpts,
        long_seq_length=FLAGS.max_long_seq_length,
        global_seq_length=FLAGS.max_global_seq_length)


if __name__ == "__main__":
  tf.app.run(main)
