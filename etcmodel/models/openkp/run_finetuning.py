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

r"""Runs ETC finetuning for OpenKP."""

import collections
import json
import os
import re
from typing import Mapping, Optional, Text

import numpy as np
import tensorflow.compat.v1 as tf

from etcmodel.models import input_utils
from etcmodel.models import modeling
from etcmodel.models.openkp import eval_utils
from etcmodel.models.openkp import run_finetuning_lib

tf.compat.v1.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'etc_config_file', None,
    'The config json file corresponding to the pre-trained ETC model. '
    'This specifies the model architecture.')

flags.DEFINE_string(
    'output_dir', None,
    'The output directory where the model checkpoints will be written.')

flags.DEFINE_string('input_tfrecord', None,
                    'Input glob for TF examples in tfrecord format.')

flags.DEFINE_string(
    'init_checkpoint', None,
    'Initial checkpoint (usually from a pre-trained ETC model).')

flags.DEFINE_string(
    'eval_text_example_path', None,
    'A jsonl file full of `OpenKpTextExample` instances to use for '
    'evaluating model predictions. Only used if `do_eval` is True.')

flags.DEFINE_string(
    'predict_text_example_path', None,
    'A jsonl file full of `OpenKpTextExample` instances to use for '
    'running inference. Only used if `do_predict` is True.')

flags.DEFINE_float(
    'eval_fraction_of_removed_examples', None,
    'A float between 0 and 1 (inclusive) giving the fraction of examples '
    'removed during example generation. We use this to "deflate" the eval '
    'metrics assuming that all removed examples would yield a 0 '
    'precision/recall/f1 score. Examples will be removed if no key phrases '
    'appear anywhere in the text after truncating to fit the maximum input '
    'lengths.')

flags.DEFINE_string(
    'predict_checkpoint_path', None,
    'Path to the checkpoint to use for inference. Only used if `do_predict` '
    'is True. If None, the latest checkpoint in `output_dir` is used.')

flags.DEFINE_string(
    'predict_output_name', 'inference',
    'Name to use for inference outputs when `do_predict=True`.')

flags.DEFINE_integer(
    'long_max_length', 4096,
    'The maximum total long sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')

flags.DEFINE_integer(
    'global_max_length', 512,
    'The maximum total global sequence length. Sequences longer than this '
    'will be truncated, and sequences shorter than this will be padded.')

flags.DEFINE_integer(
    'url_num_code_points', 1000,
    'The number of Unicode code points in the `url_code_points` feature '
    '(which represents the example id).')

flags.DEFINE_integer('num_labels', 3,
                     'The maximum number of key phrase labels.')

flags.DEFINE_integer('kp_max_length', 5,
                     'The maximum key phrase ngram length to predict.')

flags.DEFINE_integer(
    'max_kp_predictions', 5,
    'Maximum number of key phrases to predict. Predicting fewer than 5 key '
    'phrases can only hurt OpenKP eval metrics, and predicting more has '
    'no effect.')

flags.DEFINE_integer(
    'max_position_predictions', 100,
    'Maximum number of positions to predict key phrases for. Position '
    'predictions will turn into `max_kp_predictions` key phrase predictions.')

flags.DEFINE_bool('do_train', False, 'Whether to run training.')

flags.DEFINE_bool('do_eval', False, 'Whether to run eval on the dev set.')

flags.DEFINE_bool('do_predict', False,
                  'Whether to run inference on the input examples.')

flags.DEFINE_integer('batch_size', 64, 'Total batch size to use.')

flags.DEFINE_integer('eval_batch_size', 32, 'Total eval batch size to use.')

flags.DEFINE_float('num_train_epochs', 2.0,
                   'Total number of training epochs to perform.')

flags.DEFINE_integer(
    'num_train_examples', None,
    'Total number of training examples per epoch. The number of training '
    'steps will be `num_train_epochs * num_train_examples / batch_size`.')

flags.DEFINE_float(
    'additive_smoothing_mass', 1e-4,
    'Total probability mass to add for additive smoothing for the label '
    'probabilities (on top of `1.0` mass for the actual label). 1e-6 is the '
    'minimum value.')

flags.DEFINE_bool(
    'use_visual_features_in_global', True,
    'If True (the default), we add all the visual features associated with '
    'VDOM elements to global token embeddings.')

flags.DEFINE_bool(
    'use_visual_features_in_long', True,
    'If True (the default), we add all the visual features associated with '
    'VDOM elements to long token embeddings.')

flags.DEFINE_integer(
    'extra_dense_feature_layers', 1,
    'Number of extra dense feature layers to use on top of the initial dense '
    'projection to `hidden_size`. Currently only 0 or 1 are supported.')

flags.DEFINE_float('x_coords_min', -50,
                   'Lower VDOM x coordinate to clip values to.')

flags.DEFINE_float('x_coords_max', 1500,
                   'Upper VDOM x coordinate to clip values to.')

flags.DEFINE_float('y_coords_min', -50,
                   'Lower VDOM y coordinate to clip values to.')

flags.DEFINE_float('y_coords_max', 6000,
                   'Upper VDOM y coordinate to clip values to.')

flags.DEFINE_float('widths_min', 0, 'Lower VDOM width to clip values to.')

flags.DEFINE_float('widths_max', 1500, 'Upper VDOM width to clip values to.')

flags.DEFINE_float('heights_min', 0, 'Lower VDOM height to clip values to.')

flags.DEFINE_float('heights_max', 18000, 'Upper VDOM height to clip values to.')

flags.DEFINE_list(
    'indicators_to_cross',
    [
        'global_block_indicator',  #
        'global_heading_indicator',
        'global_bold_indicator'
    ],
    'A list of indicator features to cross to form a single categorical '
    'feature to embed. All other indicators not in this list will be treated '
    'as dense features.')

flags.DEFINE_enum('optimizer', 'adamw', ['adamw', 'lamb'],
                  'The optimizer for training.')

flags.DEFINE_float('learning_rate', 5e-5, 'The initial learning rate for Adam.')

flags.DEFINE_enum(
    'learning_rate_schedule', 'poly_decay', ['poly_decay', 'inverse_sqrt'],
    'The learning rate schedule to use. The default of '
    '`poly_decay` uses tf.train.polynomial_decay, while '
    '`inverse_sqrt` uses inverse sqrt of time after the warmup.')

flags.DEFINE_float('poly_power', 1.0, 'The power of poly decay.')

flags.DEFINE_float(
    'warmup_proportion', 0.1,
    'Proportion of training to perform linear learning rate warmup for. '
    'E.g., 0.1 = 10% of training.')

flags.DEFINE_integer('start_warmup_step', 0, 'The starting step of warmup.')

flags.DEFINE_integer('save_checkpoints_steps', 1000,
                     'How often to save the model checkpoint.')

flags.DEFINE_integer('iterations_per_loop', 200,
                     'How many steps to make in each estimator call.')

flags.DEFINE_integer(
    'grad_checkpointing_period', None,
    'If specified, this overrides the corresponding `EtcConfig` value loaded '
    'from `etc_config_file`.')

flags.DEFINE_integer('random_seed', 0, 'Dummy flag used for random restarts.')

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU or GPU/CPU.')

tf.flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')

tf.flags.DEFINE_string(
    'tpu_zone', None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

tf.flags.DEFINE_string(
    'gcp_project', None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

tf.flags.DEFINE_string('master', None, '[Optional] TensorFlow master URL.')

flags.DEFINE_bool(
    'verbose_logging', False,
    'If true, all of the warnings related to data processing will be printed. '
    'A number of warnings are expected for a normal NQ evaluation.')

flags.DEFINE_string(
    'tpu_job_name', None,
    'Name of TPU worker binary. Only necessary if job name is changed from'
    ' default tpu_worker.')


class _MetricAverager:
  """Utility class for averaging metrics across examples."""

  def __init__(self):
    """Init."""
    self._count = 0
    self._total_precision = 0
    self._total_recall = 0
    self._total_f1 = 0

  def add_example(self, precision: float, recall: float, f1: float):
    """Adds metrics for 1 example."""
    self._count += 1
    self._total_precision += precision
    self._total_recall += recall
    self._total_f1 += f1

  @property
  def precision(self):
    return self._total_precision / self._count

  @property
  def recall(self):
    return self._total_recall / self._count

  @property
  def f1(self):
    return self._total_f1 / self._count


def _validate_flags():
  for flag_name in ['etc_config_file', 'output_dir', 'input_tfrecord']:
    if not getattr(FLAGS, flag_name):
      raise ValueError(f'`{flag_name}` flag must be specified.')


def _get_global_step_for_checkpoint(checkpoint_path: Text) -> int:
  """Returns the global step for the checkpoint path, or -1 if not found."""
  re_match = re.search(r'ckpt-(\d+)$', checkpoint_path)
  return -1 if re_match is None else int(re_match.group(1))


def _process_prediction(
    prediction: Mapping[Text, np.ndarray],
    text_examples: Mapping[Text, eval_utils.OpenKpTextExample],
    writer_tfrecord,
    writer_jsonl,
    metrics: Optional[Mapping[int, _MetricAverager]] = None) -> None:
  """Processes a single TF `Estimator.predict` prediction.

  This function assumes that `Estimator.predict` was called with
  `yield_single_examples=True`.

  Args:
    prediction: Prediction from `Estimator.predict` for a single example.
    text_examples: A dictionary of `OpenKpTextExample` objects, keyed by URL.
      This is used to generate the KeyPhrase predictions based on the ngram
      logits in the prediction.
    writer_tfrecord: An open `tf.python_io.TFRecordWriter` to write to.
    writer_jsonl: An open text file writer to write JSON Lines to.
    metrics: Optional `_MetricAverager`s to update with this prediction. If
      None, metric calculation is skipped completely. None is appropriate for
      example if we're just running inference for unlabeled examples.
  """
  # [kp_max_length, long_max_length] shape.
  ngram_logits = prediction['ngram_logits']

  features = collections.OrderedDict()
  features['ngram_logits'] = input_utils.create_float_feature(
      ngram_logits.flatten())

  position_predictions = eval_utils.logits_to_predictions(
      ngram_logits, max_predictions=FLAGS.max_position_predictions)
  # Sort predictions for convenience.
  position_predictions.sort(key=lambda x: x.logit, reverse=True)
  features['top_pos_logit'] = input_utils.create_float_feature(
      x.logit for x in position_predictions)
  features['top_pos_start_idx'] = input_utils.create_int_feature(
      x.start_idx for x in position_predictions)
  features['top_pos_phrase_len'] = input_utils.create_int_feature(
      x.phrase_len for x in position_predictions)

  url = ''.join(chr(x) for x in prediction['url_code_points'] if x != -1)
  features['url'] = input_utils.create_bytes_feature([url])

  if url in text_examples:
    text_example = text_examples[url]
    kp_predictions = text_example.get_key_phrase_predictions(
        position_predictions, max_predictions=FLAGS.max_kp_predictions)
    if len(kp_predictions) < FLAGS.max_kp_predictions:
      tf.logging.warn(f'Made fewer than `max_kp_predictions` for URL: {url}')
    writer_jsonl.write(
        json.dumps({
            'url': url,
            'KeyPhrases': [[kp] for kp in kp_predictions]
        }) + '\n')

    features['kp_predictions'] = input_utils.create_bytes_feature(
        kp_predictions)

    if metrics is not None:
      precision, recall, f1 = text_example.get_score_full(kp_predictions)
      for i in (1, 3, 5):
        p = precision[i - 1]
        r = recall[i - 1]
        f = f1[i - 1]
        features[f'p_at_{i}'] = input_utils.create_float_feature([p])
        features[f'r_at_{i}'] = input_utils.create_float_feature([r])
        features[f'f1_at_{i}'] = input_utils.create_float_feature([f])
        metrics[i].add_example(precision=p, recall=r, f1=f)
  else:
    tf.logging.error(f'No text example found for URL: {url}')

  writer_tfrecord.write(
      tf.train.Example(features=tf.train.Features(
          feature=features)).SerializeToString())


def make_scalar_summary(tag: Text, value: float) -> tf.summary.Summary:
  """Returns a TF Summary proto for a scalar summary value.

  Args:
    tag: The name of the summary.
    value: The scalar float value of the summary.

  Returns:
    A TF Summary proto.
  """
  return tf.summary.Summary(
      value=[tf.summary.Summary.Value(tag=tag, simple_value=value)])


def main(_):
  _validate_flags()


  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs(FLAGS.output_dir)

  model_config = modeling.EtcConfig.from_json_file(FLAGS.etc_config_file)
  if FLAGS.grad_checkpointing_period is not None:
    model_config.grad_checkpointing_period = FLAGS.grad_checkpointing_period

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          tpu_job_name=FLAGS.tpu_job_name,
          per_host_input_for_training=is_per_host))

  train_steps = None
  warmup_steps = None
  if FLAGS.do_train:
    if not FLAGS.num_train_examples:
      raise ValueError('Must specify `num_train_examples` for training.')
    train_steps = int(FLAGS.num_train_epochs * FLAGS.num_train_examples /
                      FLAGS.batch_size)
    warmup_steps = int(train_steps * FLAGS.warmup_proportion)

  model_fn = run_finetuning_lib.model_fn_builder(model_config, train_steps,
                                                 warmup_steps, FLAGS)

  eval_batch_size = FLAGS.eval_batch_size or FLAGS.batch_size

  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=eval_batch_size,
      predict_batch_size=FLAGS.batch_size)

  training_done_path = os.path.join(FLAGS.output_dir, 'training_done')

  if FLAGS.do_train:
    tf.logging.info('***** Running training *****')
    tf.logging.info('  Num training examples = %d', FLAGS.num_train_examples)
    tf.logging.info('  Batch size = %d', FLAGS.batch_size)
    tf.logging.info('  Num steps = %d', train_steps)
    train_input_fn = run_finetuning_lib.input_fn_builder(
        input_file=FLAGS.input_tfrecord,
        flags=FLAGS,
        model_config=model_config,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=train_steps)

    # Write file to signal training is done.
    with tf.gfile.GFile(training_done_path, 'w') as writer:
      writer.write('\n')

  if FLAGS.do_eval:
    tf.logging.info('***** Running eval *****')
    tf.logging.info('  Batch size = %d', eval_batch_size)

    if FLAGS.eval_fraction_of_removed_examples is None:
      raise ValueError(
          'Must specify `eval_fraction_of_removed_examples` for eval.')
    elif (FLAGS.eval_fraction_of_removed_examples < 0 or
          FLAGS.eval_fraction_of_removed_examples > 1):
      raise ValueError('Invalid `eval_fraction_of_removed_examples`: '
                       f'{FLAGS.eval_fraction_of_removed_examples}')

    eval_input_fn = run_finetuning_lib.input_fn_builder(
        input_file=FLAGS.input_tfrecord,
        flags=FLAGS,
        model_config=model_config,
        is_training=False,
        drop_remainder=False)

    # Load text examples to evaluate key phrases against.
    if FLAGS.eval_text_example_path is None:
      raise ValueError('Must specify `eval_text_example_path` for eval.')
    text_examples = eval_utils.read_text_examples(FLAGS.eval_text_example_path)
    text_examples = {x.url: x for x in text_examples}

    # Writer for TensorBoard.
    summary_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.output_dir, 'eval'))

    for checkpoint_path in tf.train.checkpoints_iterator(
        FLAGS.output_dir, min_interval_secs=15 * 60, timeout=8 * 60 * 60):
      global_step = _get_global_step_for_checkpoint(checkpoint_path)
      processed_examples = 0
      metrics = {key: _MetricAverager() for key in (1, 3, 5)}

      output_file = checkpoint_path + '.predicted-tfrecords.gz'
      output_file_jsonl = checkpoint_path + '.predicted-eval.jsonl'
      gz = tf.python_io.TFRecordOptions(
          tf.python_io.TFRecordCompressionType.GZIP)

      with tf.python_io.TFRecordWriter(output_file, options=gz) as writer:
        with tf.gfile.GFile(output_file_jsonl, 'w') as writer_jsonl:
          for prediction in estimator.predict(
              eval_input_fn, yield_single_examples=True):
            if processed_examples % 1000 == 0:
              tf.logging.info('Processing example: %d' % processed_examples)
            _process_prediction(prediction, text_examples, writer, writer_jsonl,
                                metrics)
            processed_examples += 1

      # Write summaries to TensorBoard.
      deflation_scale = 1 - FLAGS.eval_fraction_of_removed_examples
      for i in (1, 3, 5):
        summary_writer.add_summary(
            make_scalar_summary(
                tag=f'eval_metrics/precision_at_{i}',
                value=metrics[i].precision * deflation_scale),
            global_step=global_step)
        summary_writer.add_summary(
            make_scalar_summary(
                tag=f'eval_metrics/recall_at_{i}',
                value=metrics[i].recall * deflation_scale),
            global_step=global_step)
        summary_writer.add_summary(
            make_scalar_summary(
                tag=f'eval_metrics/f1_at_{i}',
                value=metrics[i].f1 * deflation_scale),
            global_step=global_step)
      summary_writer.flush()

      if tf.io.gfile.exists(training_done_path):
        # Break if the checkpoint we just processed is the last one.
        last_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
        if last_checkpoint is None:
          continue
        last_global_step = _get_global_step_for_checkpoint(last_checkpoint)
        if global_step == last_global_step:
          break

  if FLAGS.do_predict:
    tf.logging.info('***** Running inference *****')
    tf.logging.info('  Batch size = %d', FLAGS.batch_size)

    predict_input_fn = run_finetuning_lib.input_fn_builder(
        input_file=FLAGS.input_tfrecord,
        flags=FLAGS,
        model_config=model_config,
        is_training=False,
        drop_remainder=False)

    checkpoint_path = FLAGS.predict_checkpoint_path
    if checkpoint_path is None:
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)

    # Load text examples to evaluate key phrases against.
    if FLAGS.predict_text_example_path is None:
      raise ValueError(
          'Must specify `predict_text_example_path` for inference.')
    text_examples = eval_utils.read_text_examples(
        FLAGS.predict_text_example_path)
    text_examples = {x.url: x for x in text_examples}

    processed_examples = 0

    output_file = f'{checkpoint_path}.{FLAGS.predict_output_name}.tfrecords.gz'
    output_file_jsonl = f'{checkpoint_path}.{FLAGS.predict_output_name}.jsonl'
    gz = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    with tf.python_io.TFRecordWriter(output_file, options=gz) as writer:
      with tf.gfile.GFile(output_file_jsonl, 'w') as writer_jsonl:
        for prediction in estimator.predict(
            predict_input_fn,
            checkpoint_path=checkpoint_path,
            yield_single_examples=True):
          if processed_examples % 1000 == 0:
            tf.logging.info('Processing example: %d' % processed_examples)
          _process_prediction(
              prediction, text_examples, writer, writer_jsonl, metrics=None)
          processed_examples += 1


if __name__ == '__main__':
  tf.app.run()
