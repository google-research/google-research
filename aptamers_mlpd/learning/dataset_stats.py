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

# Lint as: python3
"""Utilities for calculating dataset statistics."""

import copy
import logging

import contextlib2
import tensorflow.compat.v1 as tf
from xxx import metrics as contrib_metrics
from tensorflow.contrib import labeled_tensor as lt

# Google Internal
import text_format
import gfile

from ..learning import data

logger = logging.getLogger(__name__)


def experiment_has_statistics(experiment_proto):
  """Returns True if the experiment proto has statistics."""
  has_all_statistics = True
  for round_proto in experiment_proto.rounds.values():
    for reads in [round_proto.positive_reads, round_proto.negative_reads]:
      if reads.name:
        if not reads.HasField('statistics'):
          has_all_statistics = False

  for ao_proto in experiment_proto.additional_output:
    if ao_proto.name:
      if not ao_proto.HasField('statistics'):
        has_all_statistics = False
  return has_all_statistics


def compute_experiment_statistics(
    experiment_proto,
    input_paths,
    proto_w_stats_path,
    preprocess_mode=data.PREPROCESS_SKIP_ALL_ZERO_COUNTS,
    max_size=None,
    logdir=None,
    save_stats=False):
  """Calculate the mean and standard deviation of counts from input files.

  These statistics are used for normalization. If any statistic is missing or
  save_stats=True, compute the statistics. Save the statitics to
  proto_w_stats_path if save_stats=True.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.
    input_paths: list of strings giving paths to sstables of input examples.
    proto_w_stats_path: string path to the validation proto file with stats
    preprocess_mode: optional preprocess mode defined in the `data` module.
    max_size: optional number of examples to examine to compute statistics. By
      default, examines the entire dataset.
    logdir: optional path to a directory in which to log events.
    save_stats: optional boolean indicating whether to update all the statistics
      and save to proto_w_stats_path.

  Returns:
    selection_pb2.Experiment with computed statistics.
  """
  experiment_proto = copy.deepcopy(experiment_proto)

  has_all_statistics = True

  all_reads = {}
  for round_proto in experiment_proto.rounds.values():
    for reads in [round_proto.positive_reads, round_proto.negative_reads]:
      if reads.name:
        all_reads[reads.name] = reads
        if not reads.HasField('statistics'):
          has_all_statistics = False

  all_ao = {}
  for ao_proto in experiment_proto.additional_output:
    if ao_proto.name:
      all_ao[ao_proto.name] = ao_proto
      if not ao_proto.HasField('statistics'):
        has_all_statistics = False

  if not has_all_statistics or save_stats:
    with tf.Graph().as_default():
      logger.info('Setting up graph for statistics')
      # we only care about outputs, which don't rely on training hyper
      # parameters
      hps = tf.HParams(
          preprocess_mode=preprocess_mode,
          kmer_k_max=0,
          ratio_random_dna=0.0,
          total_reads_defining_positive=0,
          additional_output=','.join([
              x.name for x in experiment_proto.additional_output]))
      _, outputs = data.input_pipeline(
          input_paths,
          experiment_proto,
          final_mbsz=100000,
          hps=hps,
          num_epochs=1,
          num_threads=1)
      size_op = tf.shape(outputs)[list(outputs.axes.keys()).index('batch')]

      all_update_ops = []
      all_value_ops = {}
      for name in all_reads:
        counts = lt.select(outputs, {'output': name})
        log_counts = lt.log(counts + 1.0)
        ops = {
            'mean': contrib_metrics.streaming_mean(counts),
            'std_dev': streaming_std(counts),
            'mean_log_plus_one': contrib_metrics.streaming_mean(log_counts),
            'std_dev_log_plus_one': streaming_std(log_counts),
        }
        value_ops, update_ops = contrib_metrics.aggregate_metric_map(ops)
        all_update_ops.extend(list(update_ops.values()))
        all_value_ops[name] = value_ops

      for name in all_ao:
        ao = lt.select(outputs, {'output': name})
        log_ao = lt.log(ao + 1.0)
        ops = {
            'mean': contrib_metrics.streaming_mean(ao),
            'std_dev': streaming_std(ao),
            'mean_log_plus_one': contrib_metrics.streaming_mean(log_ao),
            'std_dev_log_plus_one': streaming_std(log_ao),
        }
        value_ops, update_ops = contrib_metrics.aggregate_metric_map(ops)
        all_update_ops.extend(list(update_ops.values()))
        all_value_ops[name] = value_ops

      logger.info('Running statistics ops')
      sv = tf.train.Supervisor(logdir=logdir)
      with sv.managed_session() as sess:
        total = 0
        for results in run_until_exhausted(sv, sess,
                                           [size_op] + all_update_ops):
          total += results[0]
          if max_size is not None and total >= max_size:
            break
        all_statistics = {k: sess.run(v) for k, v in all_value_ops.items()}

      for reads_name, reads in all_reads.items():
        for name, value in all_statistics[reads_name].items():
          setattr(reads.statistics, name, value.item())

      for ao_name, ao in all_ao.items():
        for name, value in all_statistics[ao_name].items():
          setattr(ao.statistics, name, value.item())

      logger.info('Computed statistics: %r', all_statistics)

      if save_stats:
        logger.info('Save the proto with statistics to %s', proto_w_stats_path)
        with open('/tmp/tmp.pbtxt', 'w') as f:
          f.write(text_format.MessageToString(experiment_proto))
        gfile.Copy('/tmp/tmp.pbtxt', proto_w_stats_path, overwrite=True)
  else:
    logger.info('All the statistics exist. Nothing to compute')
  return experiment_proto


def streaming_std(tensor):

  mean_value, mean_update = contrib_metrics.streaming_mean(tensor)
  mean_squared_value, mean_squared_update = contrib_metrics.streaming_mean(
      tf.square(tensor))
  value_op = tf.sqrt(mean_squared_value - tf.square(mean_value))
  update_op = tf.group(mean_update, mean_squared_update)
  return value_op, update_op


def run_until_exhausted(supervisor, session, fetches):
  """Run the given fetches until OutOfRangeError is triggered."""
  with contextlib2.suppress(tf.errors.OutOfRangeError):
    while not supervisor.should_stop():
      yield session.run(fetches)
