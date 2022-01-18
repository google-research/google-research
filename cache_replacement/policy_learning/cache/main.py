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
# pylint: disable=line-too-long
r"""Runs cache simulation.

Example usage:

  python3 -m cache_replacement.policy_learning.cache.main \
    --experiment_base_dir=/tmp \
    --experiment_name=sample_belady_llc \
    --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
    --cache_configs=cache_replacement/policy_learning/cache/configs/eviction_policy/belady.json \
    --memtrace_file=cache_replacement/policy_learning/cache/traces/sample_trace.csv

  Simulates a cache configured by the cache configs with Belady's as the
  replacement policy on the sample trace.
"""
# pylint: enable=line-too-long

import os
from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf
import tqdm
from cache_replacement.policy_learning.cache import cache as cache_mod
from cache_replacement.policy_learning.cache import evict_trace as evict
from cache_replacement.policy_learning.cache import memtrace
from cache_replacement.policy_learning.common import config as cfg
from cache_replacement.policy_learning.common import utils

FLAGS = flags.FLAGS
flags.DEFINE_multi_string(
    "cache_configs",
    [
        "cache_replacement/policy_learning/cache/configs/default.json",  # pylint: disable=line-too-long
        "cache_replacement/policy_learning/cache/configs/eviction_policy/lru.json"  # pylint: disable=line-too-long
    ],
    "List of config paths merged front to back for the cache.")
flags.DEFINE_multi_string(
    "config_bindings", [],
    ("override config with key=value pairs "
     "(e.g., eviction_policy.policy_type=greedy)"))
flags.DEFINE_string(
    "experiment_base_dir", "/tmp/experiments",
    "Base directory to store all experiments in. Should not frequently change.")
flags.DEFINE_string(
    "experiment_name", "unnamed",
    "All data related to this experiment is written to"
    " experiment_base_dir/experiment_name.")
flags.DEFINE_string(
    "memtrace_file",
    "cache_replacement/policy_learning/cache/traces/omnetpp_train.csv",
    "Memory trace file path to use.")
flags.DEFINE_integer(
    "tb_freq", 10000, "Number of cache reads between tensorboard logs.")
flags.DEFINE_integer(
    "warmup_period", int(2e3), "Number of cache reads before recording stats.")
flags.DEFINE_bool(
    "force_overwrite", False,
    ("If true, overwrites directory at "
     " experiment_base_dir/experiment_name if it exists."))


def log_scalar(tb_writer, key, value, step):
  summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
  tb_writer.add_summary(summary, step)


def main(_):
  # Set up experiment directory
  exp_dir = os.path.join(FLAGS.experiment_base_dir, FLAGS.experiment_name)
  utils.create_experiment_directory(exp_dir, FLAGS.force_overwrite)
  tensorboard_dir = os.path.join(exp_dir, "tensorboard")
  tf.disable_eager_execution()
  tb_writer = tf.summary.FileWriter(tensorboard_dir)
  miss_trace_path = os.path.join(exp_dir, "misses.csv")
  evict_trace_path = os.path.join(exp_dir, "evictions.txt")

  cache_config = cfg.Config.from_files_and_bindings(
      FLAGS.cache_configs, FLAGS.config_bindings)
  with open(os.path.join(exp_dir, "cache_config.json"), "w") as f:
    cache_config.to_file(f)

  flags_config = cfg.Config({
      "memtrace_file": FLAGS.memtrace_file,
      "tb_freq": FLAGS.tb_freq,
      "warmup_period": FLAGS.warmup_period,
  })
  with open(os.path.join(exp_dir, "flags.json"), "w") as f:
    flags_config.to_file(f)

  logging.info("Config: %s", str(cache_config))
  logging.info("Flags: %s", str(flags_config))

  cache_line_size = cache_config.get("cache_line_size")
  with memtrace.MemoryTrace(
      FLAGS.memtrace_file, cache_line_size=cache_line_size) as trace:
    with memtrace.MemoryTraceWriter(miss_trace_path) as write_trace:
      with evict.EvictionTrace(evict_trace_path, False) as evict_trace:
        def write_to_eviction_trace(cache_access, eviction_decision):
          evict_trace.write(
              evict.EvictionEntry(cache_access, eviction_decision))

        cache = cache_mod.Cache.from_config(cache_config, trace=trace)

        # Warm up cache
        for _ in tqdm.tqdm(range(FLAGS.warmup_period), desc="Warming up cache"):
          pc, address = trace.next()
          hit = cache.read(pc, address, [write_to_eviction_trace])

          if not hit:
            write_trace.write(pc, address)

          if trace.done():
            raise ValueError()

        # Discard warm-up cache statistics
        cache.hit_rate_statistic.reset()

        num_reads = 0
        with tqdm.tqdm(desc="Simulating cache on MemoryTrace") as pbar:
          while not trace.done():
            pc, address = trace.next()
            hit = cache.read(pc, address, [write_to_eviction_trace])

            if not hit:
              write_trace.write(pc, address)

            num_reads += 1
            if num_reads % FLAGS.tb_freq == 0:
              log_scalar(tb_writer, "cache_hit_rate",
                         cache.hit_rate_statistic.success_rate(), num_reads)

            pbar.update(1)

          log_scalar(tb_writer, "cache_hit_rate",
                     cache.hit_rate_statistic.success_rate(), num_reads)

  # Force flush, otherwise last writes will be lost.
  tb_writer.flush()

if __name__ == "__main__":
  app.run(main)
