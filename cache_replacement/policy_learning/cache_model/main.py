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

# pylint: disable=line-too-long
r"""Trains a model on an eviction trace.

Example Usage:

  python3 -m cache_replacement.policy_learning.cache_model.main \
    --experiment_base_dir=/tmp \
    --experiment_name=sample_model_llc \
    --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
    --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
    --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
    --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv

Trains full model with all additions on trained and validated on the sample trace.
"""
# pylint: enable=line-too-long

import io
import os
from absl import app
from absl import flags
from absl import logging
from baselines.common import schedules
import numpy as np
import prettytable
import tensorflow.compat.v1 as tf
import torch
from torch import optim
import tqdm
from cache_replacement.policy_learning.cache import cache as cache_mod
from cache_replacement.policy_learning.cache import evict_trace
from cache_replacement.policy_learning.cache import eviction_policy
from cache_replacement.policy_learning.cache import memtrace
from cache_replacement.policy_learning.cache_model import eviction_policy as model_eviction_policy
from cache_replacement.policy_learning.cache_model import metric
from cache_replacement.policy_learning.cache_model import model
from cache_replacement.policy_learning.cache_model import utils
from cache_replacement.policy_learning.common import config as cfg
from cache_replacement.policy_learning.common import utils as common_utils


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "train_memtrace",
    "cache_replacement/policy_learning/cache/traces/omnetpp_train.csv",
    "Path to the training memory trace.")
flags.DEFINE_string(
    "valid_memtrace",
    "cache_replacement/policy_learning/cache/traces/omnetpp_valid.csv",
    "Path to the validation memory trace.")
flags.DEFINE_string(
    "experiment_base_dir", "/tmp/experiments",
    "Base directory to store all experiments in. Should not frequently change.")
flags.DEFINE_string(
    "experiment_name", "unnamed",
    "All data related to this experiment is written to"
    " experiment_base_dir/experiment_name.")
flags.DEFINE_integer("batch_size", 32, "Size of model input batches.")
flags.DEFINE_integer(
    "total_steps", int(1e6), "Number of training steps to take.")
flags.DEFINE_integer("tb_freq", 100, "Steps between logging to tensorboard.")
flags.DEFINE_integer(
    "small_eval_size", 30000,
    "Number of examples to evaluate on in small evaluations.")
flags.DEFINE_integer(
    "small_eval_freq", 4000,
    "Steps between evaluating on small_eval_size examples of validation data.")
flags.DEFINE_integer(
    "full_eval_freq", 48000,
    "Steps between evaluating on ALL of validation data.")
flags.DEFINE_integer(
    "save_freq", 20000, "Steps between saving model checkpoints.")
flags.DEFINE_integer(
    "collection_multiplier", 5,
    ("This times more training data is collected than updated on to avoid"
     " temporal correlations."))
flags.DEFINE_bool(
    "force_overwrite", False,
    ("If true, overwrites directory at "
     " experiment_base_dir/experiment_name if it exists."))
flags.DEFINE_multi_string(
    "model_configs",
    ["cache_replacement/policy_learning/cache_model/configs/default.json"],  # pylint: disable=line-too-long
    ("List of config filenames merged front to back for the model. "
     "See config.Config.merge for merging behavior."))
flags.DEFINE_multi_string(
    "model_bindings", [], "bindings to override model config.")
flags.DEFINE_multi_string(
    "cache_configs",
    ["cache_replacement/policy_learning/cache/configs/default.json"],  # pylint: disable=line-too-long
    "List of config filenames merged front to back for the cache.")
flags.DEFINE_multi_string(
    "cache_bindings", [], "bindings to override cache config.")
flags.DEFINE_multi_string(
    "dagger_schedule_configs",
    ["cache_replacement/policy_learning/cache_model/configs/schedule/linear.json"],  # pylint: disable=line-too-long
    "List of config filenames for the DAgger schedule.")
flags.DEFINE_multi_string(
    "dagger_schedule_bindings", [], "bindings to override DAgger config.")
flags.DEFINE_enum(
    "oracle_eviction_policy", "belady", ["lru", "belady"],
    "Which eviction policy to learn.")
flags.DEFINE_integer(
    "seed", 0, "Determines which random seed to use.")


def schedule_from_config(config):
  if config.get("type") == "linear":
    return schedules.LinearSchedule(
        config.get("num_steps"), config.get("final"), config.get("initial"))
  elif config.get("type") == "constant":
    return schedules.ConstantSchedule(config.get("value"))
  else:
    raise ValueError("Unsupported schedule type: {}".format(config.get("type")))


def evaluate(policy_model, data, step, descriptor, tb_writer, log_dir, k=5):
  """Computes metrics about the model on the data and logs them to tensorboard.

  Args:
    policy_model (EvictionPolicyModel): model to evaluate with.
    data (list[EvictionEntry]): consecutive eviction entries.
    step (int): the current step in training. Affects how things are written to
      tensorboard and where predictions are logged.
    descriptor (str): description of the data. Predictions are written to
      log_dir/descriptor-step.txt and metrics are logged to tensorboard with
      descriptor as the tb prefix.
    tb_writer (FileWriter): all tensorboard metrics are written here.
    log_dir (str): the directory for writing txt file predictions.
    k (int): computes top-1 to top-k success rates.

  Returns:
    metrics (list[CacheEvictionMetric]): metrics tracked during evaluation.
  """
  def pretty_print(entry, probs, attention, pred_reuse_distances):
    """Returns a human-readable string of the entry, probs, and attention.

    Args:
      entry (EvictionEntry): entry to print
      probs (torch.FloatTensor): probs of evicting entry's eviction candidates.
      attention (list[torch.FloatTensor, CacheAccess]): pairs of attention
        weight on past EvictionEntry sorted from most distant to most recent.
      pred_reuse_distances (torch.FloatTensor): the predicted reuse distance of
        each cache line in the entry of shape (num_cache_lines).

    Returns:
      string: error analysis string for probs and attention on the entry.
    """
    cache_access = entry.cache_access
    eviction_decision = entry.eviction_decision

    s = ["PC: {}\nAddress: {}\nEvict: {}\nCache lines:\n"
         .format(hex(cache_access.pc), hex(cache_access.address),
                 eviction_decision.evict)]

    _, true_ranks = probs.sort(descending=True)
    true_rank_to_pred_rank = dict(
        zip(true_ranks.cpu().data.numpy(), range(len(true_ranks))))
    headers = ["true rank", "pc", "address", "pred rank", "prob",
               "oracle score", "pred reuse distance", "rank correct?",
               "in history?"]
    cache_lines_table = prettytable.PrettyTable(headers)
    for i, (line, prob, pred_reuse) in enumerate(
        zip(cache_access.cache_lines, probs, pred_reuse_distances)):
      cand, pc = line
      pred_rank = true_rank_to_pred_rank[i]
      success = "SUCCESS" if pred_rank == i else "FAILURE"

      present_in_history = any(
          cand == prev_access.address for _, prev_access in attention)
      present = "PRESENT" if present_in_history else "ABSENT"
      cache_lines_table.add_row(
          [i, hex(pc), hex(cand), pred_rank, "{:.2f}".format(prob),
           "{:.2f}".format(eviction_decision.cache_line_scores[cand]),
           "{:.2f}".format(pred_reuse.item()), success, present])
    s.append(str(cache_lines_table))
    s.append("\n")

    s.append("Attention:\n")
    num_cache_lines = len(cache_access.cache_lines)
    headers = (
        ["timestep", "pc", "address"] + ["line {}".format(i) for i in range(
            num_cache_lines)])
    attention_table = prettytable.PrettyTable(headers)
    for i, (attention_weights, access) in enumerate(reversed(attention)):
      # Truncate padded attention to num_cache_lines
      attention_entries = ["{:.2f}".format(weight) for weight in
                           attention_weights[:num_cache_lines]]
      row = (["t - {}".format(i), hex(access.pc), hex(access.address)] +
             attention_entries)
      attention_table.add_row(row)
    s.append(str(attention_table))
    s.append("\n")
    return "".join(s)

  # Chop into batch_size parallel sequences
  subseq_length = len(data) // FLAGS.batch_size
  # (batch_size, subseq_length)
  subsequences = [data[i * subseq_length: (i + 1) * subseq_length] for i in
                  range(FLAGS.batch_size)]

  # Separate lists to concatenate at the end
  logs = [[] for _ in range(FLAGS.batch_size)]
  hidden_state = None
  metrics = [metric.SuccessRateMetric(k), metric.KendallWeightedTau(),
             metric.OracleScoreGap()]
  desc = "Evaluating for {}".format(descriptor)
  for batch in tqdm.tqdm(zip(*subsequences), desc=desc, total=subseq_length):
    probs, pred_reuse_distances, hidden_state, attention = policy_model(
        [entry.cache_access for entry in batch], hidden_state, inference=True)

    eviction_mask = torch.tensor(
        [entry.eviction_decision.evict for entry in batch])
    oracle_scores = []
    for entry in batch:
      # Set min score to -1e10 to handle -np.infs
      oracle_scores.append(np.maximum(
          [entry.eviction_decision.cache_line_scores[line]
           for line, _ in entry.cache_access.cache_lines], -1e10))
    for m in metrics:
      m.update(probs, eviction_mask, oracle_scores)

    for i in range(FLAGS.batch_size):
      logs[i].append(pretty_print(
          batch[i], probs[i], list(next(attention)), pred_reuse_distances[i]))

  filename = os.path.join(log_dir, "{}-{}.txt".format(descriptor, step))
  with open(filename, "w") as f:
    for log in logs:
      f.writelines(entry + "\n" for entry in log)

  for m in metrics:
    m.write_to_tensorboard(tb_writer, descriptor, step)
  return metrics


def measure_cache_hit_rate(
    memtrace_path, cache_config, eviction_model, model_prob_schedule,
    get_step, eviction_trace_path, max_examples=None, use_oracle_scores=True,
    k=5):
  """Measures the hit rate on the memtrace, returning the eviction entries.

  Passes through the entire memory trace and returns eviction entries
  obtained by following the model prediction model_prob of the time, and
  following the oracle eviction policy 1 - model_prob of the time.
  Passes through the memory trace in max_examples chunks at a time.

  Args:
    memtrace_path (str): path to the memory trace to simulate through.
    cache_config (Config): configures the cache to simulate with.
    eviction_model (EvictionPolicyModel): the model whose predictions to follow
      model_prob of the time
    model_prob_schedule (Schedule): returns the portion of the time to follow
      model predictions.
    get_step (Callable): called with no arguments produces the current step
      number.
    eviction_trace_path (str): all simulated eviction entries are written to
      eviction_trace_path.format(get_step()).
    max_examples (int | None): collects at most this many examples, if provided.
      Otherwise, collects the entire memory trace.
    use_oracle_scores (bool): If True, all returned eviction entries are labeled
      with oracle scores, even when the model is followed.
    k (int): see return value.

  Yields:
    entries (dict): maps set id (int) to sequence of consecutive eviction
      entries of that set id (list[EvictionEntry]).
    cache_hit_rates (list[float]): the cache hit rates on the first 1 / k,
      2 / k, ..., k / k portions of the current chunk of max_examples from the
      memtrace.
  """
  if max_examples is None:
    max_examples = np.inf

  line_size = cache_config.get("cache_line_size")
  with memtrace.MemoryTrace(memtrace_path, cache_line_size=line_size) as trace:
    def create_eviction_policy(model_prob):
      """Creates the appropriate eviction policy for collecting data.

      Args:
        model_prob (float): the returned policy is a mixture of the learned
        eviction policy and the oracle policy, where the learned eviction policy
        is played model_prob of the time.

      Returns:
        EvictionPolicy
      """
      # Need to update the eviction policy and keep the cache state around
      oracle_scorer = {
          "lru": eviction_policy.LRUScorer(),
          "belady": eviction_policy.BeladyScorer(trace),
      }[FLAGS.oracle_eviction_policy]
      learned_scorer = model_eviction_policy.LearnedScorer(eviction_model)

      # Use scoring_policy_index = 0 to always get scores from the oracle scorer
      scoring_policy_index = 0 if use_oracle_scores else None
      return eviction_policy.MixturePolicy(
          [eviction_policy.GreedyEvictionPolicy(oracle_scorer),
           eviction_policy.GreedyEvictionPolicy(learned_scorer)],
          [1 - model_prob, model_prob],
          scoring_policy_index=scoring_policy_index,
      )

    # This eviction policy isn't actually used (immediately overwritten
    # below), but we need to pass something that is not None.
    policy = create_eviction_policy(model_prob_schedule.value(get_step()))
    cache = cache_mod.Cache.from_config(
        cache_config, eviction_policy=policy, trace=trace)

    addresses = set()
    pcs = set()
    desc = "Collecting data from {} with mixture parameter: {}".format(
        memtrace_path, model_prob_schedule.value(get_step()))
    with tqdm.tqdm(desc=desc) as pbar:
      while not trace.done():
        data = []
        hit_rates = []
        model_prob = model_prob_schedule.value(get_step())
        cache.set_eviction_policy(create_eviction_policy(model_prob))
        # discard stats from previous iterations
        cache.hit_rate_statistic.reset()
        eviction_trace_path = eviction_trace_path.format(get_step())
        with evict_trace.EvictionTrace(eviction_trace_path, False) as etrace:
          def add_to_data(cache_access, eviction_decision):
            entry = evict_trace.EvictionEntry(cache_access, eviction_decision)
            data.append(entry)
            # Add here for cache line aligned address
            addresses.add(cache_access.address)
            entry = evict_trace.EvictionEntry(cache_access, eviction_decision)
            etrace.write(entry)

          while len(data) < max_examples and not trace.done():
            pc, address = trace.next()
            pcs.add(pc)
            cache.read(pc, address, [add_to_data])
            hit_rates.append(cache.hit_rate_statistic.success_rate())
            pbar.update(1)

        # Post-filter here, since length is unknown if max_examples is not
        # provided (or trace terminates earlier than max_examples).
        skip_len = len(hit_rates) // k
        hit_rates = (hit_rates[skip_len:skip_len * (k - 1) + 1:skip_len] +
                     [hit_rates[-1]])
        yield data, hit_rates

    logging.info("Number of unique addresses: %d", len(addresses))
    logging.info("Number of unique pcs: %d", len(pcs))


def log_hit_rates(tb_writer, tb_key, hit_rates, step):
  """Logs list of cumulative hit rates to tensorboard.

  Args:
    tb_writer (FileWriter): used to log.
    tb_key (str): used as the tensorboard key.
    hit_rates (list[float]): the hit rates to log. Assumed that hit_rates[i] is
      the cumulative hit rate on the first i / len(hit_rates) portion of the
      data.
    step (int): step number to use in tensorboard.
  """
  for i, hit_rate in enumerate(hit_rates[:-1]):
    utils.log_scalar(
        tb_writer, tb_key + "_{:.2f}".format((i + 1) / len(hit_rates)),
        hit_rate, step)
  utils.log_scalar(tb_writer, tb_key, hit_rates[-1], step)


def main(_):
  logging.info("Seed: %d", FLAGS.seed)
  np.random.seed(FLAGS.seed)
  torch.random.manual_seed(FLAGS.seed)

  if FLAGS.save_freq % FLAGS.small_eval_freq != 0:
    raise ValueError(
        ("Save frequency ({}) must be a multiple of evaluation frequency ({})."
         " Allows choosing checkpoints based on their evaluation scores.")
        .format(FLAGS.save_freq, FLAGS.small_eval_freq))

  if FLAGS.full_eval_freq % FLAGS.small_eval_freq != 0:
    raise ValueError(
        ("Full evaluation frequency ({}) must be a multiple of small"
         " evaluation frequency ({}) so that their values can be compared.")
        .format(FLAGS.full_eval_freq, FLAGS.small_eval_freq))

  exp_dir = os.path.join(FLAGS.experiment_base_dir, FLAGS.experiment_name)
  common_utils.create_experiment_directory(exp_dir, FLAGS.force_overwrite)
  tensorboard_dir = os.path.join(exp_dir, "tensorboard")
  tf.disable_eager_execution()
  tb_writer = tf.summary.FileWriter(tensorboard_dir)

  predictions_dir = os.path.join(exp_dir, "predictions")
  os.makedirs(predictions_dir, exist_ok=True)

  checkpoints_dir = os.path.join(exp_dir, "checkpoints")
  os.makedirs(checkpoints_dir, exist_ok=True)

  evict_trace_dir = os.path.join(exp_dir, "evictions")
  os.makedirs(evict_trace_dir, exist_ok=True)

  model_config = cfg.Config.from_files_and_bindings(
      FLAGS.model_configs, FLAGS.model_bindings)
  logging.info("Model config: %s", model_config)
  with open(os.path.join(exp_dir, "model_config.json"), "w") as f:
    model_config.to_file(f)

  cache_config = cfg.Config.from_files_and_bindings(
      FLAGS.cache_configs, FLAGS.cache_bindings)
  logging.info("Cache config: %s", cache_config)
  with open(os.path.join(exp_dir, "cache_config.json"), "w") as f:
    cache_config.to_file(f)

  dagger_schedule_config = cfg.Config.from_files_and_bindings(
      FLAGS.dagger_schedule_configs, FLAGS.dagger_schedule_bindings)
  logging.info("DAgger config: %s", dagger_schedule_config)
  with open(os.path.join(exp_dir, "dagger_config.json"), "w") as f:
    dagger_schedule_config.to_file(f)
  dagger_schedule = schedule_from_config(dagger_schedule_config)

  # Process everything on GPU if available
  device = torch.device("cpu")
  if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0")
  logging.info("Device: %s", device)

  policy_model = model.EvictionPolicyModel.from_config(model_config).to(device)
  optimizer = optim.Adam(policy_model.parameters(), lr=model_config.get("lr"))

  step = 0
  get_step = lambda: step
  oracle_valid_data, hit_rates = next(measure_cache_hit_rate(
      FLAGS.valid_memtrace, cache_config, policy_model,
      schedules.ConstantSchedule(0), get_step,
      os.path.join(evict_trace_dir, "oracle_valid.txt")))
  log_hit_rates(tb_writer, "cache_hit_rate/oracle_valid", hit_rates, step)

  with tqdm.tqdm(total=FLAGS.total_steps) as pbar:
    while True:  # loop for waiting until steps == FLAGS.total_steps
      # Optimization: Instead of passing through the whole memory trace for
      # training and only using update_freq many of them, we lazily gather k *
      # update_freq batches and still train on a subsample of update_freq.
      # The value of k=collection_multiplier trades off between:
      #   - The set of k * update_freq examples are all consecutive in the
      #   memory trace. As k gets small, the set of these examples becomes less
      #   i.i.d., as they are temporally correlated. The examples cannot be
      #   random access within the memory trace, since at time t, we require the
      #   previous cache accesses to compute the cache state at time t.
      #   - As k gets large, training becomes slower, as we must perform k times
      #   as much collecting work than training work.
      max_examples = (dagger_schedule_config.get("update_freq") *
                      FLAGS.collection_multiplier * FLAGS.batch_size)
      train_data_generator = measure_cache_hit_rate(
          FLAGS.train_memtrace, cache_config, policy_model, dagger_schedule,
          get_step, os.path.join(evict_trace_dir, "mixture-train-{}.txt"),
          max_examples=max_examples)
      for train_data, hit_rates in train_data_generator:
        log_hit_rates(
            tb_writer, "cache_hit_rate/train_mixture_policy", hit_rates, step)
        utils.log_scalar(
            tb_writer, "cache_hit_rate/mixture_parameter",
            dagger_schedule.value(step), step)

        for batch_num, batch in enumerate(utils.as_batches(
            [train_data], FLAGS.batch_size,
            model_config.get("sequence_length"))):
          def evaluate_helper(eval_size, suffix):
            """Evaluates the model on train / valid data on and off-policy.

            Args:
              eval_size (int): the number of examples to evaluate on.
              suffix (str): appended to all logging and tensorboard paths.
            """
            evaluate(policy_model, oracle_valid_data[-eval_size:], step,
                     "off_policy_valid" + suffix, tb_writer, predictions_dir)
            # train_data is defined in the loop, but evaluate_helper is only
            # called in the same loop iteration.
            # pylint: disable=cell-var-from-loop
            evaluate(policy_model, train_data[-eval_size:],
                     step, "train" + suffix, tb_writer, predictions_dir)
            # pylint: enable=cell-var-from-loop

            # Log the cache hit rates on portions of train / valid
            _, hit_rates = next(measure_cache_hit_rate(
                FLAGS.train_memtrace, cache_config, policy_model,
                schedules.ConstantSchedule(1), get_step,
                os.path.join(
                    evict_trace_dir, "train{}-{}.txt".format(suffix, step)),
                max_examples=eval_size, use_oracle_scores=False))
            log_hit_rates(
                tb_writer, "cache_hit_rate/train" + suffix, hit_rates, step)

            # Use oracle scores, since eviction trace in log_evaluate_stats will
            # log with on-policy scores.
            on_policy_valid_data, hit_rates = next(measure_cache_hit_rate(
                FLAGS.valid_memtrace, cache_config, policy_model,
                schedules.ConstantSchedule(1), get_step,
                os.path.join(
                    evict_trace_dir, "valid{}-{}.txt".format(suffix, step)),
                max_examples=eval_size))
            log_hit_rates(
                tb_writer, "cache_hit_rate/valid" + suffix, hit_rates, step)
            evaluate(policy_model, on_policy_valid_data[-eval_size:], step,
                     "on_policy_valid" + suffix, tb_writer, predictions_dir)

          if step % FLAGS.small_eval_freq == 0:
            evaluate_helper(FLAGS.small_eval_size, "")

          if step % FLAGS.full_eval_freq == 0:
            evaluate_helper(len(oracle_valid_data), "_full")

          if step % FLAGS.save_freq == 0 and step != 0:
            save_path = os.path.join(checkpoints_dir, "{}.ckpt".format(step))
            with open(save_path, "wb") as save_file:
              checkpoint_buffer = io.BytesIO()
              torch.save(policy_model.state_dict(), checkpoint_buffer)
              logging.info("Saving model checkpoint to: %s", save_path)
              save_file.write(checkpoint_buffer.getvalue())

          optimizer.zero_grad()
          losses = policy_model.loss(
              batch, model_config.get("sequence_length") // 2)
          total_loss = sum(losses.values())
          total_loss.backward()
          optimizer.step()
          pbar.update(1)
          step += 1

          if step % FLAGS.tb_freq == 0:
            utils.log_scalar(tb_writer, "loss/total", total_loss, step)
            for loss_name, loss_value in losses.items():
              utils.log_scalar(
                  tb_writer, "loss/{}".format(loss_name), loss_value, step)

          if step == FLAGS.total_steps:
            return

          # Break out of inner-loop to get next set of k * update_freq batches
          if batch_num == dagger_schedule_config.get("update_freq"):
            break


if __name__ == "__main__":
  app.run(main)
