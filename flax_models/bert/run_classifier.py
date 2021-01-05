# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Run sequence-level classification (and regression) fine-tuning."""

import collections
import functools
import os
from typing import DefaultDict, List, Tuple

from absl import logging
from flax import nn
from flax import optim
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from tensorflow.io import gfile
import tensorflow_datasets as tfds

from flax_models.bert import import_weights
from flax_models.bert import input_pipeline
from flax_models.bert import models
from flax_models.bert import train_utils
import sentencepiece as spm


def create_model(config, num_classes=2):
  """Create a model, starting with a pre-trained checkpoint."""
  model_kwargs = dict(
      config=config,
      n_classes=num_classes,
  )
  model_def = models.BertForSequenceClassification.partial(**model_kwargs)
  if "init_checkpoint" in config and config.init_checkpoint:
    initial_params = import_weights.load_params_from_tf(
        init_checkpoint=config.init_checkpoint,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_classes=num_classes)
  else:
    with nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = model_def.init_by_shape(
          jax.random.PRNGKey(0),
          [((1, config.max_seq_length), jnp.int32),
           ((1, config.max_seq_length), jnp.int32),
           ((1, config.max_seq_length), jnp.int32),
           ((1, 1), jnp.int32)],
          deterministic=True)
  model = nn.Model(model_def, initial_params)
  return model


def create_optimizer(config, model):
  optimizer_def = optim.Adam(
      learning_rate=config.learning_rate,
      beta1=0.9,
      beta2=0.999,
      eps=1e-6,
      weight_decay=0.0)
  optimizer = optimizer_def.create(model)
  return optimizer


def compute_loss_and_metrics(model, batch, rng):
  """Compute cross-entropy loss for classification tasks."""
  with nn.stochastic(rng):
    metrics = model(
        batch["input_ids"],
        (batch["input_ids"] > 0).astype(np.int32),
        batch["type_ids"],
        batch["label"])
  return metrics["loss"], metrics


def compute_classification_stats(model, batch):
  with nn.stochastic(jax.random.PRNGKey(0)):
    y = model(
        batch["input_ids"],
        (batch["input_ids"] > 0).astype(np.int32),
        batch["type_ids"],
        deterministic=True)
  return {
      "idx": batch["idx"],
      "label": batch["label"],
      "prediction": y.argmax(-1)
  }


def compute_regression_stats(model, batch):
  with nn.stochastic(jax.random.PRNGKey(0)):
    y = model(
        batch["input_ids"],
        (batch["input_ids"] > 0).astype(np.int32),
        batch["type_ids"],
        deterministic=True)
  return {
      "idx": batch["idx"],
      "label": batch["label"],
      "prediction": y[..., 0],
  }


def create_eval_metrics_fn(dataset_name, is_regression_task):
  """Create a function that computes task-relevant metrics."""
  def get_accuracy(guess, gold):
    return (guess == gold).mean()

  def get_mcc(guess, gold):
    tp = ((guess == 1) & (gold == 1)).sum()
    tn = ((guess == 0) & (gold == 0)).sum()
    fp = ((guess == 1) & (gold == 0)).sum()
    fn = ((guess == 0) & (gold == 1)).sum()
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / (mcc_denom + 1e-6)
    return mcc

  def get_f1(guess, gold):
    tp = ((guess == 1) & (gold == 1)).sum()
    fp = ((guess == 1) & (gold == 0)).sum()
    fn = ((guess == 0) & (gold == 1)).sum()
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    return f1

  def get_f1_accuracy_mean(guess, gold):
    return (get_f1(guess, gold) + get_accuracy(guess, gold)) / 2.0

  def get_pearsonr(x, y):
    return np.corrcoef(x, y)[0, 1]

  eval_metrics = {}
  if is_regression_task:
    eval_metrics["pearsonr"] = get_pearsonr
  else:
    eval_metrics["accuracy"] = get_accuracy

  if dataset_name == "glue/cola":
    eval_metrics["mcc"] = get_mcc
  elif dataset_name in ("glue/mrpc", "glue/qqp"):
    eval_metrics["f1_accuracy_mean"] = get_f1_accuracy_mean

  def metrics_fn(stats):
    res = {}
    for name, fn in eval_metrics.items():
      res[name] = fn(stats["label"], stats["prediction"])
    return res

  return metrics_fn


def get_splits_and_tsvs(dataset_name: str) -> List[Tuple[tfds.Split, str]]:
  return {
      "glue/cola": [(tfds.Split.TEST, "CoLA.tsv")],
      "glue/mrpc": [(tfds.Split.TEST, "MRPC.tsv")],
      "glue/qqp": [(tfds.Split.TEST, "QQP.tsv")],
      "glue/sst2": [(tfds.Split.TEST, "SST-2.tsv")],
      "glue/stsb": [(tfds.Split.TEST, "STS-B.tsv")],
      "glue/mnli": [("test_matched", "MNLI-m.tsv"),
                    ("test_mismatched", "MNLI-mm.tsv")],
      "glue/qnli": [(tfds.Split.TEST, "QNLI.tsv")],
      "glue/rte": [(tfds.Split.TEST, "RTE.tsv")],
      # No eval on WNLI for now. BERT accuracy on WNLI is below baseline,
      # unless a special training recipe is used.
      # "glue/wnli": (tfds.Split.TEST, "WNLI.tsv"),
  }[dataset_name]


def get_label_sets(dataset_name: str) -> DefaultDict[str, List[str]]:
  return collections.defaultdict(
      list, {
          "glue/cola": ["0", "1"],
          "glue/mrpc": ["0", "1"],
          "glue/qqp": ["0", "1"],
          "glue/sst2": ["0", "1"],
          "glue/mnli": ["entailment", "neutral", "contradiction"],
          "glue/qnli": ["entailment", "not_entailment"],
          "glue/rte": ["entailment", "not_entailment"],
      })[dataset_name]


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  ds_info = tfds.builder(config.dataset_name).info
  num_train_examples = ds_info.splits[tfds.Split.TRAIN].num_examples
  # TODO(marcvanzee): I added this so we can do a test that does only 1 train
  # step, but we should find a nicer way of doing this.
  if "num_train_steps" in config:
    num_train_steps = config.num_train_steps
  else:
    num_train_steps = int(num_train_examples * config.num_train_epochs //
                          config.train_batch_size)
  warmup_steps = int(config.warmup_proportion * num_train_steps)
  cooldown_steps = num_train_steps - warmup_steps

  is_regression_task = (ds_info.features.dtype["label"] == np.float32)
  if is_regression_task:
    num_classes = 1
    compute_stats = compute_regression_stats
  else:
    num_classes = ds_info.features["label"].num_classes
    compute_stats = compute_classification_stats

  model = create_model(config, num_classes=num_classes)
  optimizer = create_optimizer(config, model)
  optimizer = optimizer.replicate()
  del model  # don't keep a copy of the initial model

  tokenizer = spm.SentencePieceProcessor()
  if "vocab_file" in config:
    tokenizer.Load(config.vocab_file)
  tokenizer.SetEncodeExtraOptions("bos:eos")  # Auto-add [CLS] and [SEP] tokens
  glue_inputs = functools.partial(
      input_pipeline.glue_inputs, dataset_name=config.dataset_name,
      max_len=config.max_seq_length, tokenizer=tokenizer)

  learning_rate_fn = train_utils.create_learning_rate_scheduler(
      factors="constant * linear_warmup * cosine_decay",
      base_learning_rate=config.learning_rate,
      warmup_steps=warmup_steps,
      steps_per_cycle=cooldown_steps,
  )

  # Replace "GLUE/dataset" with "GLUE_dataset" for the path.
  output_dir = os.path.join(workdir, config.dataset_name.replace("/", "_"))
  gfile.makedirs(output_dir)

  train_history = train_utils.TrainStateHistory(learning_rate_fn)
  train_state = train_history.initial_state()

  if config.do_train:
    train_step_fn = train_utils.create_train_step(compute_loss_and_metrics)
    train_iter = glue_inputs(
        split=tfds.Split.TRAIN, batch_size=config.train_batch_size,
        training=True)

    for step in range(0, num_train_steps):
      with jax.profiler.StepTraceContext("train", step_num=step):
        batch = next(train_iter)
        optimizer, train_state = train_step_fn(optimizer, batch, train_state)

  if config.do_eval:
    eval_step = train_utils.create_eval_fn(compute_stats)
    eval_metrics_fn = create_eval_metrics_fn(
        config.dataset_name, is_regression_task)
    eval_results = []

    if config.dataset_name == "glue/mnli":
      validation_splits = ["validation_matched", "validation_mismatched"]
    else:
      validation_splits = [tfds.Split.VALIDATION]

    for split in validation_splits:
      eval_iter = glue_inputs(
          split=split, batch_size=config.eval_batch_size, training=False)
      eval_stats = eval_step(optimizer, eval_iter)
      eval_metrics = eval_metrics_fn(eval_stats)
      prefix = "eval_mismatched" if split == "validation_mismatched" else "eval"
      for name, val in sorted(eval_metrics.items()):
        line = f"{prefix}_{name} = {val:.06f}"
        print(line, flush=True)
        logging.info(line)
        eval_results.append(line)

    eval_results_path = os.path.join(output_dir, "eval_results.txt")
    with gfile.GFile(eval_results_path, "w") as f:
      for line in eval_results:
        f.write(line + "\n")

  if config.do_predict:
    predict_step = train_utils.create_eval_fn(compute_stats)

    for split, tsv_file in get_splits_and_tsvs(config.dataset_name):
      predict_iter = glue_inputs(
          split=split, batch_size=config.eval_batch_size, training=False)
      predict_stats = predict_step(optimizer, predict_iter)
      idxs = predict_stats["idx"]
      predictions = predict_stats["prediction"]

      tsv_path = os.path.join(output_dir, tsv_file)
      with gfile.GFile(tsv_path, "w") as f:
        f.write("index\tprediction\n")
        if config.dataset_name == "glue/stsb":
          for idx, val in zip(idxs, predictions):
            f.write(f"{idx}\t{val:.06f}\n")
        else:
          label_set = get_label_sets(config.dataset_name)
          for idx, val in zip(idxs, predictions):
            f.write(f"{idx}\t{label_set[val]}\n")
      logging.info("Wrote %s", tsv_path)
