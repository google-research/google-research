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

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import sys
import tensorflow as tf

from bam import configure
from bam.bert import modeling
from bam.bert import optimization
from bam.data import preprocessing
from bam.data import task_weighting
from bam.helpers import training_utils
from bam.helpers import utils
from bam.task_specific import task_builder
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu


class MultitaskModel(object):
  """A multi-task model built on top of BERT."""

  def __init__(self, config, tasks, task_weights, is_training,
               features, num_train_steps):
    # Create a shared BERT encoder
    bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)
    if config.debug:
      bert_config.num_hidden_layers = 3
      bert_config.hidden_size = 144
    assert config.max_seq_length <= bert_config.max_position_embeddings
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      bert_model = modeling.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=features["input_ids"],
          input_mask=features["input_mask"],
          token_type_ids=features["segment_ids"],
          use_one_hot_embeddings=config.use_tpu)
    percent_done = (tf.to_float(tf.train.get_or_create_global_step()) /
                    tf.to_float(num_train_steps))

    # Add specific tasks
    self.outputs = {"task_id": features["task_id"]}
    losses = []
    for task in tasks:
      with tf.variable_scope("task_specific/" + task.name):
        task_losses, task_outputs = task.get_prediction_module(
            bert_model, features, is_training, percent_done)
        losses.append(task_losses * task_weights[task.name])
        self.outputs[task.name] = task_outputs

    # For TPU-friendlyness inference for all tasks is performed on each example.
    # However, the losses from all tasks but the example's task are masked out.
    # This doesn't significantly slow down runtime as long as the task-specific
    # modules are much faster to run than the BERT encoder.
    self.loss = tf.reduce_sum(
        tf.stack(losses, -1) *
        tf.one_hot(features["task_id"], config.n_tasks))


def model_fn_builder(config, tasks, task_weights, num_train_steps):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    utils.log("Building model")

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = MultitaskModel(
        config, tasks, task_weights, is_training, features, num_train_steps)

    # Load pre-trained weights from checkpoint
    tvars = tf.trainable_variables()
    scaffold_fn = None
    if not config.debug:
      assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
          tvars, config.init_checkpoint)
      if config.use_tpu:
        def tpu_scaffold():
          tf.train.init_from_checkpoint(config.init_checkpoint, assignment_map)
          return tf.train.Scaffold()
        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(config.init_checkpoint, assignment_map)

    # Run training or prediction
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          config, model.loss, num_train_steps)
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=model.loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          training_hooks=[
              training_utils.ETAHook(
                  config, {} if config.use_tpu else dict(loss=model.loss),
                  num_train_steps)
          ])
    else:
      assert mode == tf.estimator.ModeKeys.PREDICT
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=utils.flatten_dict(model.outputs),
          scaffold_fn=scaffold_fn)

    utils.log("Building complete")
    return output_spec

  return model_fn


class ModelRunner(object):
  """Class for training/evaluating models."""

  def __init__(self, config, tasks):
    self._config = config
    self._tasks = tasks
    self._preprocessor = preprocessing.Preprocessor(config, self._tasks)

    is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
    tpu_cluster_resolver = None
    if config.use_tpu and config.tpu_name:
      tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
          config.tpu_name, zone=config.tpu_zone, project=config.gcp_project)
    run_config = contrib_tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=config.checkpoints_dir,
        save_checkpoints_steps=config.save_checkpoints_steps,
        save_checkpoints_secs=None,
        tpu_config=contrib_tpu.TPUConfig(
            iterations_per_loop=config.iterations_per_loop,
            num_shards=config.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    (self._train_input_fn, self.train_steps,
     sizes) = self._preprocessor.prepare_train()
    task_weights = task_weighting.get_task_weights(config, sizes)

    model_fn = model_fn_builder(
        config=config,
        tasks=self._tasks,
        task_weights=task_weights,
        num_train_steps=self.train_steps)

    self._estimator = contrib_tpu.TPUEstimator(
        use_tpu=config.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        predict_batch_size=config.predict_batch_size)

  def train(self):
    utils.log("Training for {:} steps".format(self.train_steps))
    self._estimator.train(
        input_fn=self._train_input_fn, max_steps=self.train_steps)

  def evaluate(self):
    return {task.name: self._evaluate_task(task) for task in self._tasks}

  def _evaluate_task(self, task):
    """Evaluate the current model on the dev set."""
    utils.log("Evaluating", task.name)
    eval_input_fn, _, _ = self._preprocessor.prepare_eval(task)
    results = self._estimator.predict(input_fn=eval_input_fn,
                                      yield_single_examples=True)
    scorer = task.get_scorer()
    for r in results:
      if r["task_id"] != len(self._tasks):
        r = utils.nest_dict(r, self._config.task_names)
        scorer.update(r[task.name])
    utils.log(task.name + ": " + scorer.results_str())
    utils.log()
    return dict(scorer.get_results())

  def write_outputs(self, tasks, trial, split):
    """Write model prediction to disk."""
    utils.log("Writing out predictions for", tasks, split)
    distill_input_fn, _, _ = self._preprocessor.prepare_predict(tasks, split)
    results = self._estimator.predict(input_fn=distill_input_fn,
                                      yield_single_examples=True)
    # task name -> eid -> model-logits
    logits = collections.defaultdict(dict)
    for r in results:
      if r["task_id"] != len(self._tasks):
        r = utils.nest_dict(r, self._config.task_names)
        task_name = self._config.task_names[r["task_id"]]
        logits[task_name][r[task_name]["eid"]] = (
            r[task_name]["logits"] if "logits" in r[task_name]
            else r[task_name]["predictions"])
    for task_name in logits:
      utils.log("Pickling predictions for {:} {:} examples ({:})".format(
          len(logits[task_name]), task_name, split))
      if split == "train":
        if trial <= self._config.n_writes_distill:
          utils.write_pickle(
              logits[task_name],
              self._config.distill_outputs(task_name, trial))
      else:
        if trial <= self._config.n_writes_test:
          utils.write_pickle(logits[task_name],
                             self._config.test_outputs(
                                 task_name, split, trial))


def write_results(config, results):
  """Write out evaluate metrics to disk."""
  utils.log("Writing results to", config.results_txt)
  utils.mkdir(config.results_txt.rsplit("/", 1)[0])
  utils.write_pickle(results, config.results_pkl)
  with tf.gfile.GFile(config.results_txt, "w") as f:
    results_str = ""
    for trial_results in results:
      for task_name, task_results in trial_results.items():
        results_str += task_name + ": " + " - ".join(
            ["{:}: {:.2f}".format(k, v)
             for k, v in task_results.items()]) + "\n"
    f.write(results_str)


def main(_):
  topdir, model_name, hparams = sys.argv[-3:]  # pylint: disable=unbalanced-tuple-unpacking
  config = configure.Config(topdir, model_name, **json.loads(hparams))

  # Setup for training
  tasks = task_builder.get_tasks(config)
  results = []
  trial = 1
  utils.rmkdir(config.checkpoints_dir)
  heading_info = "model={:}, trial {:}/{:}".format(
      config.model_name, trial, config.num_trials)
  heading = lambda msg: utils.heading(msg + ": " + heading_info)

  # Train and evaluate num_trials models with different random seeds
  while trial <= config.num_trials:
    heading("Start training")
    model_runner = ModelRunner(config, tasks)
    model_runner.train()
    utils.log()

    heading("Run evaluation")
    results.append(model_runner.evaluate())
    write_results(config, results)

    if ((config.write_distill_outputs and trial <= config.n_writes_distill) or
        (config.write_test_outputs and trial <= config.n_writes_test)):
      heading("Write outputs")
      for task in tasks:
        if config.write_distill_outputs:
          model_runner.write_outputs([task], trial, "train")
        if config.write_test_outputs:
          for split in task.get_test_splits():
            model_runner.write_outputs([task], trial, split)

    utils.rmkdir(config.checkpoints_dir)
    trial += 1


if __name__ == "__main__":
  tf.app.run()
