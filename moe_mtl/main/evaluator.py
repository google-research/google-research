# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Classes and functions useful for evaluating models."""
import enum
import functools
import json
import os
# import pickle
import time
from typing import Any, Callable, Dict, Optional, Union, Mapping
from absl import logging
import cachetools
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
from flax.core import frozen_dict
import flax.struct
import gin
import jax
from jax.experimental import pjit
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from vmoe import utils



Array = jnp.ndarray
PartitionSpec = jax.sharding.PartitionSpec
PRNGKey = jnp.ndarray
PyTree = Any
EvalStepPjitFn = Callable[["EvalState", PyTree, Any], "EvalState"]


def tree_shape_dtype_struct(tree):
  """Converts a PyTree with array-like objects to jax.ShapeDtypeStruct."""

  def fn(x):
    shape, dtype = x.shape, x.dtype
    # Useful to convert Tensorflow Tensors.
    dtype = dtype.as_numpy_dtype if hasattr(dtype, "as_numpy_dtype") else dtype
    return jax.ShapeDtypeStruct(shape=shape, dtype=dtype)

  return jax.tree.map(fn, tree)


class ExecutionMode(enum.Enum):
  """Defines the model execution mode."""
  TRAIN = 1
  EVAL = 2
  PREDICT = 3


def log_model_size(params):
  """Logs the number of parameters.

  Args:
    params: A dictionary of string to parameter arrays.
  """
  parameter_overview.log_parameter_overview(params)
  params_size = jax.tree.map(lambda x: x.size, params)
  params_size = sum(jax.tree.flatten(params_size)[0])
  try:
    # Catch exceptions if running locally.
    xm_client = xmanager_api.XManagerApi(xm_deployment_env="alphabet")
    xm_client.get_current_experiment()
    work_unit = xm_client.get_current_work_unit()
    measurements = work_unit.get_measurement_series(label="params")
    measurements.create_measurement(objective_value=params_size, step=0)
  except RuntimeError:
    pass  # Not running on xmanager.


class EvalResults(flax.struct.PyTreeNode):
  eval_results: PyTree
  eval_labels: PyTree

  def update(self, eval_results, eval_labels):
    return self.replace(eval_results=eval_results, eval_labels=eval_labels)


class EvalState(flax.struct.PyTreeNode):
  """Evaluation state."""
  num: int
  labels: PyTree
  rngs: Dict[str, PRNGKey]
  outputs: PyTree

  def update(self, num, outputs, labels, rngs):
    num = self.num + num
    return self.replace(num=num, outputs=outputs, labels=labels, rngs=rngs)


class EvaluateMultipleDatasets(object):
  """Periodic action that evaluates a model on multiple datasets.

  Usage:
    eval_action = EvaluateMultipleDatasets(
      apply_fn=model.apply, datasets=eval_datasets, every_steps=10, ...)
    for step in range(100):
      params = train_step(params, ...)
      eval_action(step, params=params)  # Runs at steps 10, 20, 30, ...
  """

  def __init__(
      self,
      *,
      model_fn,
      metrics_fn,
      train_state_axis_resources,
      input_axis_resources,
      input_fn,
      metric_writer,
      rngs,
      workdir = None,
      use_ema = False,
      report_progress = None,
      report_progress_name = "eval"):
    """Evaluator."""
    if isinstance(metrics_fn, str):
      metrics_fn = json.loads(metrics_fn.replace("\"", "'"))
      for k, v in metrics_fn.items():
        metrics_fn[k] = gin.query_parameter(v.replace("%", ""))

    callback = self._make_callback_fn(
        model_fn=model_fn,
        metrics_fn=metrics_fn,
        train_state_axis_resources=train_state_axis_resources,
        input_axis_resources=input_axis_resources,
        metric_writer=metric_writer,
        input_fn=input_fn,
        rngs=rngs,
        report_progress=report_progress,
        use_ema=use_ema,
        report_progress_name=report_progress_name,
        workdir=workdir)
    self.callback = callback
    """super().__init__(

        every_steps=every_steps,
        every_secs=every_secs,
        on_steps=on_steps,
        callback_fn=callback,
        execute_async=False,
        pass_step_and_time=True)
    """

  def __call__(self, step, t, train_state):
    return self.callback(step, t, train_state)

  def _make_callback_fn(
      self, *, model_fn, metrics_fn,
      train_state_axis_resources, input_axis_resources,
      input_fn, metric_writer, rngs, report_progress,
      report_progress_name, workdir, use_ema):
    """Make a callback function for evaluation."""
    if isinstance(input_fn, dict):
      for key in input_fn:
        _, as_numpy_dataset = input_fn[key]()
        samples = (d for d in as_numpy_dataset)
    else:
      _, as_numpy_dataset = input_fn()
      samples = (d for d in as_numpy_dataset)

    batch = next(samples)

    eval_step_pjit = make_eval_step_pjit(
        train_state_axis_resources=train_state_axis_resources,
        input_axis_resources=input_axis_resources,
        model_fn=model_fn,
        metrics_fn=metrics_fn,
        use_ema=False,
        return_images=True)

    if use_ema:
      eval_step_pjit_ema = make_eval_step_pjit(
          train_state_axis_resources=train_state_axis_resources,
          input_axis_resources=input_axis_resources,
          model_fn=model_fn,
          metrics_fn=metrics_fn,
          use_ema=True,
          return_images=True)

    @cachetools.cached(
        cache={}, key=lambda name, *_: cachetools.keys.hashkey(name))
    def compile_for_dataset(name, train_state, train_step):
      # Note: This is not the initial EvalState, this only serves to compile the
      # eval step for a given dataset.
      logging.info(name, jax.tree.map(lambda x: x.shape, batch))

      t0 = time.time()

      args = tree_shape_dtype_struct((rngs, train_state, batch))

      eval_step_pjit_ds = eval_step_pjit.lower(*args).compile()  # pytype: disable=attribute-error
      t1 = time.time()
      metric_writer.write_scalars(train_step, {"eval/compile_secs": t1 - t0})
      if use_ema:
        eval_step_pjit_ema_ds = eval_step_pjit_ema.lower(*args).compile()  # pytype: disable=attribute-error
        t1 = time.time()
        metric_writer.write_scalars(train_step,
                                    {"eval_ema/compile_secs": t1 - t0})
        return eval_step_pjit_ds, eval_step_pjit_ema_ds
      else:
        return eval_step_pjit_ds

    def callback_fn(step, t, train_state):
      del t  # Unused.
      metrics_values = {}
      # NOTE: Fold-in the dataset name and/or the train_step to the seed
      # in order to use different initial seeds for each dataset and/or
      # evaluation run. Notice that each eval step will use a different seed,
      # since it"s updated in the EvalState (see evaluate_step).
      nonlocal input_fn
      logging.info(input_fn)
      if not isinstance(input_fn, dict):
        input_fn = {"eval": input_fn}

      for dataset_key in input_fn:
        if not use_ema:
          t1 = time.time()
          eval_step_pjit_ds = compile_for_dataset(dataset_key, train_state,
                                                  step)
          metric_values = evaluate_dataset(
              eval_step_pjit=eval_step_pjit_ds,
              dataset=input_fn[dataset_key],
              train_state=train_state,
              workdir=workdir,
              rngs=rngs)

          logging.info(metric_values)
          metrics_values[dataset_key] = {}
          metrics_values[dataset_key].update(metric_values)
        else:
          t1 = time.time()
          eval_step_pjit_ds, eval_step_pjit_ema_ds = compile_for_dataset(
              dataset_key, train_state, step)
          metric_values = evaluate_dataset(
              eval_step_pjit=eval_step_pjit_ds,
              dataset=input_fn[dataset_key],
              train_state=train_state,
              workdir=workdir,
              rngs=rngs)
          logging.info(metric_values)
          metrics_values[dataset_key] = {}
          metrics_values[dataset_key].update(metric_values)

          metric_values_ema = evaluate_dataset(
              eval_step_pjit=eval_step_pjit_ema_ds,
              dataset=input_fn[dataset_key],
              train_state=train_state,
              workdir=workdir,
              rngs=rngs)
          metrics_values[dataset_key + "_ema"] = {}
          metrics_values[dataset_key + "_ema"].update(metric_values_ema)
        for key in metric_values:
          try:
            for mkey in metric_values[key]:
              metric_writer.write_scalars(
                  step,
                  {f"{dataset_key}/{key}/{mkey}": metric_values[key][mkey]})
          except KeyError:
            metric_writer.write_scalars(
                step, {f"{dataset_key}/{key}": metric_values[key]})
        if use_ema:
          for key in metric_values_ema:
            try:
              for mkey in metric_values_ema[key]:
                metric_writer.write_scalars(step, {
                    f"{dataset_key}_ema/{key}/{mkey}": metric_values_ema[
                        key][mkey]
                })
            except KeyError:
              metric_writer.write_scalars(
                  step, {f"{dataset_key}_ema/{key}": metric_values_ema[key]})
        t2 = time.time()
        metric_writer.write_scalars(step, {f"{dataset_key}/eval_secs": t2 - t1})
      jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
      return metrics

    if report_progress is None:
      return callback_fn
    else:
      return report_progress.timed(
          report_progress_name, wait_jax_async_dispatch=False)(
              callback_fn)


def evaluate_dataset(
    *,
    eval_step_pjit,
    dataset,
    train_state,
    rngs,
    workdir,
):
  """Evaluates a given model on the given dataset."""
  log_model_size(train_state.optimizer.target["backbone_fn"])
  _, as_numpy_dataset = dataset()
  samples = (d for d in as_numpy_dataset)
  results = None
  for idx, batch in enumerate(samples):

    metrics_value, rngs, _, labels = eval_step_pjit(
        rngs, train_state, batch)
    if idx > 5:
      labels, _ = labels
    metrics_value = jax.tree.map(lambda x: x * 1.0, metrics_value)
    if results is None:
      results = metrics_value
    else:
      results = {
          name: results[name].merge(metrics_value[name])
          for name in metrics_value
      }

    logging.info(jax.tree.map(lambda x: x.shape, results))

  if not os.path.exists(workdir):
    tf.io.gfile.makedirs(workdir)
  results = jax.tree.map(lambda x: x.block_until_ready(), results)
  computed_results = {name: results[name].compute() for name in results}
  return computed_results


def evaluate_step(
    rngs,
    train_state,
    batch,
    model_fn,
    metrics_fn,
    mode=ExecutionMode.EVAL,
    use_ema = False,
    return_images = False):
  """Performs one evaluation step, updating the given state."""
  # valid = jnp.logical_not(fake)
  if isinstance(batch, tuple):
    features, labels = batch
  else:
    features = {"images": batch["images"], "labels": batch["labels"]}
  logging.info(features)
  rngs, next_rngs = utils.tree_rngs_split(rngs)
  variables = frozen_dict.unfreeze(train_state.model_state)
  if not use_ema:
    variables.update({"params": train_state.optimizer.target})
  else:
    variables.update({"params": train_state.ema_target})
  variables = frozen_dict.freeze(variables)
  model_outputs = model_fn(mode=mode).apply(
      variables=variables,
      **features,
      mutable=False,
      _do_remap=True,
      # blocked=True,
      rngs=rngs)
  metrics_value = model_outputs["metrics"]["metrics"]
  logging.info(metrics_value.keys())
  labels = batch["labels"]
  if "groundtruths" in labels and "masks" not in labels["groundtruths"]:
    labels["groundtruths"]["masks"] = np.zeros((batch["images"].shape[0],),
                                               dtype=jnp.float32)
  if "dispatch_weight" not in metrics_value:
    dispatch_weight = {}
  else:
    dispatch_weight = metrics_value["dispatch_weight"]
  logging.info(dispatch_weight)
  if "__valid__" in batch:
    valid_mask = batch["__valid__"]
    metrics_result = {
        key: value.from_model_output(
            model_outputs, _do_remap=True, labels=labels,
            mask=valid_mask) for key, value in metrics_fn.items()
    }
  else:
    metrics_result = {
        key: value.from_model_output(
            model_outputs, _do_remap=True, labels=labels)
        for key, value in metrics_fn.items()
    }
  if return_images:
    return metrics_result, next_rngs, dispatch_weight, [labels, batch["images"]]
  else:
    return metrics_result, next_rngs, dispatch_weight, labels


def make_eval_step_pjit(
    train_state_axis_resources,
    input_axis_resources,
    model_fn,
    metrics_fn,
    use_ema = False,
    return_images = False
):
  """Create a pjitted function that performs one evaluation step."""

  eval_step_pjit = pjit.pjit(
      fun=functools.partial(
          evaluate_step,
          model_fn=model_fn,
          metrics_fn=metrics_fn,
          mode=ExecutionMode.EVAL,
          use_ema=use_ema,
          return_images=return_images,
      ),
      in_shardings=(
          None,  # rng
          train_state_axis_resources,  # train_state_axis_resources
          input_axis_resources,  # batch
      ),
      out_shardings=(
          None,
          None,
          None,
          None,
      ),
      donate_argnums=(0, 2),
  )
  return eval_step_pjit


class EvaluateMultipleDatasetsMTL:
  """Evaluator for MTL."""

  def __init__(
      self,
      *,
      model_fn,
      metrics_fn,
      train_state_axis_resources,
      input_axis_resources_det,
      input_axis_resources_cls,
      input_fn_det,
      input_fn_cls,
      metric_writer,
      rngs,
      use_ema = True,
      report_progress = None,
      report_progress_name = "eval"):
    """Initializer."""
    if isinstance(metrics_fn, str):
      metrics_fn = json.loads(metrics_fn)
      for k, v in metrics_fn.items():
        metrics_fn[k] = gin.query_parameter(v.replace("%", ""))
      logging.info(metrics_fn)

    callback = self._make_callback_fn(
        model_fn=model_fn,
        metrics_fn=metrics_fn,
        train_state_axis_resources=train_state_axis_resources,
        input_axis_resources_det=input_axis_resources_det,
        input_axis_resources_cls=input_axis_resources_cls,
        metric_writer=metric_writer,
        input_fn_det=input_fn_det,
        input_fn_cls=input_fn_cls,
        rngs=rngs,
        use_ema=use_ema,
        report_progress=report_progress,
        report_progress_name=report_progress_name)
    self.callback = callback

  def __call__(self, step, t, train_state):
    return self.callback(step, t, train_state)

  def _make_callback_fn(
      self, *, model_fn, metrics_fn,
      train_state_axis_resources, input_axis_resources_det,
      input_axis_resources_cls, input_fn_det, input_fn_cls,
      metric_writer, rngs, report_progress, use_ema, report_progress_name):
    """Callback."""
    logging.info(metrics_fn)

    # Note: We create the eval_step_pjit here to avoid multiple compilation
    # steps. If the shapes of inputs/outputs for all datasets is the same, this
    # will be only compiled once.
    _, as_numpy_dataset_cls = input_fn_cls()
    _, as_numpy_dataset_det = input_fn_det()
    samples_cls = (d for d in as_numpy_dataset_cls)
    samples_det = (d for d in as_numpy_dataset_det)
    # data_generator = jax_utils.prefetch_to_device(data_generator, 1)
    b_cls = next(samples_cls)
    b_det = next(samples_det)

    eval_step_pjit = make_eval_step_pjit_mtl(
        train_state_axis_resources=train_state_axis_resources,
        input_axis_resources_cls=input_axis_resources_cls,
        input_axis_resources_det=input_axis_resources_det,
        model_fn=model_fn,
        metrics_fn=metrics_fn)
    eval_step_pjit_ema = make_eval_step_pjit_mtl(
        train_state_axis_resources=train_state_axis_resources,
        input_axis_resources_cls=input_axis_resources_cls,
        input_axis_resources_det=input_axis_resources_det,
        model_fn=model_fn,
        use_ema=True,
        metrics_fn=metrics_fn)
    nb_det = {}
    nb_cls = {}
    for key in b_det:
      nb_det[key + "_det"] = b_det[key]
    for key in b_cls:
      nb_cls[key + "_cls"] = b_cls[key]

    # log_model_flops(model_fn, b_det, b_cls)
    def compile_for_dataset(train_state, train_step):
      # Note: This is not the initial EvalState, this only serves to compile the
      # eval step for a given dataset.
      t0 = time.time()

      args = tree_shape_dtype_struct((rngs, train_state, nb_det, nb_cls))
      if not use_ema:
        eval_step_pjit_ds = eval_step_pjit.lower(*args).compile()
        t1 = time.time()
        metric_writer.write_scalars(train_step, {"eval/compile_secs": t1 - t0})
        return eval_step_pjit_ds
      else:
        eval_step_pjit_ema_ds = eval_step_pjit_ema.lower(*args).compile()
        return None, eval_step_pjit_ema_ds

    def callback_fn(step, t, train_state):
      del t  # Unused.
      t1 = time.time()
      if use_ema:
        eval_step_pjit_ds, eval_step_pjit_ema_ds = compile_for_dataset(
            train_state, step)
        metric_values = evaluate_dataset_mtl(
            eval_step_pjit=eval_step_pjit_ema_ds,
            dataset_cls=input_fn_cls,
            dataset_det=input_fn_det,
            train_state=train_state,
            b_det=nb_det,
            b_cls=nb_cls,
            rngs=rngs)

        for key in metric_values:
          try:
            for mkey in metric_values[key]:
              metric_writer.write_scalars(
                  step, {f"eval_ema/{key}/{mkey}": metric_values[key][mkey]})
          except KeyError:
            metric_writer.write_scalars(step,
                                        {f"eval_ema/{key}": metric_values[key]})
        t2 = time.time()
        metric_writer.write_scalars(step, {"eval_ema/eval_secs": t2 - t1})
      else:
        eval_step_pjit_ds = compile_for_dataset(train_state, step)
        metric_values = evaluate_dataset_mtl(
            eval_step_pjit=eval_step_pjit_ds,
            dataset_cls=input_fn_cls,
            dataset_det=input_fn_det,
            train_state=train_state,
            b_det=nb_det,
            b_cls=nb_cls,
            rngs=rngs)

        for key in metric_values:
          try:
            for mkey in metric_values[key]:
              metric_writer.write_scalars(
                  step, {f"eval/{key}/{mkey}": metric_values[key][mkey]})
          except KeyError:
            metric_writer.write_scalars(step,
                                        {f"eval/{key}": metric_values[key]})
        t2 = time.time()
        metric_writer.write_scalars(step, {"eval/eval_secs": t2 - t1})

    if report_progress is None:
      return callback_fn
    else:
      return report_progress.timed(
          report_progress_name, wait_jax_async_dispatch=False)(callback_fn)


def evaluate_dataset_mtl(
    *, eval_step_pjit, dataset_det, dataset_cls,
    train_state, b_cls, b_det, rngs):
  """Evaluates a given model on the given dataset."""
  log_model_size(train_state.optimizer.target["backbone_fn"])
  _, as_numpy_dataset_det = dataset_det()
  _, as_numpy_dataset_cls = dataset_cls()
  samples_det = (d for d in as_numpy_dataset_det)
  samples_cls = (d for d in as_numpy_dataset_cls)
  results = {"det": None, "cls": None}
  for idx, batch in enumerate(samples_det):
    logging.info(idx)
    nbatch = {}
    for key in batch:
      nbatch[key + "_det"] = batch[key]
    metrics_value, rngs, _, _, _, _ = eval_step_pjit(
        rngs, train_state, nbatch, b_cls)
    metrics_value = jax.tree.map(lambda x: x * 1.0, metrics_value)
    if results["det"] is None:
      results["det"] = metrics_value["det"]
    else:
      results["det"] = results["det"].merge(metrics_value["det"])

  for idx, batch in enumerate(samples_cls):
    logging.info(idx)
    nbatch = {}
    for key in batch:
      nbatch[key + "_cls"] = batch[key]
    metrics_value, rngs, _, _, _, _ = eval_step_pjit(
        rngs, train_state, b_det, nbatch)
    metrics_value = jax.tree.map(lambda x: x * 1.0, metrics_value)
    if results["cls"] is None:
      results["cls"] = metrics_value["cls"]
    else:
      results["cls"] = results["cls"].merge(metrics_value["cls"])

  computed_results = {name: results[name].compute() for name in results}  #  pytype: disable=attribute-error
  # logging.info(computed_results)
  return computed_results


def evaluate_step_mtl(
    rngs,
    train_state,
    batch_det,
    batch_cls,
    model_fn,
    metrics_fn,
    mode=ExecutionMode.EVAL,
    use_ema = False):
  """Performs one evaluation step, updating the given state."""
  # valid = jnp.logical_not(fake)

  if isinstance(batch_det, tuple):
    features_det, labels_det = batch_det
  else:
    features_det = batch_det
    labels_det = batch_det["labels_det"]

  if isinstance(batch_cls, tuple):
    features_cls, labels_cls = batch_cls
  else:
    features_cls = batch_cls
    labels_cls = batch_cls["labels_cls"]
  logging.info(features_cls.keys())
  valid = features_cls["__valid___cls"]

  del features_cls["__valid___cls"]

  rngs, next_rngs = utils.tree_rngs_split(rngs)
  variables = frozen_dict.unfreeze(train_state.model_state)
  if not use_ema:
    variables.update({"params": train_state.optimizer.target})
  else:
    variables.update({"params": train_state.ema_target})
  variables = frozen_dict.freeze(variables)
  model_outputs = model_fn(mode=mode).apply(
      variables=variables,
      **features_det,
      **features_cls,
      second_stage=True,
      mutable=False,
      _do_remap=True,
      rngs=rngs)

  if "groundtruths" in labels_det and "masks" not in labels_det["groundtruths"]:
    labels_det["groundtruths"]["masks"] = np.zeros(
        (features_det["images_det"].shape[0],), dtype=jnp.float32)
  metrics_value = dict()
  metrics_value["cls"] = metrics_fn["cls"].from_model_output(
      model_outputs,
      _do_remap=True,
      labels=labels_cls,
      mask=valid)
  metrics_value["det"] = metrics_fn["det"].from_model_output(
      model_outputs, _do_remap=True, labels=labels_det)
  metrics_dispatch_weight = model_outputs["metrics"]["metrics"]
  logging.info(metrics_dispatch_weight.keys())
  return metrics_value, next_rngs, metrics_dispatch_weight[
      "logits_det"], metrics_dispatch_weight["logits_cls"], [
          labels_det, features_det["images_det"].astype(jnp.float32)], [
              labels_cls, features_cls["images_cls"].astype(jnp.float32)]


def make_eval_step_pjit_mtl(
    train_state_axis_resources,
    input_axis_resources_det,
    input_axis_resources_cls,
    model_fn,
    metrics_fn,
    use_ema = False):
  """Create a pjitted function that performs one evaluation step."""

  eval_step_pjit = pjit.pjit(
      fun=functools.partial(
          evaluate_step_mtl,
          use_ema=use_ema,
          model_fn=model_fn,
          metrics_fn=metrics_fn,
      ),
      in_shardings=(
          None,  # rng
          train_state_axis_resources,  # train_state_axis_resources
          input_axis_resources_det,  # batch_det
          input_axis_resources_cls,  # batch_cls
      ),
      out_shardings=(None, None, None, None, None, None),
      donate_argnums=(0, 2, 3),
  )
  return eval_step_pjit
