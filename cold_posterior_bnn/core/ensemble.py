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

"""Methods and classes to work with Bayesian model ensembles.

This file contains methods to evaluate performance, make predictions, and work
with Bayesian model ensembles.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import io
import math
import os
import random
import tempfile
from absl import logging

import numpy as np
import tensorflow as tf    # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_datasets as tfds
from tqdm import tqdm

from cold_posterior_bnn.core import model as bnnmodel
from cold_posterior_bnn.core import statistics as stats


def _process_dataset(dataset, output_names):
  """Turns dataset into keras-compatible inputs, and collated targets.

  Args:
    dataset: tf.data.Dataset, tuple(inputs,outputs), inputs
    output_names: in case of implied order of targets.

  Returns:
    inputs, ordered_targets.
  """

  if isinstance(dataset, tuple):
    if len(dataset) == 1:
      ordered_targets = (None,)
      inputs = dataset
    else:
      ordered_targets = (dataset[1],)
      inputs = dataset[0]
  elif isinstance(dataset, tf.data.Dataset) or \
      isinstance(dataset, tf.compat.v2.data.Dataset):

    all_targets = [
        np.squeeze(b[1]) if isinstance(b, list) else b[1]
        for b in tfds.as_numpy(dataset)
    ]
    ordered_targets = _collate_tensor(all_targets, output_names)
    inputs = dataset
  else:
    ordered_targets = (None,)
    inputs = dataset
  return inputs, ordered_targets


class Ensemble(object):
  """Ensemble base class suitable for evaluation and prediction."""

  def reset(self):
    """Clears the ensemble .

    E.g. an empirical ensemble will clear its members.
    Returns:

    """
    raise NotImplementedError()

  def predict_ensemble(self, dataset):
    """Predict using all members of ensemble.

    Args:
      dataset: Either: - tf.data.Dataset yielding (inputs,outputs,
        [sample_weight]) tuples suitable for self.model.fit() API. Input/outputs
        can be singular, tuples (ordered according to self.model.inputs/outputs)
        or dict (with keys matching from self.model.input/output_names). - Tuple
        (inputs, outputs) suitable for self.model.fit(). - Inputs suitable for
        self.model.predict(inputs).

    Returns:
      ordered_outputs: (tuple of) model outputs of shape (len(ensemble),
      *model.output_shape)
    """
    ordered_outputs, _ = self.pred_and_eval(dataset, [])
    return ordered_outputs

  def evaluate_ensemble(self, dataset, statistics):
    """Evaluate statistics on ensemble.

    Depending on self.model having a single output, a list of outputs or named
    outputs, `statistics` and the return value are formatted as a list, a tuple
    of lists or a dictionary of lists, respectively.

    Args:
      dataset: Either: - tf.data.Dataset yielding (inputs,outputs,
        [sample_weight]) tuples suitable for self.model.fit() API. Input/outputs
        can be singular, tuples (ordered according to self.model.inputs/outputs)
        or dict (with keys matching from self.model.input/output_names). - Tuple
        (inputs, outputs) suitable for self.model.fit(). - Inputs suitable for
        self.model.predict(inputs).
      statistics: Output statistics following self.model.compile(metrics=...)
        API.
        - For single-output model (e.g. keras.Sequential()): [Statistic, ...].
        - Multiple-output model (e.g. keras.Model(outputs=(...,))):
          tuple([Statistic, ...], ...) in order of outputs.
        - Multiple named output model: (e.g. keras.Model(outputs={...})):
          dict(output_name=[Statistic, ...], ...)

    Returns:
      statistics_output: results in order of `statistics` following format:
        - For single-output model (e.g. keras.Sequential()): [result, ...].
        - Multiple-output model (e.g. keras.Model(outputs=(...,))):
          tuple([result, ...], ...) in order of outputs.
        - Multiple named output model: (e.g. keras.Model(outputs={...})):
          dict(output_name=[result, ...], ...).
    """
    _, formatted_results = self.pred_and_eval(
        dataset, statistics, retain_outputs=False)
    return formatted_results

  def pred_and_eval(self, dataset, statistics, retain_outputs=True):
    """Predict using all members of ensemble and evaluate statistics.

    Depending on self.model having a single output, a list of outputs or named
    outputs, `statistics` and the return value are formatted as a list, a tuple
    of lists or a dictionary of lists, respectively.

    Args:
      dataset: Either: - tf.data.Dataset yielding (inputs,outputs,
        [sample_weight]) tuples suitable for self.model.fit() API. Input/outputs
        can be singular, tuples (ordered according to self.model.inputs/outputs)
        or dict (with keys matching from self.model.input/output_names). - Tuple
        (inputs, outputs) suitable for self.model.fit(). - Inputs suitable for
        self.model.predict(inputs).
      statistics: Output statistics following self.model.compile(metrics=...)
        API.
        - For single-output model (e.g. keras.Sequential()): [Statistic, ...].
        - Multiple-output model (e.g. keras.Model(outputs=(...,))):
          tuple([Statistic, ...], ...) in order of outputs.
        - Multiple named output model: (e.g. keras.Model(outputs={...})):
          dict(output_name=[Statistic, ...], ...)
      retain_outputs: if False, do not store member outputs.

    Returns:
      statistics_output: results in order of `statistics` following format:
        - For single-output model (e.g. keras.Sequential()): [result, ...].
        - Multiple-output model (e.g. keras.Model(outputs=(...,))):
          tuple([result, ...], ...) in order of outputs.
        - Multiple named output model: (e.g. keras.Model(outputs={...})):
          dict(output_name=[result, ...], ...).
    """
    if len(self) < 1:
      raise ValueError("Ensemble cannot be empty during evaluation.")
    if hasattr(self.model, "output_names"):
      output_names = self.model.output_names
    else:
      output_names = None

    ordered_stats_lists = _statistics_to_ordered_tuple(statistics, output_names)
    for stats_list in ordered_stats_lists:
      for stat in stats_list:
        stat.reset()
    inputs, ordered_targets = _process_dataset(dataset, output_names)
    all_outputs = []
    for model in self.iter_members():
      # TODO(basv): take care of batching here to immediately retrieve targets?
      outputs = self._forward_pass(model, inputs)
      ordered_outputs = outputs if isinstance(outputs, list) else [outputs]
      if retain_outputs:
        all_outputs.append(ordered_outputs)
      for stats_list, output, target in zip(ordered_stats_lists,
                                            ordered_outputs, ordered_targets):
        if stats_list is not None:
          for stat in stats_list:
            stat.update(output, target)

    # pylint: disable=g-complex-comprehension
    def get_result(stat):
      res = stat.result()
      if hasattr(res, "numpy"):
        res = res.numpy()
      elif isinstance(res, tf.Tensor):
        res = tf.keras.backend.get_value(res)
      return res

    ordered_results_list = tuple([get_result(stat)
                                  for stat in stats_list]
                                 for stats_list in ordered_stats_lists)
    formatted_results = _format_like_example(ordered_results_list, statistics,
                                             output_names)

    ordered_all_outputs = _collate_tensor(all_outputs, None, func=np.stack)
    if len(ordered_all_outputs) == 1:
      ordered_all_outputs = ordered_all_outputs[0]

    return ordered_all_outputs, formatted_results

  @staticmethod
  def _forward_pass(model, inputs):
    """Forward pass through a single ensemble member.

    We define different forward_pass functions for different types of ensembles
    because the model.predict() used in EmpricialEnsemble does not support
    outputs of type tfp.distributions. See:
    https://github.com/tensorflow/probability/issues/427

    Therefore, in EmpricialEnsemble we use forward_pass with model.predict() and
    in VBEnsemble we use forward_pass with model().

    Args:
      model: `tf.keras.Model` for the forward pass.
      inputs: inputs in the format acceptable by either model() or
        model.predict() depending on the implementation of this function.

    Returns:
      `tf.keras.Model` outputs.
    """
    raise NotImplementedError()

  def iter_members(self):
    """Iterate over ensemble members.

    Yields:
      `tf.keras.Model` for each member to be used for prediction.
    """
    raise NotImplementedError()

  def __len__(self):
    """Return the total number of models in the ensemble."""
    raise NotImplementedError()

  def save_model(self, ensemble_dir):
    """Store `self.model` as a json file.

    See tf.keras.Model().to_json()

    Does not store weights, see `self.save_ensemble()`.

    Args:
      ensemble_dir: path to ensemble weights.
    """
    with tf.io.gfile.GFile(os.path.join(ensemble_dir, "model.json"), "w") as f:

      json_str = self.model.to_json()
      try:  # Python 2/3 compatible decoding to unicode.
        json_str = json_str.decode("utf-8")
      except AttributeError:
        pass
      f.write(json_str)

  def load_model(self, ensemble_dir):
    """Load `self.model` from json file.

    See tf.keras.models.model_from_json()

    Note: custom objects can be loaded by calling this function from inside a
    `with tf.keras.utils.CustomObjectScope({'ObjectName': ObjectName}:` scope.
    Does not load ensemble weights, see `self.save_ensemble()`.

    Args:
      ensemble_dir: path to ensemble weights.
    """
    # TODO(basv) consider tf.distribute.Strategy api and model building.
    with tf.io.gfile.GFile(os.path.join(ensemble_dir, "model.json"), "r") as f:
      json_string = str(f.read())

    with bnnmodel.bnn_scope():
      self.model = tf.keras.models.model_from_json(json_string)

  def save_ensemble(self, ensemble_dir):
    """Store model(s) and weights.

    Args:
      ensemble_dir: path to ensemble weights.
    """
    raise NotImplementedError()

  def load_ensemble(self, ensemble_dir):
    """Load model(s) and weights.

    Args:
      ensemble_dir: path to ensemble weights.
    """
    raise NotImplementedError()


def _format_like_example(args, example, key_order):
  """Returns args as instance, ordered list or dict, following `example` format.

  Note: facilitates single, multi-output or named multi-output Keras model API.

  Args:
    args: ordered tuple of arguments.
    example: example of format to follow: single instance, tuple or dict.
    key_order: keys for arguments in case `example` of type dict.

  Returns:
    args formatted as `example`.
  """
  if isinstance(example, dict):
    result = dict(zip(key_order, args))
  elif isinstance(example, (tuple, list)) and not len(example):  # pylint: disable=g-explicit-length-test
    # Empty single instance.
    result = []
  elif (isinstance(example, (tuple, list)) and
        isinstance(example[0], (tuple, list))):
    result = args
  else:
    result = args[0]
  return result


def _statistics_to_ordered_tuple(statistics, key_order):
  """Turn inputs into an ordered tuple.

  Note: facilitates single, multi-output or named multi-output Keras model API.

  Args:
    statistics: instance, tuple or dict of (named) statistics.
    key_order: in case of dict args, implied order.

  Returns:
    ordered tuple of statistics.
  """
  if statistics is None:
    ordered_stats = ()
  elif isinstance(statistics, dict):
    missing_keys = set(statistics.keys()) - set(key_order)
    if missing_keys:
      raise ValueError("Missing output(s) in model: %s" %
                       " ".join(missing_keys))
    ordered_stats = tuple(statistics.get(key, None) for key in key_order)
  elif isinstance(statistics, (tuple, list)) and len(statistics) and isinstance(
      statistics[0], (tuple, list)):
    ordered_stats = statistics
  else:
    ordered_stats = (statistics,)
  return ordered_stats


def _collate_tensor(all_tensors, key_order, func=np.concatenate):
  """Combines iterable of groups of tensors using `func` along first dimension.

  Args:
    all_tensors: iterable of dicts, tuples or single instances.
    key_order: keys of model output order.
    func: function to use for combination of elements.

  Returns:
    ordered_targets: Unzipped tensors with structure of `all_tensors` elements,
    combined along first dimension using combination function `func`.
  """

  def concat_or_none(key):
    if all_tensors[0].get(key, None) is None:
      return None
    return func([targets[key] for targets in all_tensors])

  if not all_tensors:
    return tuple()

  if isinstance(all_tensors[0], dict):
    ordered_targets = tuple(concat_or_none(key) for key in key_order)
  elif isinstance(all_tensors[0], (tuple, list)):
    ordered_targets = tuple(func(res) for res in zip(*all_tensors))
  else:
    ordered_targets = (func(all_tensors),)
  return ordered_targets


class EmpiricalEnsemble(Ensemble):
  """Ensemble with each member defined by a set of weights for the same model.

  This ensemble can be created in a number of ways, including non-Bayesian
  ensembles; for example, we can use a collection of models obtained from
  multiple training runs started from scratch.
  """

  def __init__(self,
               model,
               input_shape=None,
               weights_list=None,
               clone_model=True):
    """Initialize an empirical ensemble.

    Args:
      model: tf.keras.models.Model, tf.keras.models.Sequential, or a factory
        that instantiates an object that behaves as tf.keras.Model.
        The latter case suits e.g. a `lambda: YourModelClass(param=value,...)`.
      input_shape: tf.keras.models.Model input_shape to be used with Model.build
        function.  Note: currently cannot be None; TODO(nowozin): we hope this
          requirement will change in the future, see b/132994200
      weights_list: list of weights compatible with `model`
      clone_model: bool, default True. If using keras model.fit(), set to false.

    Raises:
      ValueError: unsupported argument.
    """
    if clone_model:
      if input_shape is None:
        raise ValueError("input_shape cannot be None in EmpiricalEnsemble "
                         "constructor")

      if isinstance(model, tf.keras.Model):
        self.model = bnnmodel.clone_model_and_weights(model, input_shape)
      else:
        self.model = model()
        self.model.build(input_shape)
    else:
      self.model = model
    self.input_shape = input_shape

    self.weights_list = weights_list if weights_list is not None else []

  @staticmethod
  def _forward_pass(model, inputs):
    """Forward pass through a single ensemble member.

    Args:
      model: `tf.keras.Model` for the forward pass.
      inputs: inputs in the format acceptable by model.predict().

    Returns:
      `tf.keras.Model` outputs.
    """
    return model.predict(inputs)

  def __len__(self):
    """Return the total number of models in the ensemble."""
    return len(self.weights_list)

  def append(self, weights):
    """Add a member to the ensemble.

    Args:
      weights: list of numpy arrays compatible with `self.model`.

    Returns:
      member_index: respective index of member in ensemble.
    """
    return self.append_maybe(lambda: weights)

  def append_maybe(self, get_weights_fn):
    """Add a member to the ensemble.

    Allows for lazy copy of weights.

    Args:
      get_weights_fn: function that returns set of weights.

    Returns:
      index: the index where the weights have been inserted.
    """
    self.weights_list.append(get_weights_fn())
    return len(self.weights_list) - 1

  def reset(self):
    """Resets the ensemble by deleting all members."""
    self.weights_list = []

  def save_ensemble(self, ensemble_dir):
    """Store model and weights.

    Args:
      ensemble_dir: path to ensemble weights.
    """
    self.save_model(ensemble_dir)
    self.save_weights(ensemble_dir)

  def load_ensemble(self, ensemble_dir):
    """Load model and weights.

    Args:
      ensemble_dir: path to ensemble weights.
    """
    self.load_model(ensemble_dir)
    self.load_weights(ensemble_dir)

  @staticmethod
  def save_model_weights(ensemble_dir, member_index, weights):
    """Save weight of single member ensemble.

    Args:
      ensemble_dir: path to ensemble weights.
      member_index: respective index of member in ensemble.
      weights: list of numpy arrays.
    """
    try:
      with tempfile.TemporaryFile() as tmp, tf.io.gfile.GFile(
          os.path.join(ensemble_dir, "weights_%d.npz" % int(member_index)),
          "wb") as f:
        # Note: CNS does not support np.savez, writing to tmp and copy instead.
        np.savez(tmp,
                 **{("weights_%050d" % i): w for i, w in enumerate(weights)})
        tmp.seek(0)
        f.write(tmp.read())
    except IOError:
      pass
    else:
      return
    try:
      with io.BytesIO() as tmp, tf.io.gfile.GFile(
          os.path.join(ensemble_dir, "weights_%d.npz" % int(member_index)),
          "wb") as f:
        # Note: CNS does not support np.savez, writing to tmp and copy instead.
        np.savez(tmp,
                 **{("weights_%050d" % i): w for i, w in enumerate(weights)})
        tmp.seek(0)
        f.write(tmp.read())
    except Exception as e:    # pylint: disable=broad-except
      logging.warn("Can not save model weights due to lack of space or due to:")
      logging.warn(str(e))

  def save_weights(self, ensemble_dir):
    """Save weights of all members.

    Args:
      ensemble_dir: path to ensemble weights.
    """
    for ensemble_id, weights in enumerate(self.weights_list):
      self.save_model_weights(ensemble_dir, ensemble_id, weights)

  def load_weights(self, ensemble_dir):
    """Load stored weights for ensemble.

    Does not use tf.keras.Model().load_weights, as this directly loads to a
    model, rather than the required fetching to memory.

    Args:
      ensemble_dir: path to ensemble weights.
    """
    self.weights_list = []
    for weights_file in \
        sorted(tf.io.gfile.glob(os.path.join(ensemble_dir, "weights_*.npz"))):

      with tf.io.gfile.GFile(weights_file, "rb") as f:
        weights = np.load(f)
        sorted_keys = sorted(weights.files)
        weights = [weights[k] for k in sorted_keys]
      self.weights_list.append(weights)

  def iter_members(self):
    """Iterate over ensemble members, handles weight swapping.

    Yields:
      `tf.keras.Model` for each member to be used for prediction.
    """
    backup_weights = self.model.get_weights()
    try:
      for weight in self.weights_list:
        self.model.set_weights(weight)
        yield self.model
    finally:
      self.model.set_weights(backup_weights)


class VBEnsemble(Ensemble):
  """Ensemble where each member is sampled from a tf.keras.models.Model.

  This ensemble assumes that the tf.keras.models.Model given in the constructor
  is a Variational Bayes (VB) model where an ensemble member is sampled
  each time a forward pass through the model is taken.
  """

  def __init__(self, model, n_members):
    """Initialize the ensemble with the VB model.

    Args:
      model: An object that behave like a VB tf.keras.models.Model.
      n_members: The number of ensemble members to sample from the model. Should
        be greater or equal to 1.
    """
    if n_members < 1:
      raise ValueError("Ensemble should consist of at least one model.")

    self.model = model
    self.n_members = n_members

  @staticmethod
  @tf.function
  def _forward_pass(model, inputs):
    """Forward pass through a single ensemble member.

    Note that we use the tf.function. Without this annotation this function
    becomes super slow (around 1000x times slower for a small problem).

    Args:
      model: `tf.keras.Model` for the forward pass.
      inputs: inputs in the format acceptable by model().

    Returns:
      `tf.keras.Model` outputs.
    """
    outputs = model(inputs)
    return outputs

  def iter_members(self):
    for _ in range(len(self)):
      # self.model is expected to handle stochasticity.
      yield self.model

  def __len__(self):
    return self.n_members


class FreshReservoirIterator(object):
  """Iterator class for iterating over fresh items in the reservoir."""

  def __init__(self, res, pos=0):
    self.res = res
    self.pos = pos

  def __iter__(self):
    return self

  def __next__(self):
    fresh_pos = self.res.fresh_position()
    while self.pos < len(self.res.reservoir):
      if self.res.reservoir[self.pos] is None:
        raise StopIteration

      if self.res.reservoir[self.pos][0] < fresh_pos:
        self.pos += 1
        continue

      self.pos += 1
      return self.res.reservoir[self.pos - 1][1]

    raise StopIteration

  next = __next__


class FreshReservoir(object):
  """Fresh reservoir sampling.

  [Reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) is an
  algorithm to obtain a set of items uniformly sampled from a stream of items.
  A _fresh_ reservoir is a reservoir which allows selection over a given
  fraction of the last items seen.

  As an example, in the context of MCMC sampling we would like to uniformly
  subsample iterates from the last 50% of iterates we have seen.
  """

  def __init__(self, capacity, freshness=50.0):
    """Create a new fresh reservoir.

    Args:
      capacity: int, total number of items stored.
      freshness: float or int, >0, <=100, the fraction of ordered iterates which
        are considered fresh.  For example, after 20000 items, with
        `freshness=80.0` we would consider the last 16000 items fresh,
        discarding the first 4000 items.
    """
    if capacity < 1:
      raise ValueError("Reservoir capacity must be at least one.")

    if freshness <= 0.0 or freshness > 100.0:
      raise ValueError("Reservoir freshness must be must in (0,100].")

    self.capacity = capacity
    self.freshness = freshness
    self.count = 0
    self.reservoir = [None] * capacity

  def fresh_position(self):
    """Return the absolute fresh time position.

    At the fresh position and after it models are considered fresh.

    Returns:
      fresh_pos: the absolute time position for which models are fresh.
    """
    return int(math.floor(((100.0 - self.freshness) / 100.0) * self.count))

  def append_maybe(self, get_item_fn):
    """Append an item to the reservoir and increase time counter.

    Implements lazy evaluation via an item getter function. This is preferred if
    get_item_fn requires an expensive copy of memory between processor units.

    Args:
      get_item_fn: function that when called returns the item to append.

    Returns:
      Insertion index in case item was inserted, None if item was not inserted.
    """
    self.count += 1
    if self.count <= self.capacity:
      self.reservoir[self.count - 1] = (self.count - 1, get_item_fn())
      return self.count - 1
    else:
      j = random.randint(1, self.count)
      if j <= self.capacity:
        self.reservoir[j - 1] = (self.count - 1, get_item_fn())
        return j - 1

  def __len__(self):
    """Return the number of all fresh items in the reservoir."""
    count = 0
    fresh_pos = self.fresh_position()

    for item in self.reservoir:
      if item is not None:
        if item[0] < fresh_pos:
          continue
        count += 1

    return count

  def __iter__(self):
    """Iterator over all fresh items in the reservoir."""
    return FreshReservoirIterator(self)


class FreshReservoirEnsemble(EmpiricalEnsemble):
  """Fresh reservoir ensemble which dynamically subsamples ensemble members.

  This ensemble can dynamically grow using the `append` method and can be called
  throughout MCMC sampling to keep models.
  """

  def __init__(self,
               model,
               input_shape,
               capacity=20,
               freshness=50.0,
               clone_model=True):
    """Initialize a fresh reservoir ensemble.

    Args:
      model: tf.keras.Model
      input_shape: tf.keras.models.Model input_shape to be used with Model.build
      capacity: total capacity of the reservoir.
      freshness: int or float, >0, <100.  The last `freshness` percent of models
        seen via the `append` method are treated as _fresh_ and take part in any
        evaluation.
      clone_model: bool, default=True, if using keras model.fit(), set to False.
    """
    reservoir = FreshReservoir(capacity, freshness)
    super(FreshReservoirEnsemble, self).__init__(model, input_shape, reservoir,
                                                 clone_model)

  def append_maybe(self, get_weights_fn):
    """Add a member to the ensemble.

    Allows for lazy copy of weights.

    Args:
      get_weights_fn: function that returns set of weights.

    Returns:
      Insertion index in case item was inserted, None if item was not inserted.
    """
    return self.weights_list.append_maybe(get_weights_fn)


class FreshReservoirSamplingEnsemble(FreshReservoirEnsemble):
  """Fresh reservoir ensemble which dynamically subsamples ensemble members.

  A custom sampling function must be provided by the user to iterate over the
  fresh reservoir.

  This ensemble can dynamically grow using the `append[_maybe]` methods and can
  be called throughout MCMC sampling to keep models.
  """

  def __init__(self, model, input_shape, sampler, capacity=20, freshness=50.0):
    """Initialize a fresh reservoir ensemble.

    Args:
      model: tf.keras.Model
      input_shape: tf.keras.models.Model input_shape to be used with Model.build
      sampler: function that accepts self as an argument and iterates over
        weights.
      capacity: total capacity of the reservoir.
      freshness: int or float, >0, <100.  The last `freshness` percent of models
        seen via the `append` method are treated as _fresh_ and take part in any
        evaluation.
    """
    self.sampler = sampler
    super(FreshReservoirSamplingEnsemble,
          self).__init__(model, input_shape, capacity, freshness)

  def iter_members(self):
    """Iterate over ensemble members, handles weight swapping.

    Yields:
      `tf.keras.Model` for each member to be used for prediction.
    """
    backup_weights = self.model.get_weights()
    try:
      for weight in self.sampler(self):
        self.model.set_weights(weight)
        yield self.model
    finally:
      self.model.set_weights(backup_weights)


class DeepEnsemble(EmpiricalEnsemble):
  """An empirical ensemble that enables training of Deep Ensembles [1, 2].

  Note: DeepEnsemble.fit() is experimental.

  A deep ensemble combines copies of the same deep neural network, each trained
  from scratch using different initializations. It is an effective baseline for
  model uncertainty.

  References:
    - [1] Lee, S. et al. (2015, November 19). Why M Heads are Better than One:
          Training a Diverse Ensemble of Deep Networks.
    - [2] Lakshminarayanan, B. et al. (2017). Simple and Scalable Predictive
          Uncertainty Estimation using Deep Ensembles.
  """

  def __init__(self, model, n_members, input_shape=None, weights_list=None):
    """Initialize a deep ensemble.

    Args:
      model: tf.keras.models.Model, tf.keras.models.Sequential, or a factory
        that instantiates an object that behaves as tf.keras.Model.
        The latter case suits e.g. a `lambda: YourModelClass(param=value,...)`.
      n_members: number of members to use.
      input_shape: tf.keras.models.Model input_shape to be used with Model.build
          function.  Note: currently cannot be None; TODO(nowozin): we hope this
            requirement will change in the future, see b/132994200
      weights_list: list of weights compatible with `model`, if provided, length
        must match n_members.

    Raises:
      ValueError: unsupported argument.
    """
    super(DeepEnsemble, self).__init__(model, input_shape, weights_list)
    self._optimizers_list = dict()
    self.n_members = n_members

  def iter_members(self):
    """Iterate over ensemble members, handles (optimizer) weights swapping.

    Yields:
      `tf.keras.Model` for each member to be used for prediction.
    """
    backup_weights = self.model.get_weights()
    try:
      for member_index, weight in enumerate(self.weights_list):
        self.model.set_weights(weight)
        if member_index in self._optimizers_list:
          self.model.optimizer.set_weights(self._optimizers_list[member_index])
        yield self.model
    finally:
      self.model.set_weights(backup_weights)

  def _ensemble_validation(self, ordered_stats_lists, validation_data,
                           statistics, history):
    """Function for validation of DeepEnsembles and BaggingDeepEnsembles.

    Args:
      ordered_stats_lists: List of stats for the ensemble.
      validation_data: Data to be used for validation.
      statistics: Statistics against which validation needs to occur.
      history: List of objects with the results of validation.

    Returns:
      history: The updated history object.
    """

    for stats_list in ordered_stats_lists:
      for stat in stats_list:
        stat.reset()
      results = self.evaluate_ensemble(validation_data, statistics)
      print(results)
      history.append(results)
    return history

  def _epoch_progress_logger(self, epoch, member_index, progress_bar,
                             fit_verbose):
    """Logging at the start of each epoch for each member.

    This function performs logging done at the start of each epoch for each
    member of the ensemble of a DeepEnsemble and a BaggingDeepEnsemble.

    Args:
      epoch: The epoch number that is currently being logged.
      member_index: The index of the member currently being trained.
      progress_bar: The tqdm bar displaying training progress.
      fit_verbose: Whether we want verbose training of the member in keras.
    """
    # Refresh progress bar description and update bar.
    progress_bar.set_description("Training member %4d at epoch %4d" %
                                 (member_index + 1, epoch + 1))
    progress_bar.update(1)

    logging.info("Training member %4d at epoch %4d", member_index + 1,
                 epoch + 1)

    if fit_verbose:
      progress_bar.write("\nMember training progress:")

  def _validate_fit_params(self, validation_split, validation_freq):
    """Allows for validation of params passed to DeepEnsemble and subclasses.

    Args:
      validation_split: The validation split percentage passed to fit().
      validation_freq: The validation frequency (in terms of the epochs) passed
        to the ensemble.

    Raises:
      ValueError: If the values for validation_split or validation_freq are
        invalid.
      RuntimeError: If the model for the ensemble is not provided with an
        optimizer.
    """
    # Check unsupported tf.keras.model.fit() arguments.
    if validation_split != 0.:
      raise ValueError("Validation_split not supported by ensemble.")

    # Check value of validation frequency.
    if validation_freq <= 0:
      raise ValueError("validation_freq must be greater than 0")

    # Check if model is compiled and well.
    if not self.model.optimizer:
      raise RuntimeError("Please call self.model.compile(...) to define "
                         "member training specifics.")

  def _get_ordered_stats_lists(self, statistics):
    """Returns an ordered list of stats objects for each output from the model.

    Args:
      statistics: A list of statistics objects to be used for each output.

    Returns:
      ordered_stats_lists: Lists of stats to be computed per output of the
        models in the ensemble.
    """
    if hasattr(self.model, "output_names"):
      output_names = self.model.output_names
    else:
      output_names = None
    ordered_stats_lists = _statistics_to_ordered_tuple(statistics, output_names)
    for stats_list in ordered_stats_lists:
      for stat in stats_list:
        if not isinstance(stat, stats.MeanStatistic):
          # TODO(basv): automaticaly wrap SampleStatistics with MeanStatistic
          raise ValueError("Invalid entry in statistics argument: only "
                           "MeanStatistics are supported.")
    return ordered_stats_lists

  def fit(self,
          dataset,
          y=None,
          statistics=None,
          epochs=1,
          validation_split=0.,
          validation_data=None,
          validation_freq=1,
          initial_epoch=0,
          **fit_kwargs):
    """Trains ensemble members using a different initialization for each model.

    Note: alpha functionality, experimental and prone to change.

    Arguments:
      dataset: Either: - tf.data.Dataset yielding (inputs,outputs,
        [sample_weight]) tuples suitable for self.model.fit() API. Input/outputs
        can be singular, tuples (ordered according to self.model.inputs/outputs)
        or dict (with keys matching from self.model.input/output_names). - Tuple
        (inputs, outputs) suitable for self.model.fit(). - Inputs suitable for
        self.model.predict(inputs).
      y: outputs in case dataset is input tensor.
      statistics: mean statistics to evaluate on output, see
          self.model.compile(metrics=...) API:
        - For single-output model (e.g. keras.Sequential()): [Statistic, ...].
        - Multiple-output model (e.g. keras.Model(outputs=(...,))):
          tuple([Statistic, ...], ...) in order of outputs.
        - Multiple named output model: (e.g. keras.Model(outputs={...})):
          dict(output_name=[Statistic, ...], ...)
      epochs: Number of epochs to train for.
      validation_split: Float between 0 and 1. Fraction of the training data to
        be used as validation data. The model will set apart this fraction of
        the training data, will not train on it, and will evaluate the loss and
        any model metrics on this data at the end of each epoch. The validation
        data is selected from the last samples in the `x` and `y` data provided,
        before shuffling. This argument is not supported when `x` is a dataset,
        generator or `keras.utils.Sequence` instance.
      validation_data: Data on which to evaluate the loss and any model metrics
        at the end of each epoch. The model will not be trained on this data.
        `validation_data` will override `validation_split`.
          `validation_data` could be: - tuple `(x_val, y_val)` of Numpy arrays
            or tensors - tuple `(x_val, y_val, val_sample_weights)` of Numpy
            arrays - dataset For the first two cases, `batch_size` must be
            provided. For the last case, `validation_steps` must be provided.
      validation_freq: Only relevant if validation data is provided. Integer or
        `collections.Container` instance (e.g. list, tuple, etc.). If an
        integer, specifies how many training epochs to run before a new
        validation run is performed, e.g. `validation_freq=2` runs validation
        every 2 epochs.
      initial_epoch: Integer. Epoch at which to start training (useful for
        resuming a previous training run).
      **fit_kwargs: Additional arguments.

    Returns:
      history:  A list of statistics results per epoch following format of
           `statistics`.

    Raises:
        RuntimeError: If the model was never compiled.
        ValueError: In case of mismatch between the provided input data
            and what the model expects.
    """
    # Check unsupported tf.keras.model.fit() arguments.
    fit_verbose = fit_kwargs.pop("verbose", None) or 0

    self._validate_fit_params(validation_split, validation_freq)
    # Collect output statistics.
    if hasattr(self.model, "output_names"):
      output_names = self.model.output_names
    else:
      output_names = None
    ordered_stats_lists = _statistics_to_ordered_tuple(statistics, output_names)
    for stats_list in ordered_stats_lists:
      for stat in stats_list:
        if not isinstance(stat, stats.MeanStatistic):
          # TODO(basv): automaticaly wrap SampleStatistics with MeanStatistic
          raise ValueError("Invalid entry in statistics argument: only "
                           "MeanStatistics are supported.")

    # Initialize new members:
    # TODO(basv): parametrize re-initialization or shared initialization.
    for member_index in range(len(self.weights_list), self.n_members):
      assert initial_epoch == 0, "New members can only be initialized at start."
      cloned = bnnmodel.clone_model_and_weights(self.model, self.input_shape)
      self.weights_list.append(cloned.get_weights())

    history = []

    # Add progres bar
    with tqdm(
        total=epochs * len(self.weights_list),
        position=0,
        leave=True,
        unit="epoch") as progress_bar:

      # Loop over epochs
      for epoch in range(initial_epoch, epochs):
        # For each member
        for member_index, model in enumerate(self.iter_members()):

          self._epoch_progress_logger(epoch, member_index, progress_bar,
                                      fit_verbose)
          # TODO(basv): check if member is already trained up to epoch.
          # TODO(basv): consider member callback state switching?

          # Fit for one epoch
          if y is not None:
            dataset_in, y_in = dataset, y
          elif isinstance(dataset, tuple) and len(dataset) == 2:
            dataset_in, y_in = dataset
          else:
            dataset_in, y_in = dataset, None

          model.fit(
              dataset_in,
              y_in,
              epochs=epoch + 1,
              initial_epoch=epoch,
              verbose=fit_verbose,
              **fit_kwargs)
          self.weights_list[member_index] = model.get_weights()
          self._optimizers_list[member_index] = model.optimizer.get_weights()

        if validation_data is not None and statistics is not None:
          # TODO(basv): statistics per dataset?
          # TODO(basv): unlabeled statistics?
          # Report ensemble statistics
          if epoch % validation_freq == 0:  # TODO(basv): verify validation_freq
            # Update statistics
            for stats_list in ordered_stats_lists:
              for stat in stats_list:
                stat.reset()
              results = self.evaluate_ensemble(validation_data, statistics)
              history.append(results)

        # [Handle Callbacks]
        # TODO(basv): implement some kind of ensemble callback structure.
        # TODO(basv): implement ensemble checkpointing.
        # TODO(basv): implement parallel workers and evaluate worker.

    return history


class BaggingDeepEnsemble(DeepEnsemble):
  """Class for deep ensembles with bagging."""

  def __init__(self, model, n_members, input_shape=None, weights_list=None):
    """Initialize a Bagging[1] Deep Ensemble that uses bootstrap resampled data.

    [1] Breiman, Leo. "Bagging predictors." Machine learning 24.2 (1996).

    Args:
      model: tf.keras.models.Model, tf.keras.models.Sequential, or a factory
        that instantiates an object that behaves as tf.keras.Model.
        The latter case suits e.g. a `lambda: YourModelClass(param=value,...)`.
      n_members: number of members to use.
      input_shape: tf.keras.models.Model input_shape to be used with Model.build
          function.  Note: currently cannot be None; TODO(nowozin): we hope this
            requirement will change in the future, see b/132994200
      weights_list: list of weights compatible with `model`, if provided, length
        must match n_members.

    Raises:
      ValueError: unsupported argument.
    """
    super(BaggingDeepEnsemble, self).__init__(model, n_members, input_shape,
                                              weights_list)
    self.sample_weights = []

  def fit(self,
          dataset,
          y=None,
          statistics=None,
          epochs=1,
          validation_split=0.,
          validation_data=None,
          validation_freq=1,
          initial_epoch=0,
          **fit_kwargs):
    """Trains ensemble members using a different initialization for each model.

    Note: alpha functionality, experimental and prone to change.

    Arguments:
      dataset: Either: - tf.data.Dataset yielding (inputs,outputs,
        [sample_weight]) tuples suitable for self.model.fit() API. Input/outputs
        can be singular, tuples (ordered according to self.model.inputs/outputs)
        or dict (with keys matching from self.model.input/output_names). - Tuple
        (inputs, outputs) suitable for self.model.fit(). - Inputs suitable for
        self.model.predict(inputs).
      y: outputs in case dataset is input tensor.
      statistics: mean statistics to evaluate on output, see
          self.model.compile(metrics=...) API:
        - For single-output model (e.g. keras.Sequential()): [Statistic, ...].
        - Multiple-output model (e.g. keras.Model(outputs=(...,))):
          tuple([Statistic, ...], ...) in order of outputs.
        - Multiple named output model: (e.g. keras.Model(outputs={...})):
          dict(output_name=[Statistic, ...], ...)
      epochs: Number of epochs to train for.
      validation_split: Float between 0 and 1. Fraction of the training data to
        be used as validation data. The model will set apart this fraction of
        the training data, will not train on it, and will evaluate the loss and
        any model metrics on this data at the end of each epoch. The validation
        data is selected from the last samples in the `x` and `y` data provided,
        before shuffling. This argument is not supported when `x` is a dataset,
        generator or `keras.utils.Sequence` instance.
      validation_data: Data on which to evaluate the loss and any model metrics
        at the end of each epoch. The model will not be trained on this data.
        `validation_data` will override `validation_split`.
          `validation_data` could be: - tuple `(x_val, y_val)` of Numpy arrays
            or tensors - tuple `(x_val, y_val, val_sample_weights)` of Numpy
            arrays - dataset For the first two cases, `batch_size` must be
            provided. For the last case, `validation_steps` must be provided.
      validation_freq: Only relevant if validation data is provided. Integer or
        `collections.Container` instance (e.g. list, tuple, etc.). If an
        integer, specifies how many training epochs to run before a new
        validation run is performed, e.g. `validation_freq=2` runs validation
        every 2 epochs.
      initial_epoch: Integer. Epoch at which to start training (useful for
        resuming a previous training run).
      **fit_kwargs: Additional kwargs.

    Returns:
      history:  A list of statistics results per epoch following format of
           `statistics`.

    Raises:
        RuntimeError: If the model was never compiled.
        ValueError: In case of mismatch between the provided input data
            and what the model expects.
    """
    fit_verbose = fit_kwargs.pop("verbose", None) or 0

    self._validate_fit_params(validation_split, validation_freq)

    # Collect output statistics.
    # TODO(basv): write statistics map function.
    ordered_stats_lists = self._get_ordered_stats_lists(statistics)

    if y is not None:
      dataset_in, y_in = dataset, y
    elif isinstance(dataset, tuple) and len(dataset) == 2:
      dataset_in, y_in = dataset
    else:
      raise NotImplementedError("Non-numpy dataset not supported for bagging.")
      # dataset_in, y_in = dataset, None
    # TODO(basv): assert dataset_in is a single tensor.

    # Initialize new members:
    # TODO(basv): parametrize re-initialization or shared initialization.
    for member_index in range(len(self.weights_list), self.n_members):
      assert initial_epoch == 0, "New members can only be initialized at start."
      cloned = bnnmodel.clone_model_and_weights(self.model, self.input_shape)
      self.weights_list.append(cloned.get_weights())

      self.sample_weights.append(np.random.poisson(1, size=len(dataset_in)))

    history = []

    # Add progress bar
    with tqdm(
        total=epochs * len(self.weights_list),
        position=0,
        leave=True,
        unit="epoch") as progress_bar:

      # Loop over epochs
      for epoch in range(initial_epoch, epochs):
        # For each member
        for member_index, model in enumerate(self.iter_members()):

          self._epoch_progress_logger(epoch, member_index, progress_bar,
                                      fit_verbose)

          # TODO(basv): check if member is already trained up to epoch.
          # TODO(basv): consider member callback state switching?

          # Fit for one epoch
          model.fit(
              dataset_in,
              y_in,
              sample_weight=self.sample_weights[member_index],
              epochs=epoch + 1,
              initial_epoch=epoch,
              verbose=fit_verbose,
              **fit_kwargs)
          self.weights_list[member_index] = model.get_weights()
          self._optimizers_list[member_index] = model.optimizer.get_weights()

        if (epoch + 1) % validation_freq == 0:
          if validation_data is not None and statistics is not None:
          # TODO(basv): statistics per dataset?
          # TODO(basv): unlabeled statistics?
          # Report ensemble statistics
            logging.info("Evaluating ensemble at epoch %4d", (epoch))
            history = self._ensemble_validation(ordered_stats_lists,
                                                validation_data, statistics,
                                                history)
      # [Handle Callbacks]
      # TODO(basv): implement some kind of ensemble callback structure.
      # TODO(basv): implement ensemble checkpointing.
      # TODO(basv): implement parallel workers and evaluate worker.

    return history
