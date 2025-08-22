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

"""Class for enumerative sequential synthesizer."""

from __future__ import annotations

import copy
import itertools
import random
import sys
import traceback
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Sequence, Set, Tuple, TypeVar

from absl import logging

from abstract_nas.abstract.base import AbstractProperty
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import Op
from abstract_nas.model.concrete import OpType
from abstract_nas.model.subgraph import SubgraphModel
from abstract_nas.model.utils import split_div_mul
from abstract_nas.synthesis.sequential import AbstractSequentialSynthesizer

T = TypeVar("T")


def sequence_generator(gen,
                       max_len,
                       min_len = 0):
  """Takes a generator of elements of type T and returns a generator that enumerates over sequences of type T.

  e.g., if gen yields [0,1,2]
  then sequence_generator will yield:
    [[0], [1], [2],
     [0,0], [0,1], [0,2], [1,0], [1,1], [1,2], ...,
     [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], ...,
     ...]
  up to all sequences of length max_len.

  If max_len < 0, does not enforce a max length.

  Args:
    gen: a generator for individual elements of type T
    max_len: the maximum length to generate
    min_len: the minimum length to generate

  Yields:
    Sequences of individual elements of type T
  """
  seq = []
  iterators = [gen() for _ in range(max(min_len, 1))]
  while True:
    while len(seq) < len(iterators):
      try:
        el = next(iterators[len(seq)])
        seq.append(el)
      except StopIteration:
        iterators[len(seq)] = gen()
        if seq:
          seq.pop()
        else:
          if max_len >= 0 and len(iterators) >= max_len:
            return
          iterators.append(gen())

    yield copy.deepcopy(list(seq))
    seq.pop()


class EnumerativeSequentialSynthesizer(AbstractSequentialSynthesizer):
  """Synthesizer that enumerates through sequential subgraphs in length order.

  Synthesis is split into two parts. First, we enumerate over ops using the
  TEST* kwargs. For example, convolution of different strides does not affect
  the abstract semantics of the op, so we only need to test using stride=1.
  After synthesizing a satisfying subgraph using only TEST* kwargs, we then
  iterate through the subgraph in order, replacing each op in the synthesized
  subgraph with a new, random op of the same type such that the subgraph with
  the replaced op still satisfies the required properties. For this process,
  we use the FULL* kwargs.
  """

  KERNEL_INITS = ["I:xavier_uniform", "I:kaiming_uniform", "I:zeros"]
  BIAS_INITS = ["I:normal:stddev:1e-6", "I:zeros"]

  TEST_FEATURE_FACTORS = ["", "*2", "*4", "%2", "%4"]
  TEST_OUT_FEATURES = ["S:-1"]
  TEST_CONV_KERNEL_SIZES = [1, 3]
  TEST_STRIDES = [1]
  TEST_FEATURE_GROUPS = [1]
  TEST_POOL_WINDOW_SIZES = [0, 2, 3]

  FULL_FEATURE_FACTORS = ["%4", "%2", "", "*2", "*4"]
  FULL_OUT_FEATURES = ["S:-1"]
  FULL_CONV_KERNEL_SIZES = [1, 3, 5]
  FULL_STRIDES = [1, 2, 3]
  FULL_FEATURE_GROUPS = [1, 2, 4, 16, 64]
  FULL_POOL_WINDOW_SIZES = [0, 2, 3, 4, 8]

  MERGE_KWARGS = [["strides", "window_shape"]]

  def __init__(self,
               subgraphs_and_props,
               generation,
               abstract = True,
               max_len = -1,
               max_delta = -1,
               min_len = 0,
               min_delta = -1):
    super().__init__(subgraphs_and_props, generation, abstract)
    if max_len >= 0 and max_delta >= 0:
      raise ValueError("Provided both max_len and max_delta.")
    self.max_len = max_len
    self.max_delta = max_delta
    self.min_len = min_len
    self.min_delta = min_delta
    self.kwarg_defaults = self.get_subgraph_kwarg_defaults()

  def get_subgraph_kwarg_defaults(self):
    kwarg_defaults = {}

    # get all the kwargs from ops in the subgraph model
    for node in self.subgraphs_and_props[0][0].subgraph:
      for kwargs in [node.op.op_kwargs, node.op.input_kwargs]:
        for k, v in kwargs.items():
          # If there are multiple subgraphs, we cannot use any constants for the
          # feature dimension.
          if k == "features" and len(self.subgraphs_and_props) > 1:
            try:
              int(v)
              continue
            except ValueError:
              pass
          if k not in kwarg_defaults:
            kwarg_defaults[k] = set()
          kwarg_defaults[k].add(v)

    # Merge kwargs.
    for merge_list in self.MERGE_KWARGS:
      merged_kwargs = set()
      for k in merge_list:
        kwargs = kwarg_defaults.get(k, set())
        merged_kwargs.update(kwargs)
      for k in merge_list:
        kwarg_defaults[k] = merged_kwargs

    return kwarg_defaults

  @classmethod
  def get_test_kwarg_defaults(cls):
    kv = {
        "features": cls.TEST_OUT_FEATURES,
        "kernel_size": cls.TEST_CONV_KERNEL_SIZES,
        "strides": cls.TEST_STRIDES,
        "num_groups": cls.TEST_FEATURE_GROUPS,
        "feature_group_count": cls.TEST_FEATURE_GROUPS,
        "window_shape": cls.TEST_POOL_WINDOW_SIZES,
        "kernel_dilation": cls.TEST_FEATURE_GROUPS,
    }
    return {k: set(v) for k, v in kv.items()}

  @classmethod
  def get_full_kwarg_defaults(cls):
    kv = {
        "features": cls.FULL_OUT_FEATURES,
        "kernel_size": cls.FULL_CONV_KERNEL_SIZES,
        "strides": cls.FULL_STRIDES,
        "num_groups": cls.FULL_FEATURE_GROUPS,
        "feature_group_count": cls.FULL_FEATURE_GROUPS,
        "window_shape": cls.FULL_POOL_WINDOW_SIZES,
        "kernel_dilation": cls.FULL_STRIDES,
    }
    return {k: set(v) for k, v in kv.items()}

  def synthesize(self):
    """Synthesizes a subgraph satisfying all the properties."""
    prefix = f"gen{self.generation}/"
    if self.max_delta > 0:
      max_len = self.num_ops + self.max_delta
    else:
      max_len = self.max_len
    if self.min_delta > 0:
      min_len = max(0, self.num_ops - self.min_delta)
    else:
      min_len = self.min_len
    subg_enum = self.subg_enumerator(prefix, max_len, min_len,
                                     self.kwarg_defaults, full=False)
    for subg in subg_enum:
      adjust_features = len(subg) == self.max_len
      subgraph_spec = self.make_subgraph_spec(subg, adjust_features)
      try:
        subgraph_models = self.make_subgraph_models(subgraph_spec)
      except ValueError:
        continue
      try:
        if self.verify(subgraph_models):
          # Because of adjust_features in make_subgraph_spec, the subgraph ops
          # may have changed, so make sure to randomize from the most up-to-date
          # subgraph.
          subg = [node.op for node in subgraph_spec]
          subgraph_models = self.randomize_subgraph(subg, self.kwarg_defaults)
          return subgraph_models
      except Exception:  # pylint: disable=broad-except
        # Catch everything else for now... This is the safest way to filter out
        # malformed subgraphs which will not execute.
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logging.info(
            "%s", "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)))
    raise StopIteration

  def randomize_subgraph(self,
                         subg,
                         kwarg_defaults = None,
                         full = True):
    orig_subgraph_spec = self.make_subgraph_spec(subg, adjust_features=False)
    orig_subgraph_models = self.make_subgraph_models(orig_subgraph_spec)
    if not self.verify(orig_subgraph_models):
      raise ValueError("Original subg must satisfy properties.")

    match_kwargs = [
        # kwarg key, preferred values, probability of staying on preferred
        ("features", [], 1.0),
        ("kernel_size", [1], 0.5),
        ("feature_group_count", [1], 0.5),
        ("kernel_dilation", [1], 0.5),
        ("window_shape", [0], 1.0),
    ]

    def match_kwarg_defaults(kwarg_defaults, op):
      for k, vs, p in match_kwargs:
        if k not in kwarg_defaults:
          continue
        for kwargs in [op.input_kwargs, op.op_kwargs]:
          if k in kwargs:
            v = kwargs[k]
            if (not vs or v in vs) and random.random() < p:
              kwarg_defaults[k] = set([v])
      return kwarg_defaults

    new_subg = list(subg)
    kwarg_defaults = self.make_default_kwargs(kwarg_defaults, full)

    # Replace each op in sequence.
    for idx, op in enumerate(subg):
      if op.type not in [
          OpType.DENSE,
          OpType.CONV,
          OpType.GROUP_NORM,
          OpType.AVG_POOL,
          OpType.MAX_POOL,
          OpType.MEAN,
      ]:
        continue

      op_kwargs_dict, input_kwargs_dict = self.all_kwargs_for_op_type(
          kwarg_defaults, full, op.type)

      replacement_op = copy.deepcopy(op)
      found_replacement = False
      for kwargs_type, kwargs_dict in [("op", op_kwargs_dict),
                                       ("input", input_kwargs_dict)]:
        match_kwarg_defaults(kwargs_dict, op)
        keys = list(kwargs_dict.keys())
        random.shuffle(keys)
        for k in keys:
          vs = list(kwargs_dict[k])
          if not vs: continue
          current_op = copy.deepcopy(replacement_op)
          random.shuffle(vs)
          for v in vs:
            if kwargs_type == "op":
              current_op.op_kwargs[k] = v
            else:
              assert kwargs_type == "input"
              current_op.input_kwargs[k] = v
            new_subg[idx] = current_op
            try:
              subgraph_spec = self.make_subgraph_spec(new_subg)
              subgraph_models = self.make_subgraph_models(subgraph_spec)
              if self.verify(subgraph_models):
                found_replacement = True
                replacement_op = current_op
                break
            except Exception:  # pylint: disable=broad-except
              pass

      if not found_replacement:
        logging.warn("Was unable to find replacement for op of type %s, "
                     "op_kwargs=%s, input_kwargs=%s",
                     op.type.name.lower(),
                     op.op_kwargs, op.input_kwargs)
        new_subg[idx] = op

    subgraph_spec = self.make_subgraph_spec(new_subg)
    subgraph_models = self.make_subgraph_models(subgraph_spec)
    assert self.verify(subgraph_models)
    return subgraph_models

  @classmethod
  def subg_enumerator(cls,
                      prefix = None,
                      max_len = -1,
                      min_len = 0,
                      kwarg_defaults = None,
                      full = True):
    return sequence_generator(
        lambda: cls.op_enumerator(prefix, kwarg_defaults, full), max_len,
        min_len)

  @classmethod
  def make_default_kwargs(cls,
                          kwarg_defaults = None,
                          full = True):
    if not kwarg_defaults:
      kwarg_defaults = {}
    else:
      kwarg_defaults = dict(kwarg_defaults)

    if full:
      cls_kwarg_defaults = cls.get_full_kwarg_defaults()
    else:
      cls_kwarg_defaults = cls.get_test_kwarg_defaults()
    for k, v in cls_kwarg_defaults.items():
      if k not in kwarg_defaults:
        kwarg_defaults[k] = set(v)
      else:
        kwarg_defaults[k].update(v)
    return kwarg_defaults

  @classmethod
  def all_kwargs_for_op_type(
      cls,
      kwarg_defaults,
      full,
      op_type,
  ):
    """Returns possible values for op_kwargs, input_kwargs."""
    # Not supported.
    if op_type in [
        # TODO(charlesjin): how to enumerate batch_dims, etc.?
        OpType.DENSE_GENERAL,
        # Does not support binary ops.
        OpType.ADD, OpType.MUL,
        # TODO(charlesjin): how to enumerate over scalar values?
        OpType.SCALAR_ADD,
        # Does not support binary ops.
        OpType.DOT_GENERAL,
        # Does not support binary ops.
        OpType.EINSUM,
        OpType.FLATTEN,
        # TODO(charlesjin): how to enumerate over dims?
        OpType.RESHAPE,
        # TODO(charlesjin): how to enumerate over dims?
        OpType.TRANSPOSE,
        OpType.PARAM,
        OpType.SELF_ATTENTION,
        OpType.STOCH_DEPTH,
    ]:
      return {}, {}

    # No kwargs.
    if op_type in [
        # TODO(charlesjin): how to enumerate over scalar values?
        OpType.SCALAR_MUL,
        OpType.BATCH_NORM,
        OpType.LAYER_NORM,
        OpType.RELU,
        OpType.GELU,
        OpType.SWISH,
        OpType.SIGMOID,
        OpType.SOFTMAX,
    ]:
      return {}, {}

    if full:
      feature_factors = cls.FULL_FEATURE_FACTORS
    else:
      feature_factors = cls.TEST_FEATURE_FACTORS

    def make_features():
      """Returns a set of features to use for synthesis.

      We need the synthesized subgraph to be generalizable across multiple
      contexts (as specified by subgraphs_and_props), and in general, each
      context may require a different output shape. Hence we should only take
      features which are not constants (i.e., we should set the output features
      relative to the input features).

      The reason kwarg_defaults may contain constants is because we augment it
      with values from the subgraphs to be mutated in
      get_subgraph_kwarg_defaults during initialization.
      """
      features_list = []
      base_features_set = kwarg_defaults.get("features", set())
      for factor in feature_factors:
        if not factor:
          features_list.extend(base_features_set)
          continue
        op, val = factor[0], int(factor[1:])
        assert op in ("*", "%")
        for features in base_features_set:
          # If features is a concrete value, then compute the new concrete
          # value. Otherwise, features is a symbolic value, so we defer
          # concretization until execution.
          try:
            features = int(features)
            if op == "*":
              features *= val
            elif op == "%" and features % val == 0:
              features //= val
            else:
              continue
            features = str(features)
          except ValueError:
            v, div, mul = split_div_mul(features)
            if op == "*":
              mul *= val
            else:
              div *= val
            if div > mul:
              if div % mul > 0:
                continue
              factor = div // mul
              features = f"{v}{op}{factor}"
            elif div < mul:
              if mul % div > 0:
                continue
              factor = mul // div
              features = f"{v}{op}{factor}"
            else:
              features = str(v)
          features_list.append(features)
      return set(features_list)

    if op_type == OpType.DENSE:
      features_list = make_features()
      return {"features": features_list}, {}
    elif op_type == OpType.CONV:
      features_list = make_features()
      feature_group_count = kwarg_defaults.get("feature_group_count", set())
      if full:
        feature_group_count = feature_group_count.union(features_list)
      return {
          "features": features_list,
          "kernel_size": kwarg_defaults.get("kernel_size", set()),
          "strides": kwarg_defaults.get("strides", set()),
          "padding": set(["SAME"]),
          "feature_group_count": set(feature_group_count),
          "kernel_dilation": kwarg_defaults.get("kernel_dilation", set())
      }, {}
    elif op_type == OpType.GROUP_NORM:
      num_groups = list(kwarg_defaults.get("num_groups", set()))
      if full:
        for group_factor in cls.FULL_FEATURE_FACTORS[:3]:
          num_groups.append(f"S:-1{group_factor}")
      return {"num_groups": set(num_groups)}, {}
    elif op_type == OpType.DROPOUT:
      rates = list(kwarg_defaults.get("rate", []))
      rates.append(.2)  # dropout rate of .2 to match efficient-net.
      return {"rate": set(rates)}, {}
    elif op_type in [OpType.AVG_POOL, OpType.MAX_POOL]:
      return {}, {"window_shape": kwarg_defaults.get("window_shape", set()),
                  "strides": kwarg_defaults.get("strides", set()),
                  "padding": set(["SAME"])}
    elif op_type == OpType.MEAN:
      return {}, {"axis": set([1, 2, 3])}
    else:
      assert False, f"op_type {op_type} not supported"

  @classmethod
  def kwargs_for_op_to_product(
      cls,
      op_kwargs,
      input_kwargs
  ):
    """Returns an iterator over all settings for (op_kwargs, input_kwargs)."""
    def kwargs_dict_iter(kwargs):
      kwargs_tuple_iters = []
      for k, v_iter in kwargs.items():
        kwargs_tuple_iters.append([(k, v) for v in v_iter])
      kwargs_tuple_iters = itertools.product(*kwargs_tuple_iters)
      for kwargs_tuple in kwargs_tuple_iters:
        yield dict(kwargs_tuple)
      return
    kwargs_iters = [kwargs_dict_iter(k) for k in [op_kwargs, input_kwargs]]
    all_kwargs = list(itertools.product(*kwargs_iters))
    return all_kwargs

  @classmethod
  def op_enumerator(
      cls,
      prefix = None,
      kwarg_defaults = None,
      full = True,
      op_types = None,
  ):
    if not prefix:
      prefix = ""
    elif not prefix.endswith("/"):
      prefix = f"{prefix}/"

    kwarg_defaults = cls.make_default_kwargs(kwarg_defaults, full)

    if op_types is None:
      op_types = OpType

    for op_type in op_types:

      name = f"{prefix}{op_type.name.lower()}"
      inputs = ["inputs"]

      if op_type in [
          OpType.IDENTITY,
          OpType.NONE,
          OpType.DENSE_GENERAL,
          OpType.ADD,
          OpType.MUL,
          OpType.SCALAR_ADD,
          OpType.DOT_GENERAL,
          OpType.EINSUM,
          OpType.FLATTEN,
          OpType.RESHAPE,
          OpType.TRANSPOSE,
          OpType.PARAM,
          OpType.SELF_ATTENTION,
          OpType.STOCH_DEPTH,
          OpType.MEAN,
      ]:
        # Not supported for synthesis.
        pass
      elif op_type in [
          OpType.SCALAR_MUL,
          OpType.BATCH_NORM,
          OpType.LAYER_NORM,
          OpType.RELU,
          OpType.GELU,
          OpType.SWISH,
          OpType.SIGMOID,
          OpType.SOFTMAX,
      ]:
        # No kwargs.
        yield new_op(name, op_type, inputs)
      elif op_type in [
          OpType.DENSE,
          OpType.CONV,
          OpType.GROUP_NORM,
          OpType.AVG_POOL,
          OpType.MAX_POOL,
          OpType.DROPOUT,
      ]:
        op_kwargs_dict, input_kwargs_dict = cls.all_kwargs_for_op_type(
            kwarg_defaults, full, op_type)
        for op_kwargs, input_kwargs in cls.kwargs_for_op_to_product(
            op_kwargs_dict, input_kwargs_dict):
          yield new_op(
              name,
              op_type,
              inputs,
              op_kwargs=op_kwargs,
              input_kwargs=input_kwargs)
      else:
        assert False, f"op_type {op_type} not supported"
    return
