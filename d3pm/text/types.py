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

"""A set of standard types and classes for the D3PM codebase."""

import dataclasses
import datetime
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union

import jax
import seqio
import tensorflow as tf

CallableFn = Callable[Ellipsis, Any]

DType = Any
Shape = Tuple[Union[None, int], Ellipsis]
Vocabulary = seqio.SentencePieceVocabulary


@dataclasses.dataclass
class DatasetInfo:
  """A dataset object."""
  # A set of features that the dataset object should provide.
  features: Iterable[str]

  # A pytree containing the shapes and dtypes of arrays in a dataset batch.
  shapes: Any

  # the vocabulary to use, if applicable
  vocab: Optional[Vocabulary] = None


def _convert_spec(spec):
  return jax.ShapeDtypeStruct(shape=spec.shape, dtype=spec.dtype.as_numpy_dtype)


def get_dataset_info(ds, vocab=None):
  """Wraps a set of TFDS datasets with vocabularies."""
  shapes = jax.tree.map(_convert_spec, ds.element_spec)

  if not isinstance(shapes, dict):
    raise ValueError(
        f"Dataset entries must always be dictionaries. Found {type(shapes)}.")

  features = list(shapes.keys())
  dataset_info = DatasetInfo(vocab=vocab, features=features, shapes=shapes)
  return dataset_info


def _apply_fns(dataset, fns):
  for fn in fns:
    dataset = dataset.map(fn)

  return dataset


@dataclasses.dataclass
class Dataset:
  """A D3PM dataset class."""

  # the core tfds or iterator dataset
  dataset: tf.data.Dataset

  # the dataset vocab if applicable, used to generate the info object.
  _vocab: Optional[Vocabulary] = None

  @property
  def info(self):
    """Returns the dataset info object specifying shapes and vocabularies."""
    return get_dataset_info(self.dataset, vocab=self._vocab)

  def map(self, fn_or_fns):
    """Applies a set of functions to the dataset."""
    if not isinstance(fn_or_fns, Iterable):
      fn_or_fns = [fn_or_fns]

    ds = _apply_fns(self.dataset, fn_or_fns)

    return Dataset(
        dataset=ds,
        _vocab=self._vocab,
    )

  def prefetch(self):
    return Dataset(
        dataset=self.dataset.prefetch(tf.data.experimental.AUTOTUNE),
        _vocab=self._vocab,
    )

  def get_iterator(self):
    return self.dataset.as_numpy_iterator()

  def __iter__(self):
    return self.get_iterator()


@dataclasses.dataclass
class Metric:
  # the value to log
  value: Any

  # the name of the metric
  name: str

  # scalar, text, image
  type: str


MetricFn = Callable[Ellipsis, Metric]


@dataclasses.dataclass
class Feature:
  """A Feature contained in a dataset or returned by a model."""

  # The shape of the Feature
  shape: Shape

  # The DType of the feature
  dtype: DType

  # Vocabulary if the feature is a text feature
  vocab: Optional[Vocabulary]


FeatureDict = Mapping[str, Feature]


class Clock:
  """A simple clock to help with reloading."""

  def __init__(self,):
    self.time = datetime.datetime.now().strftime("%H:%M:%S")

  def __repr__(self):
    return self.time

  def __str__(self):
    return self.time


@dataclasses.dataclass
class State:
  """State that gets passed to the loss function."""

  # static state that gets enclosed in the loss and predict functions
  static_state: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  # dynamic state that gets passed as an argument to the JIT-ed function
  dynamic_state: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  # takes static_args + dynamic_args + model itself, returns new dynamic_args
  dynamic_update_fn: Optional[Any] = None

  dynamic_update_freq: int = 1

  jit_update: bool = True


def _default_init_fn(dataset_info, task):
  del dataset_info, task

  return State({}, {}, None, 1)


@dataclasses.dataclass
class Task:
  """Defines a D3PM task."""

  # a loss function
  loss_fn: CallableFn

  # a list of features that should be passed to the task.
  input_features: Iterable[str]

  # the inputs that are passed to the model during initialization
  init_features: Iterable[str]

  # an optional prediction function
  predict_fn: Optional[CallableFn] = None

  # features to pass to predict_fn
  predict_features: Optional[Iterable[str]] = None

  # tells the trainer whether to vmap over the batch axis
  vmap_batch: bool = False

  # any additional information that the model might need. Passed to the loss_fn,
  # score_fn, and metric_fns as kwargs.
  state_init_fn: CallableFn = _default_init_fn

  # a list of functions to call to preprocess the inputs.
  preprocessors: Iterable[CallableFn] = dataclasses.field(default_factory=list)

  # a list of functions to call on the set of output features. This can include
  # the loss function, as well as other metrics like BLEU.
  metric_fns: Iterable[MetricFn] = dataclasses.field(default_factory=list)

  # function which takes the model_cls object and returns an updated model_cls.
  model_init_fn: Optional[CallableFn] = None

  # monitors the time at which the object was created
  creation_time: Clock = Clock()
