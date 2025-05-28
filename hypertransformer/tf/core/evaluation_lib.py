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

"""Evaluation library."""

import copy

import dataclasses
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow.compat.v1 as tf

from hypertransformer.tf.core import common
from hypertransformer.tf.core import common_ht
from hypertransformer.tf.core import layerwise
from hypertransformer.tf.core import train_lib

# In some evaluation scenarios we may want to run multiple "CNN batches"
# for the same meta-training Transformer batch, which means that we have
# to fix Transformer batch and update it only occasionally.
#
# This is done in `make_train_samples` that returns:
# (a) cached Transformer images (or original tensor if `same_batch` is False);
# (b) cached Transformer labels;
# (c) operation updating Transformer image/label cache (or None if `same_batch`
#     parameter is False).
TrainSamples = Tuple[tf.Tensor, tf.Tensor, Optional[tf.Operation]]

# Type returned by `evaluate_dataset`:
# (a) a dictionary mapping a task number to a list of accuracies for all
#     batches;
# (b) a list of mean accuracies for all tasks.
Accuracies = Tuple[Dict[int, List[float]], List[float]]

# Either a list of tf.Variables or a function that returns such a list.
VarList = Union[List[tf.Variable], Callable[Ellipsis, List[tf.Variable]]]


@dataclasses.dataclass
class EvaluationConfig:
  num_task_evals: int
  num_eval_batches: int
  load_vars: Optional[VarList] = None


def dataset_with_custom_labels(
    model_config,
    dataset_config
    ):
  """Returns a dataset with a controlled label set (should be reshuffled)."""
  custom_labels = copy.copy(dataset_config.use_label_subset)
  dataset_config = dataclasses.replace(  # pytype: disable=wrong-arg-types  # dataclasses-replace-types
      dataset_config, use_label_subset=lambda: custom_labels)
  dataset, _ = train_lib.make_dataset(model_config=model_config,
                                      data_config=dataset_config,
                                      shuffle_labels=False)
  return dataset, custom_labels


def make_train_samples(dataset,
                       same_batch = True):
  """Makes input samples for training a baseline model."""
  train_images = dataset.transformer_images
  train_labels = dataset.transformer_labels
  assign_op = None

  if same_batch:
    batch_size = train_images.shape[0]
    train_images = tf.get_variable('train_images',
                                   shape=train_images.shape,
                                   dtype=train_images.dtype,
                                   initializer=tf.zeros_initializer,
                                   trainable=False)
    train_labels = tf.get_variable('train_labels',
                                   shape=(batch_size,),
                                   dtype=train_labels.dtype,
                                   initializer=tf.zeros_initializer,
                                   trainable=False)
    assign_op = tf.group(tf.assign(train_images, dataset.transformer_images),
                         tf.assign(train_labels, dataset.transformer_labels))

  return train_images, train_labels, assign_op


def evaluate_dataset(custom_labels,
                     dataset,
                     assign_op,
                     outputs,
                     eval_config
                     ):
  """Runs evaluation loop for a specific dataset."""
  sess = tf.get_default_session()
  test_accs = {}
  all_accs = []
  for task_num in range(eval_config.num_task_evals):
    if custom_labels:
      np.random.shuffle(custom_labels)
    sess.run(dataset.randomize_op)
    # Assign op should be executed last for us to have the same augmentation
    # and labels for both Transformer and CNN samples
    if assign_op is not None:
      sess.run(assign_op)
    accs = []
    for _ in range(eval_config.num_eval_batches):
      accs.append(sess.run(outputs.accuracy))
    test_accs[task_num] = accs
    all_accs.append(np.mean(accs))
  return test_accs, all_accs


def evaluation_loop(state,
                    custom_labels,
                    dataset,
                    assign_op,
                    outputs,
                    eval_config,
                    ):
  """Model evaluation loop."""
  with tf.Session():
    common.init_training(state)
    return evaluate_dataset(
        custom_labels=custom_labels,
        dataset=dataset,
        assign_op=assign_op,
        outputs=outputs,
        eval_config=eval_config)


def apply_layerwise_model(model,
                          model_config,
                          dataset):
  """Applies a layerwise model to a dataset."""
  with tf.variable_scope('model'):
    weight_blocks = model.train(dataset.transformer_images,
                                dataset.transformer_labels,
                                mask=dataset.transformer_masks)
    predictions = model.evaluate(dataset.cnn_images,
                                 weight_blocks=weight_blocks)

  pred_labels = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
  accuracy = tf.cast(tf.math.equal(dataset.cnn_labels, pred_labels),
                     tf.float32)
  accuracy = tf.reduce_sum(accuracy) / model_config.num_cnn_samples
  return common.ModelOutputs(predictions=pred_labels,
                             accuracy=accuracy)


def build_layerwise_model(model_config,
                          dataset
                          ):
  """Builds a layerwise model."""
  model = layerwise.build_model(model_config.cnn_model_name,
                                model_config=model_config)
  return apply_layerwise_model(model,
                               model_config=model_config,
                               dataset=dataset)


def evaluate_layerwise(model_config,
                       dataset_config,
                       eval_config):
  """Evaluates a pretrained 'layerwise' model."""
  dataset, custom_labels = dataset_with_custom_labels(
      model_config, dataset_config)
  images, labels, assign_op = make_train_samples(dataset, same_batch=True)
  dataset.transformer_images, dataset.transformer_labels = images, labels
  outputs = build_layerwise_model(model_config, dataset)

  load_vars = eval_config.load_vars
  if callable(load_vars):
    load_vars = load_vars()

  state = common.TrainState(train_op=tf.no_op(),
                            saver=tf.train.Saver(load_vars))

  return evaluation_loop(state=state,
                         custom_labels=custom_labels,
                         dataset=dataset,
                         assign_op=assign_op,
                         outputs=outputs,
                         eval_config=eval_config)
