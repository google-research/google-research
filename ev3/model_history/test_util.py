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

"""Helper methods to test ev3.model_history."""

import jax
import jax.numpy as jnp
import optax

from ev3 import base
from ev3.model_history import struct as model_history_struct
from ev3.utils import eval_util
from ev3.utils import test_util as tu


class XEntLossState(base.EvalState):
  temperature: float


DATA_SIZE = 100000


def get_loss_fn(num_labels, use_batch_norm):
  """Returns the cross-entropy loss function."""

  # Cross-entropy loss functions
  def xent_fn_1(logits, label, temperature):
    one_hot_label = jax.nn.one_hot(label, num_labels)
    return optax.softmax_cross_entropy(
        logits=temperature * logits, labels=one_hot_label
    )

  xent_kwargs = {
      'batch_axes': (0, 0, None),
      'get_batch_size_fn': tu.get_batch_size_for_xent,
      'use_vmap': True,
  }
  mean_xent_fn = eval_util.get_mean_eval_fn(xent_fn_1, **xent_kwargs)

  def loss_fn(
      model_params,
      model_graph,
      loss_state,
      batch,
  ):
    if use_batch_norm:
      logits, batch_stats, labels = eval_util.basic_pred_label_extractor_bn(
          model_params, batch, model_graph
      )
      return mean_xent_fn(logits, labels, loss_state.temperature), batch_stats
    else:
      logits, labels = eval_util.basic_pred_label_extractor(
          model_params, batch, model_graph
      )
      return mean_xent_fn(logits, labels, loss_state.temperature)

  return jax.jit(loss_fn)


def get_metric_fns():
  """Returns the averaged and vectorized accuracy metric functions."""

  # Accuracy metric functions
  def accuracy_fn_1(logits, labels):
    preds = jnp.argmax(logits, axis=-1)
    return preds == labels

  vec_accuracy_fn = eval_util.vectorize_eval_fn(accuracy_fn_1)
  mean_accuracy_fn = eval_util.get_mean_eval_fn(accuracy_fn_1)

  @jax.jit
  def vec_metric_fn(
      model_params,
      model_graph,
      batch,
  ):
    logits, labels = eval_util.basic_pred_label_extractor(
        model_params, batch, model_graph
    )
    return vec_accuracy_fn(logits, labels)

  @jax.jit
  def mean_metric_fn(
      model_params,
      model_graph,
      batch,
  ):
    logits, labels = eval_util.basic_pred_label_extractor(
        model_params, batch, model_graph
    )
    return mean_accuracy_fn(logits, labels)

  return vec_metric_fn, mean_metric_fn


def get_ev3_state_inputs(
    num_features,
    num_labels,
    batch_size,
    rand_seed,
    dataset_name='test',
    process_fn=None,
    model_name='2lp',
    use_batch_norm=False,
):
  """Returns the inputs needed to create EV3 states."""

  rand_key = jax.random.PRNGKey(rand_seed)
  (
      propose_data_key,
      optimize_data_key,
      decide_data_key,
      data_param_key,
      model_param_key,
      model_rand_key,
  ) = jax.random.split(rand_key, 6)

  # tu.get_data_iterator arguments.
  data_iter_kwargs = {
      'num_features': num_features,
      'num_labels': num_labels,
      'data_size': DATA_SIZE,
      'data_rand_key': propose_data_key,
      'param_rand_key': data_param_key,
      'batch_size': batch_size,
      'dataset_name': dataset_name,
      'process_fn': process_fn,
  }

  # Propose data
  propose_data_iter = tu.get_data_iterator(**data_iter_kwargs)

  # Optimize data
  data_iter_kwargs.update({
      'data_rand_key': optimize_data_key,
      'batch_size': 4 * batch_size,
  })
  optimize_data_iter = tu.get_data_iterator(**data_iter_kwargs)

  # Decide data
  data_iter_kwargs.update({
      'data_rand_key': decide_data_key,
      'batch_size': 64 * batch_size,
  })
  decide_data_iter = tu.get_data_iterator(**data_iter_kwargs)

  # Model
  batch = next(propose_data_iter)
  x = batch['feature']
  if model_name == '2lp':
    tlp = tu.TwoLayerPerceptron(
        num_hidden_nodes=2 * (num_features + num_labels),
        num_labels=num_labels,
    )
    params = tlp.init(model_param_key, x)
    model_graph = model_history_struct.ModelGraph(
        nn_model=tlp, apply_fn=tlp.apply
    )  # pytype: disable=wrong-keyword-args  # dataclass_transform
  elif model_name == 'mlp':
    mlp = tu.MLP(layer_widths=[2, 3], num_labels=num_labels)
    params = mlp.init(model_param_key, x)
    model_graph = model_history_struct.ModelGraph(
        nn_model=mlp, apply_fn=mlp.apply
    )  # pytype: disable=wrong-keyword-args  # dataclass_transform
  elif model_name == 'cnn':
    cnn = tu.CNN(num_classes=num_labels, dtype=x.dtype)
    params = cnn.init(model_param_key, x)
    model_graph = model_history_struct.ModelGraph(
        nn_model=cnn, apply_fn=cnn.apply
    )  # pytype: disable=wrong-keyword-args  # dataclass_transform
  else:
    raise ValueError(f'Model name {model_name} is not supported.')

  model = model_history_struct.Model(
      graph=model_graph,
      params=params,
      stable_params=params,
      rand_key=model_rand_key,
  )

  # Loss
  loss_states = [XEntLossState(temperature=t * 0.5) for t in range(1, 4)]  # pytype: disable=wrong-keyword-args  # dataclass_transform
  loss_fn = get_loss_fn(num_labels, use_batch_norm)

  # Metric
  vec_metric_fn, mean_metric_fn = get_metric_fns()

  return (
      propose_data_iter,
      optimize_data_iter,
      decide_data_iter,
      loss_fn,
      loss_states,
      vec_metric_fn,
      mean_metric_fn,
      model,
  )
