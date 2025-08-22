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

# Copyright 2024 Google LLC
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

"""Definitions for the main model."""

import functools
from typing import Callable, List, Optional

import modules
import tensorflow as tf
import transformers



# These paths can be used to retrieve local checkpoints.
_MODEL_BASE_PATH = ""
_TOKENIZER_BASE_PATH = ""


def get_pretrained_tokenizer(
    tokenizer_path,
    local_files_only = False,
):
  """Obtains pretrained tokenizer."""
  return transformers.AutoTokenizer.from_pretrained(
      tokenizer_path,
      local_files_only=local_files_only,
  )


def get_model_tokenizer_path_from_name(
    model_name, get_tokenizer = False
):
  """Gets model or tokenizer path from model name."""
  base_path = _TOKENIZER_BASE_PATH if get_tokenizer else _MODEL_BASE_PATH
  if model_name == "bert":
    return base_path + "bert-base-cased"
  elif model_name == "roberta":
    return base_path + "roberta-base"
  else:
    raise ValueError(f"Unsupported model: {model_name}")


def get_pretrained_model(
    model_path,
    local_files_only = False,
):
  """Obtains pretrained model."""
  pretrained_model = transformers.TFAutoModel.from_pretrained(
      model_path,
      local_files_only=local_files_only,
  )
  _mark_submodules_non_trainable(pretrained_model)
  return pretrained_model


def get_prediction_model(
    pretrained_model,
    seq_len,
    hidden_dimension = 768,
    num_classes = None,
):
  """Creates prediction model that wraps around main transformer model."""
  inputs = tf.keras.Input((seq_len,), dtype=tf.int32)
  dense_layer = tf.keras.layers.Dense(hidden_dimension, activation="gelu")
  if num_classes:
    predictor = tf.keras.layers.Dense(num_classes)
  else:
    predictor = tf.keras.layers.Dense(1)

  pooled_output = pretrained_model(inputs)["pooler_output"]
  logits = predictor(dense_layer(pooled_output))
  return tf.keras.Model(inputs, logits)


def get_bert_replace_layer_fn(
    module_to_replace = "query",
):
  """Creates layer replacement functions to swap in adaptive dense layers."""
  def make_new_einsum_dense(dense_layer, rank):
    # construct new dense layer from pre-built dense_layer
    # TODO(yihed): experiment with separate dimension for different heads.
    einsum_dense = tf.keras.layers.EinsumDense(
        # n represents number of tokens, d the latent dimension.
        "...nd,dc->...nc",
        output_shape=dense_layer.kernel.shape[-1],
        bias_axes="c",
        activation=dense_layer.activation,
        trainable=False,
    )
    mock_input = tf.zeros((1, dense_layer.kernel.shape[0]))
    # build layer.
    einsum_dense(mock_input)
    einsum_dense.set_weights([dense_layer.kernel, dense_layer.bias])
    return modules.AdaptiveEinsumDense(einsum_dense, rank=rank)

  def bert_replace_module_fn(
      attention_module, rank, replaced_module
  ):
    # Model hierarchy takes form e.g.
    # bert_model.bert.encoder.layer[0].attention.self_attention.query,
    # Or bert_model_tf.roberta.encoder.layer[0].attention.self_attention.query.
    # Or bert_model_tf.roberta.encoder.layer[0].attention.dense_output.dense.
    new_dense = make_new_einsum_dense(
        getattr(attention_module, replaced_module),
        rank=rank,
    )
    setattr(attention_module, replaced_module, new_dense)
    # don't set as this is recursive:
    # getattr(attention_module, replaced_module).trainable = True

  replace_layer_fn = functools.partial(
      bert_replace_module_fn, replaced_module=module_to_replace
  )
  return replace_layer_fn


def _mark_submodules_non_trainable(module):
  """Marks all, but only, *leaf* submodules of a model as non-trainable."""
  submodules = module.submodules
  if not submodules or "embeddings" in module.name:
    module.trainable = False
  else:
    for submodule in submodules:
      _mark_submodules_non_trainable(submodule)


def get_model_layer(model, layer_index, model_name):
  """Gets the layer of the model at the specified index."""
  if model_name == "bert":
    return model.bert.encoder.layer[layer_index]
  elif model_name == "roberta":
    return model.roberta.encoder.layer[layer_index]
  else:
    raise ValueError(f"Unsupported model: {model_name}")


def replace_model_dense_layers(
    layers_to_replace,
    replace_layer_fn,
    ranks,
    module_to_replace = "query",
):
  """Replaces specified model dense layers with adaptive layers.

  Args:
    layers_to_replace: modules for which to replace specified dense layers.
    replace_layer_fn: callable used to replace the dense layers.
    ranks: list of ranks to use for each layer.
    module_to_replace: which module to replace, can be query, key, value, dense.
  """
  if len(layers_to_replace) != len(ranks):
    raise ValueError(
        "Number of ranks specified must match the number of model layers."
    )
  for i, layer in enumerate(layers_to_replace):
    # layer can be e.g. bert_model.bert.encoder.layer[0]
    if module_to_replace == "dense":
      replace_layer_fn(layer.attention.dense_output, ranks[i])
    else:
      replace_layer_fn(layer.attention.self_attention, ranks[i])
