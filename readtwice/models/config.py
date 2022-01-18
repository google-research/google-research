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

"""The config for ReadItTwice BERT model."""

import base64
import collections
import json
import os
from typing import Optional, Text

import dataclasses
import tensorflow.compat.v1 as tf

MODEL_CONFIG_FILENAME = "read_it_twice_bert_config.json"


@dataclasses.dataclass(frozen=True)
class ReadItTwiceBertConfig:
  """Configuration for `ReadItTwice BERT model."""

  # Vocabulary size of `token_ids`.
  vocab_size: int

  # Whether to enable non-entity tokens to attend entity-based summaries.
  use_sparse_memory_attention: bool

  # Max length of the `token_ids`.
  max_seq_length: int = 512

  # Max number of blocks within the same document - max value for `block_pos`
  max_num_blocks_per_document: int = 256

  # Whether to add positional embeddings based on block_pos
  cross_attention_pos_emb_mode: Optional[Text] = None

  # Size of `token_ids` embeddings. The default of `None`
  # makes this equal to `hidden_size` like original BERT, but it can be set
  # to a smaller value (e.g. 128) like ALBERT. Must be positive.
  embedding_size: Optional[int] = None

  # Size of the encoder layers and the pooler layer.
  hidden_size: int = 768

  # Number of hidden layers in the Transformer encoder.
  num_hidden_layers: int = 12

  # Number of attention heads for each attention layer
  # in the Transformer encoder.
  num_attention_heads: int = 12

  # The size of the "intermediate" (i.e., feed-forward)
  # layer in the Transformer encoder.
  intermediate_size: int = 3072

  # The non-linear activation function (function or string) in the
  # encoder and pooler.
  hidden_act: Text = "gelu"

  # If True, key and value projections will be shared
  # between main-to-main and main-to-side components in attention layers.
  # This results in 1 key projection per layer instead of 2 (and similarly
  # for value projections).
  share_kv_projections: bool = False

  # The dropout probability for all fully connected
  # layers in the embeddings, encoder, and pooler.
  hidden_dropout_prob: float = 0.1

  # The dropout ratio for the attention probabilities.
  attention_probs_dropout_prob: float = 0.1

  # The stdev of the truncated_normal_initializer for
  # initializing all weight matrices.
  initializer_range: float = 0.02

  # How often to checkpoint activations. The
  # default of 0 stores all activations. If greater than 0, activations are
  # are recomputed as necessary when calculating gradients to save memory.
  grad_checkpointing_period: int = 0

  # The type of a model to use for the second read. Possible options are
  # (1) 'from_scratch' (default) -- equivalent to re-reading the input
  # from scratch using summaries as a side input.
  # (2) 'new_layers' -- apply new layers on top of the first read.
  second_read_type: Text = "from_scratch"

  # Only applicable when `second_read_type` is `new_layers`.
  # The number of Transformer layers to apply.
  second_read_num_new_layers: Optional[int] = None

  second_read_num_cross_attention_heads: Optional[int] = None

  second_read_enable_default_side_input: bool = False

  summary_mode: Text = "cls"

  # Post-processing to apply on top of extracted summaries. Options are
  # (1) "none" (default) -- options are passes as is.
  # (2) "linear" -- apply a linear linear
  # (3) "transformer" -- apply a Transformer on top of the summaries with
  # `summary_postprocessing_num_layers` number of layers
  summary_postprocessing_type: Text = "none"

  # Only applicable when `summary_postprocessing_type` is `transformer`.
  # The number of Transformer layers to apply.
  summary_postprocessing_num_layers: Optional[int] = None

  # Whether to apply top-K op before computing attention over summaries.
  # Currently, only supported in `second_read_num_new_layers=cross_attend_once`.
  cross_attention_top_k: Optional[int] = None

  text_block_extract_every_x: Optional[int] = None

  @classmethod
  def from_json_file(cls, json_file_path):
    """Constructs a `ReadItTwiceBertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file_path, "r") as reader:
      text = reader.read()
    return cls(**json.loads(text))

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n"


def get_model_config(
    model_dir,
    source_file = None,
    source_base64 = None,
    write_from_source = True):
  """Reads model config from `model_dir`, falling back to source file/base64.

  If the JSON config file isn't found in `model_dir`, then exactly one of
  `source_file` or `source_base64` should be given to read the config from
  instead.

  Args:
    model_dir: Model directory containing the config file.
    source_file: Optional source file to read config file from if not present in
      `model_dir`.
    source_base64: Optional Base64 encoding of JSON content to read config file
      from if not present in `model_dir`.  If this is specified, then
      `source_file` must not be.
    write_from_source: If True (default), write the source config to `model_dir`
      if it isn't present already.

  Returns:
    A ReadItTwiceBertConfig object.
  """
  model_config_path = os.path.join(model_dir, MODEL_CONFIG_FILENAME)
  if tf.io.gfile.exists(model_config_path):
    return ReadItTwiceBertConfig.from_json_file(model_config_path)

  if source_file is None and source_base64 is None:
    raise ValueError(
        "Either `source_file` or `source_base64` must be specified for initial "
        "model configuration.")
  elif source_file is not None and source_base64 is not None:
    raise ValueError("Only one of `source_file` or `source_base64` can be "
                     "specified, not both.")

  if source_file is not None:
    with tf.io.gfile.GFile(source_file, "r") as reader:
      model_config_json_str = reader.read()
  elif source_base64 is not None:
    model_config_json_str = base64.b64decode(
        source_base64.encode("utf-8")).decode("utf-8")
  model_config_dict = json.loads(
      model_config_json_str, object_pairs_hook=collections.OrderedDict)

  if write_from_source:
    with tf.io.gfile.GFile(model_config_path, "w") as writer:
      writer.write(model_config_json_str)

  return ReadItTwiceBertConfig(**model_config_dict)
