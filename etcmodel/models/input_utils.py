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

"""Utilities for creating or transforming model inputs."""

import base64
import collections
import json
import os
import re
from typing import (Dict, Iterable, List, Mapping, Optional, Sequence, Text,
                    Tuple, MutableSequence, Union)

import attr
import tensorflow.compat.v1 as tf

from etcmodel import feature_utils
from etcmodel import tensor_utils
from etcmodel.models import modeling

MODEL_CONFIG_FILENAME = 'etc_config.json'


@attr.s
class GlobalLocalTransformerSideInputs(object):
  """GlobalLocalTransformer side inputs ("att_mask" and "relative_att_ids").

  See `GlobalLocalTransformerLayers.call()` in `layers/transformer.py` for a
  description of these side inputs.
  """

  l2l_att_mask = attr.ib()  # type: Optional[tf.Tensor]
  g2g_att_mask = attr.ib()  # type: Optional[tf.Tensor]
  l2g_att_mask = attr.ib()  # type: Optional[tf.Tensor]
  g2l_att_mask = attr.ib()  # type: Optional[tf.Tensor]
  l2l_relative_att_ids = attr.ib()  # type: Optional[tf.Tensor]
  g2g_relative_att_ids = attr.ib()  # type: Optional[tf.Tensor]
  l2g_relative_att_ids = attr.ib()  # type: Optional[tf.Tensor]
  g2l_relative_att_ids = attr.ib()  # type: Optional[tf.Tensor]

  def to_dict(self, exclude_none_values=False):
    """Returns attributes in a Python dictionary."""
    if exclude_none_values:
      return {k: v for k, v in self.__dict__.items() if v is not None}
    else:
      return dict(self.__dict__)


def get_model_config(
    model_dir: Text,
    source_file: Optional[Text] = None,
    source_base64: Optional[Text] = None,
    write_from_source: Optional[bool] = True) -> modeling.EtcConfig:
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
    An `EtcConfig` object.
  """
  model_config_path = os.path.join(model_dir, MODEL_CONFIG_FILENAME)
  if tf.io.gfile.exists(model_config_path):
    return modeling.EtcConfig.from_json_file(model_config_path)

  if source_file is None and source_base64 is None:
    raise ValueError(
        'Either `source_file` or `source_base64` must be specified for initial '
        'model configuration.')
  elif source_file is not None and source_base64 is not None:
    raise ValueError('Only one of `source_file` or `source_base64` can be '
                     'specified, not both.')

  if source_file is not None:
    with tf.io.gfile.GFile(source_file, 'r') as reader:
      model_config_json_str = reader.read()
  elif source_base64 is not None:
    model_config_json_str = base64.b64decode(
        source_base64.encode('utf-8')).decode('utf-8')
  model_config_dict = json.loads(
      model_config_json_str, object_pairs_hook=collections.OrderedDict)

  if write_from_source:
    with tf.io.gfile.GFile(model_config_path, 'w') as writer:
      writer.write(model_config_json_str)

  return modeling.EtcConfig.from_dict(model_config_dict)


def make_global_local_transformer_side_inputs(
    long_breakpoints: tf.Tensor,
    global_breakpoints: tf.Tensor,
    sentence_ids: tf.Tensor,
    local_radius: int,
    relative_pos_max_distance: int,
    use_hard_g2l_mask: bool = False,
    use_hard_l2g_mask: bool = False,
    name: Optional[Text] = None) -> GlobalLocalTransformerSideInputs:
  """Makes side input tensors based on the given breakpoints and sentence ids.

  Note that the concept of `breakpoints` is primarily relevant for
  pre-training, where we pack multiple shorter examples into 1 long example.
  The breakpoints are used to separate the original shorter examples, with
  a `1` occurring at the last token of each packed example.
  For instance, if we packed three examples, with long lengths 2, 3, and 4,
  and the maximum long length is 10, then `long_breakpoints` would look
  like: [0, 1, 0, 0, 1, 0, 0, 0, 1, 0].

  If we're not packing examples (e.g. for all our fine-tuning tasks), the
  `breakpoints` features will only have a single `1`.  For instance, if our
  example has 8 long tokens, and the max long length is 10, then
  `long_breakpoints` would look like: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0].
  Note that in this case a classic BERT `input_mask` can be obtained from
  `long_breakpoints` via `tf.cumsum(long_breakpoints, axis=-1, reverse=True)`.

  The `sentence_ids` tensor defines the connection between long and global
  tokens.  There's one global token per sentence, and the value in
  `sentence_ids` represents which global token (sentence) a given long token
  belongs to.  For instance, if we had a single example with three sentences
  of lengths 3, 1, and 2, and the max length is 10, then `sentence_ids`
  would look like: [0, 0, 0, 1, 2, 2, 0, 0, 0, 0].
  Note that the padding tokens use value 0 above, but any value is fine since
  padding tokens will be masked.

  Args:
    long_breakpoints: <int32>[batch_size, long_seq_len] Tensor of ending
      breakpoints separating different packed examples.
    global_breakpoints: <int32>[batch_size, global_seq_len] Tensor of ending
      breakpoints separating different packed examples.
    sentence_ids: <int32>[batch_size, long_seq_len] Tensor of ids indicating
      which sentence each token belongs to. For this dataset, "sentence" refers
      to real natural language sentence, not a BERT "sentence" from the "next
      sentence prediction" task.
    local_radius: How many tokens to the left/right for input tokens to locally
      self-attend to. For example, a value of 1 would allow each token to only
      attend to 1 token to the left and 1 token to the right of it.
    relative_pos_max_distance: Maximum distance to use for relative position
      representations. All larger distances will be clipped to this value. Use 0
      to skip relative position representations entirely.
    use_hard_g2l_mask: If True, global tokens only attend to tokens of the
      corresponding sentences in the long input. If False, global tokens attend
      to all sentences within the corresponding global example.
    use_hard_l2g_mask: If True, long tokens only attend to tokens of the
      corresponding global tokens. If False, long tokens attend to all the
      global tokens within the corresponding global example.
    name: A name for the operation (optional).

  Returns:
    A `GlobalLocalTransformerSideInputs` with all relevant tensors set.
  """
  with tf.name_scope(name or 'make_global_local_transformer_side_inputs'):
    long_breakpoints = tf.convert_to_tensor(long_breakpoints)
    global_breakpoints = tf.convert_to_tensor(global_breakpoints)

    long_example_ids = tf.cumsum(long_breakpoints, axis=-1, reverse=True)
    global_example_ids = tf.cumsum(global_breakpoints, axis=-1, reverse=True)

    return make_global_local_transformer_side_inputs_from_example_ids(
        long_example_ids=long_example_ids,
        global_example_ids=global_example_ids,
        sentence_ids=sentence_ids,
        local_radius=local_radius,
        relative_pos_max_distance=relative_pos_max_distance,
        use_hard_g2l_mask=use_hard_g2l_mask,
        use_hard_l2g_mask=use_hard_l2g_mask,
        name=name)


def make_global_local_transformer_side_inputs_from_example_ids(
    long_example_ids: tf.Tensor,
    global_example_ids: tf.Tensor,
    sentence_ids: tf.Tensor,
    local_radius: int,
    relative_pos_max_distance: int,
    use_hard_g2l_mask: bool = False,
    use_hard_l2g_mask: bool = False,
    name: Optional[Text] = None) -> GlobalLocalTransformerSideInputs:
  """Makes side input tensors based on the given example and sentence ids.

  When packing examples (e.g. for pre-training), each example must have a
  unique id for `long_example_ids`/`global_example_ids`, and padding must
  also have a unique id distinct from all the example ids.

  When not packing examples, there will simply be two unique ids: one for
  example tokens, and another for padding.  Note that in this case, the classic
  BERT `input_mask` is a valid special case of `long_example_ids`.

  The other arguments have the same interpretation as in
  `make_global_local_transformer_side_inputs`.

  Args:
    long_example_ids: <int32>[batch_size, long_seq_len] Tensor of example ids of
      different packed examples.
    global_example_ids: <int32>[batch_size, global_seq_len] Tensor of example
      ids of different packed examples.
    sentence_ids: <int32>[batch_size, long_seq_len] Tensor of ids indicating
      which sentence each token belongs to. For this dataset, "sentence" refers
      to real natural language sentence, not a BERT "sentence" from the "next
      sentence prediction" task.
    local_radius: How many tokens to the left/right for input tokens to locally
      self-attend to. For example, a value of 1 would allow each token to only
      attend to 1 token to the left and 1 token to the right of it.
    relative_pos_max_distance: Maximum distance to use for relative position
      representations. All larger distances will be clipped to this value. Use 0
      to skip relative position representations entirely.
    use_hard_g2l_mask: If True, global tokens only attend to tokens of the
      corresponding sentences in the long input. If False, global tokens attend
      to all sentences within the corresponding global example.
    use_hard_l2g_mask: If True, long tokens only attend to tokens of the
      corresponding global tokens. If False, long tokens attend to all the
      global tokens within the corresponding global example.
    name: A name for the operation (optional).

  Returns:
    A `GlobalLocalTransformerSideInputs` with all relevant tensors set.
  """
  with tf.name_scope(name or 'make_global_local_transformer_side_inputs'):
    long_example_ids = tf.convert_to_tensor(long_example_ids)
    global_example_ids = tf.convert_to_tensor(global_example_ids)
    sentence_ids = tf.convert_to_tensor(sentence_ids)

    long_seq_len = tensor_utils.get_shape_list(long_example_ids)[1]
    global_seq_len = tensor_utils.get_shape_list(global_example_ids)[1]

    l2l_att_mask = feature_utils.make_local_segmented_att_mask(
        long_example_ids, local_radius)
    g2g_att_mask = feature_utils.make_segmented_att_mask(global_example_ids)

    l2g_att_mask = tf.cast(
        tf.equal(long_example_ids[:, :, tf.newaxis],
                 global_example_ids[:, tf.newaxis, :]), tf.int32)
    g2l_att_mask = tf.transpose(l2g_att_mask, perm=[0, 2, 1])

    if use_hard_g2l_mask:
      # Have each global token attend to just one sentence instead of having
      # it attend to all the sentences within a global example.
      global_range = tf.range(global_seq_len, dtype=sentence_ids.dtype)
      hard_g2l_att_mask = tf.cast(
          tf.equal(global_range[tf.newaxis, :, tf.newaxis],
                   sentence_ids[:, tf.newaxis, :]), tf.int32)
      g2l_att_mask *= hard_g2l_att_mask

    if use_hard_l2g_mask:
      # Have each long token attend to just the corresponding global token
      # instead of having it attend to all the global tokens within a
      # global example.
      global_range = tf.range(global_seq_len, dtype=sentence_ids.dtype)
      hard_l2g_att_mask = tf.cast(
          tf.equal(sentence_ids[:, :, tf.newaxis],
                   global_range[tf.newaxis, tf.newaxis, :]), tf.int32)
      l2g_att_mask *= hard_l2g_att_mask

    batch_size = tf.shape(long_example_ids)[0]

    l2l_relative_att_ids = None
    g2g_relative_att_ids = None
    l2g_relative_att_ids = None
    g2l_relative_att_ids = None

    if relative_pos_max_distance > 0:
      relative_pos_generator = feature_utils.RelativePositionGenerator(
          relative_pos_max_distance)
      l2l_relative_att_ids = relative_pos_generator.make_local_relative_att_ids(
          seq_len=long_seq_len,
          local_radius=local_radius,
          batch_size=batch_size)
      g2g_relative_att_ids = relative_pos_generator.make_relative_att_ids(
          seq_len=global_seq_len, batch_size=batch_size)
      global_range = tf.range(global_seq_len, dtype=sentence_ids.dtype)
      l2g_relative_att_ids = tf.cast(
          tf.equal(sentence_ids[:, :, tf.newaxis],
                   global_range[tf.newaxis, tf.newaxis, :]), tf.int32)
      g2l_relative_att_ids = tf.transpose(l2g_relative_att_ids, perm=[0, 2, 1])

      # For fused attention, l2l and l2g share the same relative vocabulary, as
      # do g2g and g2l, so we add an offset for l2g and g2l so their original
      # 0/1 ids don't collide with l2l and g2g relative position ids.
      l2g_relative_att_ids += relative_pos_generator.relative_vocab_size
      g2l_relative_att_ids += relative_pos_generator.relative_vocab_size

    return GlobalLocalTransformerSideInputs(
        l2l_att_mask=l2l_att_mask,
        g2g_att_mask=g2g_att_mask,
        l2g_att_mask=l2g_att_mask,
        g2l_att_mask=g2l_att_mask,
        l2l_relative_att_ids=l2l_relative_att_ids,
        g2g_relative_att_ids=g2g_relative_att_ids,
        l2g_relative_att_ids=l2g_relative_att_ids,
        g2l_relative_att_ids=g2l_relative_att_ids)


def make_fixed_block_side_inputs(
    input_mask: tf.Tensor,
    num_tokens_per_block: int,
    local_radius: int,
    relative_pos_max_distance: int,
    use_hard_g2l_mask: bool = False,
    use_hard_l2g_mask: bool = False,
    global_token_id: int = 1,
    name: Optional[Text] = None
) -> Tuple[GlobalLocalTransformerSideInputs, tf.Tensor]:
  """Utility for creating side inputs in a "fixed blocks" pattern.

  The "fixed blocks" experiments for NQ and OpenKP are implemented via example
  generation rather than using this function, but we include this function
  to illustrate how side inputs can be generated given just a BERT-style
  `input_mask` feature.  The corresponding global tokens are generated
  as part of this function too, so no global features are required as input.

  Args:
    input_mask: <int32>[batch_size, long_seq_len] Tensor of 1 and 0 values, with
      1 for actual tokens and 0 for padding.  This is the same format as
      original BERT.  `long_seq_len` must be statically known.
    num_tokens_per_block: Positive integer number of long tokens to assign to
      each global token.  For pre-training on the original BERT data (which was
      also used for ETC pre-training), the dataset implied a value of about 27,
      but values like 16 or 32 would also be reasonable.
    local_radius: How many tokens to the left/right for input tokens to locally
      self-attend to.  For example, a value of 1 would allow each token to only
      attend to 1 token to the left and 1 token to the right of it.
    relative_pos_max_distance: Maximum distance to use for relative position
      representations.  All larger distances will be clipped to this value. Use
      0 to skip relative position representations entirely.
    use_hard_g2l_mask: If True, global tokens only attend to tokens of their
      corresponding block in the long input.  If False, global tokens attend to
      all non-padding long tokens.  False is the default setup.
    use_hard_l2g_mask: If True, long tokens only attend to the global token
      corresponding to their block.  If False, long tokens attend to all the
      non-padding global tokens.  False is the default setup.
    global_token_id: Integer id to use for global tokens.  The default is `1`,
      which was the value used during ETC pre-training.
    name: A name for the operation (optional).

  Returns:
    A tuple with the following 2 elements:
      side_inputs: A `GlobalLocalTransformerSideInputs` object containing all
        side input tensors.
      global_token_ids: <int32>[batch_size, global_seq_len] Tensor of global
        tokens ids suitable to pass into `EtcModel`.  All global tokens will
        use the same `global_token_id`, except for padding tokens.
  """
  if num_tokens_per_block <= 0:
    raise ValueError('`num_tokens_per_block` must be positive.')

  with tf.name_scope(name or 'make_fixed_block_side_inputs'):
    input_mask = tf.convert_to_tensor(input_mask)

    batch_size = tensor_utils.get_shape_list(input_mask)[0]
    long_seq_len = input_mask.shape.as_list()[1]
    if long_seq_len is None:
      raise ValueError('`long_seq_len` must be statically known.')

    global_seq_len = (long_seq_len + num_tokens_per_block -
                      1) // num_tokens_per_block

    # [batch_size, global_seq_len, num_tokens_per_block]
    blocked_input_mask = tensor_utils.split_into_blocks(
        input_mask, block_len=num_tokens_per_block, axis=-1)
    assert blocked_input_mask.shape.as_list()[1] == global_seq_len

    # [batch_size, global_seq_len]
    global_input_mask = tf.minimum(
        tf.reduce_max(blocked_input_mask, axis=-1), 1)

    # [long_seq_len]
    sentence_ids = tf.repeat(
        tf.range(global_seq_len, dtype=tf.int32),
        num_tokens_per_block)[:long_seq_len]

    # [batch_size, long_seq_len]
    sentence_ids = tf.broadcast_to(sentence_ids, [batch_size, long_seq_len])

    side_inputs = make_global_local_transformer_side_inputs_from_example_ids(
        long_example_ids=input_mask,
        global_example_ids=global_input_mask,
        sentence_ids=sentence_ids,
        local_radius=local_radius,
        relative_pos_max_distance=relative_pos_max_distance,
        use_hard_g2l_mask=use_hard_g2l_mask,
        use_hard_l2g_mask=use_hard_l2g_mask)
    global_token_ids = global_token_id * global_input_mask
    return side_inputs, global_token_ids


def add_side_input_features(
    model_config: modeling.EtcConfig,
    features: Mapping[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
  """Replaces raw input features with derived ETC side inputs.

  This function is meant to be called as part of a Dataset pipeline.

  Args:
    model_config: An `EtcConfig`.
    features: A dictionary of Tensor features, crucially including
      `long_breakpoints`, `global_breakpoints`, `sentence_ids`.

  Returns:
    A new `features` dictionary with global-local transformer side inputs.
  """
  features = dict(features)
  side_inputs = make_global_local_transformer_side_inputs(
      long_breakpoints=features['long_breakpoints'],
      global_breakpoints=features['global_breakpoints'],
      sentence_ids=features['sentence_ids'],
      local_radius=model_config.local_radius,
      relative_pos_max_distance=model_config.relative_pos_max_distance,
      use_hard_g2l_mask=model_config.use_hard_g2l_mask,
      use_hard_l2g_mask=model_config.use_hard_l2g_mask)
  features.update(side_inputs.to_dict(exclude_none_values=True))
  return features


def get_assignment_map_from_checkpoint(
    variables: Sequence[tf.Variable],
    ckpt_path: Text,
    variable_scope: Text = '',
    ckpt_variable_scope: Text = '') -> Tuple[Dict[Text, Text], List[Text]]:
  """Gets the mapping from checkpoint variable names to `variable` names.

  Computes the *intersection* of `variables` (under `variable_scope`) and
  checkpoint variables (under `ckpt_variable_scope`) and gets the name
  mapping from the latter to the former.

  Args:
    variables: The list of Tensorflow variables one aims to initialize.
    ckpt_path: Path to the checkpoint to load `variables` from.
    variable_scope: The scope of `variables` to initialize. `Variables` outside
      this scope will be ignored. If "", use all `variables`; otherwise it
      should end with '/'.
    ckpt_variable_scope: The scope of checkpoint variables to initialize from.
      Checkpoint variables outside this scope will be ignored. If "", use all
      `variables`; otherwise it should end with '/'.

  Returns:
    assignment_map: Mapping from checkpoint variable names to `variable`.
      Keys and values are matching variables under the `ckpt_variable_scope`
      and `variable_scope` (sub-)trees.
    initialized_variable_names: Names of `variables` that get matched to
      checkpoint variables.

  Raises:
    ValueError if
      (a) input scope name is not empty and doesn't end with "/"; or
      (b) names of `variables` doesn't end with ':0' (unlikely to happen).

  Example
    Input:
      variables: ["a/aa/aaa:0", "a/c/cc/ccc:0", "d/dd:0"]
      ckpt_variables: ["b/aa/aaa", "b/f"]
      variable_scope: "a/"
      ckpt_variable_scope: "b/"
    Output:
      assignment_map: {"b/aa/aaa": <tf.Variable "a/aa/aaa:0">}
      initialized_variable_names: ["a/aa/aaa:0"]
  """
  if variable_scope and not variable_scope.endswith('/'):
    raise ValueError('{} should end with "/".'.format(variable_scope))

  if ckpt_variable_scope and not ckpt_variable_scope.endswith('/'):
    raise ValueError('{} should end with "/".'.format(ckpt_variable_scope))

  variable_names_stripped = set()
  for var in variables:
    var_name = var.name

    # Ignores `variables` outside scope.
    # Note that all strings start with "".
    if not var_name.startswith(variable_scope):
      continue

    # Names of variables from Tensorflow API all have the suffix of ":0"
    # while those from checkpoint don't. Here we strip the suffix out.
    m = re.match('^(.*):\\d+$', var_name)
    if m is not None:
      var_name = m.group(1)
    else:
      raise ValueError(
          'Variable name does not end with ":0": {}'.format(var_name))

    # Strips the `variable_scope` prefix out.
    var_name_stripped = var_name[len(variable_scope):]
    if var_name_stripped:
      variable_names_stripped.add(var_name_stripped)

  var_name_to_variable = {var.name: var for var in variables}
  assignment_map = collections.OrderedDict()
  initialized_variable_names = []

  for ckpt_var_name, _ in tf.train.list_variables(ckpt_path):
    # Ignores checkpoint variables outside scope.
    # Note that all strings start with "".
    if not ckpt_var_name.startswith(ckpt_variable_scope):
      continue

    ckpt_var_name_stripped = ckpt_var_name[len(ckpt_variable_scope):]
    if ckpt_var_name_stripped not in variable_names_stripped:
      continue

    var_name = variable_scope + ckpt_var_name_stripped + ':0'

    assignment_map[ckpt_var_name] = var_name_to_variable[var_name]
    initialized_variable_names.append(var_name)

  return (assignment_map, initialized_variable_names)


def create_int_feature(values: Iterable[int]) -> tf.train.Feature:
  """Creates TensorFlow int features.

  Args:
    values: A sequence of integers.

  Returns:
    An entry of int tf.train.Feature.
  """
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


def create_float_feature(values: Iterable[float]) -> tf.train.Feature:
  """Creates TensorFlow float features.

  Args:
    values: A sequence of floats.

  Returns:
    An entry of float tf.train.Feature.
  """
  return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))


def create_bytes_feature(values: Iterable[Text]) -> tf.train.Feature:
  """Creates TensorFlow string features.

  Args:
    values: A sequence of unicode strings.

  Returns:
    An entry of byte tf.train.Feature.
  """
  values = [value.encode('utf-8') for value in values]

  feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=values))
  return feature


def get_feature(feature_name: Text,
                example: tf.train.Example) -> tf.train.Feature:
  """Gets Tensorflow feature by name.

  Args:
    feature_name: The name of the feature.
    example: A Tensorflow example.

  Returns:
    The Tensorflow feature with the given feature name in the example.

  Raises:
    ValueError: If the given feature name is not in the Tensorflow example.
  """
  if feature_name in example.features.feature:
    return example.features.feature[feature_name]
  else:
    raise ValueError('Feature name {} is not in the example {}'.format(
        feature_name, example))


def get_repeated_values(
    feature_name: Text,
    example: tf.train.Example) -> MutableSequence[Union[bytes, float, int]]:
  """Gets the underlying repeated values of a feature by feature name.

  The return type depends on which oneof `kind` is populated for the feature.
  Whichever one is populated is returned.

  Args:
    feature_name: The name of the feature.
    example: A Tensorflow example.

  Returns:
    The underlying repeated values for the given feature name in the example.
    Modifying these repeated values will modify the example.

  Raises:
    ValueError: If the given feature name is not in the Tensorflow example or
      none of the oneof `kind` fields is populated.
  """
  feature = get_feature(feature_name, example)
  which_oneof = feature.WhichOneof('kind')
  if which_oneof is None:
    raise ValueError(
        'No field populated in oneof `kind` for feature name {} in example '
        '{}'.format(feature_name, example))
  return getattr(feature, which_oneof).value
