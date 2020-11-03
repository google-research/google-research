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

# Lint as: python3
"""Utility functions for operations on Model."""

import ast
import os.path
from typing import Sequence

from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.models import model_flags
from kws_streaming.models import model_params
from kws_streaming.models import models as kws_models
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import functional
# pylint: enable=g-direct-tensorflow-import


def conv2d_bn(x,
              filters,
              kernel_size,
              padding='same',
              strides=(1, 1),
              activation='relu',
              use_bias=False,
              scale=False):
  """Utility function to apply conv + BN.

  Arguments:
    x: input tensor.
    filters: filters in `Conv2D`.
    kernel_size: size of convolution kernel.
    padding: padding mode in `Conv2D`.
    strides: strides in `Conv2D`.
    activation: activation function applied in the end.
    use_bias: use bias for convolution.
    scale: scale batch normalization.

  Returns:
    Output tensor after applying `Conv2D` and `BatchNormalization`.
  """

  x = tf.keras.layers.Conv2D(
      filters, kernel_size,
      strides=strides,
      padding=padding,
      use_bias=use_bias)(x)
  x = tf.keras.layers.BatchNormalization(scale=scale)(x)
  x = tf.keras.layers.Activation(activation)(x)
  return x


def save_model_summary(model, path, file_name='model_summary.txt'):
  """Saves model topology/summary in text format.

  Args:
    model: Keras model
    path: path where to store model summary
    file_name: model summary file name
  """
  with open(os.path.join(path, file_name), 'wt') as fd:
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))  # pylint: disable=unnecessary-lambda
    model_summary = '\n'.join(stringlist)
    fd.write(model_summary)


def _set_mode(model, mode):
  """Set model's inference type and disable training."""
  for i in range(len(model.layers)):
    config = model.layers[i].get_config()
    # for every layer set mode, if it has it
    if 'mode' in config:
      model.layers[i].mode = mode
      # with any mode of inference - training is False
    if 'training' in config:
      model.layers[i].training = False
    if mode == modes.Modes.NON_STREAM_INFERENCE:
      if 'unroll' in config:
        model.layers[i].unroll = True
  return model


def _get_input_output_states(model):
  """Get input/output states of model with external states."""
  input_states = []
  output_states = []
  for i in range(len(model.layers)):
    config = model.layers[i].get_config()
    # input output states exist only in layers with property 'mode'
    if 'mode' in config:
      input_states.append(model.layers[i].get_input_state())
      output_states.append(model.layers[i].get_output_state())
  return input_states, output_states


def _clone_model(model, input_tensors):
  """Clone model with configs, except of weights."""
  new_input_layers = {}  # Cache for created layers.
  # pylint: disable=protected-access
  if input_tensors is not None:
    # Make sure that all input tensors come from a Keras layer.
    input_tensors = tf.nest.flatten(input_tensors)
    for i, input_tensor in enumerate(input_tensors):
      if not tf.keras.backend.is_keras_tensor(input_tensor):
        raise ValueError('Expected keras tensor but get', input_tensor)
      original_input_layer = model._input_layers[i]
      newly_created_input_layer = input_tensor._keras_history.layer
      new_input_layers[original_input_layer] = newly_created_input_layer

  model_config, created_layers = models._clone_layers_and_model_config(
      model, new_input_layers, models._clone_layer)
  # pylint: enable=protected-access

  # Reconstruct model from the config, using the cloned layers.
  input_tensors, output_tensors, created_layers = (
      functional.reconstruct_from_config(
          model_config, created_layers=created_layers))

  new_model = tf.keras.Model(input_tensors, output_tensors, name=model.name)
  return new_model


def _copy_weights(new_model, model):
  """Copy weights of trained model to an inference one."""

  def _same_weights(weight, new_weight):
    # Check that weights are the same
    # Note that states should be marked as non trainable
    return (weight.trainable == new_weight.trainable and
            weight.shape == new_weight.shape and
            weight.name[weight.name.rfind('/'):None] ==
            new_weight.name[new_weight.name.rfind('/'):None])

  if len(new_model.layers) != len(model.layers):
    raise ValueError(
        'number of layers in new_model: %d != to layers number in model: %d ' %
        (len(new_model.layers), len(model.layers)))

  for i in range(len(model.layers)):
    layer = model.layers[i]
    new_layer = new_model.layers[i]

    # if number of weights in the layers are the same
    # then we can set weights directly
    if len(layer.get_weights()) == len(new_layer.get_weights()):
      new_layer.set_weights(layer.get_weights())
    elif layer.weights:
      k = 0  # index pointing to weights in the copied model
      new_weights = []
      # iterate over weights in the new_model
      # and prepare a new_weights list which will
      # contain weights from model and weight states from new model
      for k_new in range(len(new_layer.get_weights())):
        new_weight = new_layer.weights[k_new]
        new_weight_values = new_layer.get_weights()[k_new]
        same_weights = True

        # if there are weights which are not copied yet
        if k < len(layer.get_weights()):
          weight = layer.weights[k]
          weight_values = layer.get_weights()[k]
          if (weight.shape != weight_values.shape or
              new_weight.shape != new_weight_values.shape):
            raise ValueError('weights are not listed in order')

          # if there are weights available for copying and they are the same
          if _same_weights(weight, new_weight):
            new_weights.append(weight_values)
            k = k + 1  # go to next weight in model
          else:
            same_weights = False  # weights are different
        else:
          same_weights = False  # all weights are copied, remaining is different

        if not same_weights:
          # weight with index k_new is missing in model,
          # so we will keep iterating over k_new until find similar weights
          new_weights.append(new_weight_values)

      # check that all weights from model are copied to a new_model
      if k != len(layer.get_weights()):
        raise ValueError(
            'trained model has: %d weights, but only %d were copied' %
            (len(layer.get_weights()), k))

      # now they should have the same number of weights with matched sizes
      # so we can set weights directly
      new_layer.set_weights(new_weights)
  return new_model


def _flatten_nested_sequence(sequence):
  """Returns a flattened list of sequence's elements."""
  if not isinstance(sequence, Sequence):
    return [sequence]
  result = []
  for value in sequence:
    result.extend(_flatten_nested_sequence(value))
  return result


def _get_state_shapes(model_states):
  """Converts a nested list of states in to a flat list of their shapes."""
  return [state.shape for state in _flatten_nested_sequence(model_states)]


def convert_to_inference_model(model, input_tensors, mode):
  """Convert functional `Model` instance to a streaming inference.

  It will create a new model with new inputs: input_tensors.
  All weights will be copied. Internal states for streaming mode will be created
  Only functional Keras model is supported!

  Args:
      model: Instance of `Model`.
      input_tensors: list of input tensors to build the model upon.
      mode: is defined by modes.Modes

  Returns:
      An instance of streaming inference `Model` reproducing the behavior
      of the original model, on top of new inputs tensors,
      using copied weights.

  Raises:
      ValueError: in case of invalid `model` argument value or input_tensors
  """

  # scope is introduced for simplifiyng access to weights by names
  scope_name = 'streaming'
  with tf.name_scope(scope_name):
    if not isinstance(model, tf.keras.Model):
      raise ValueError(
          'Expected `model` argument to be a `Model` instance, got ', model)
    if isinstance(model, tf.keras.Sequential):
      raise ValueError(
          'Expected `model` argument '
          'to be a functional `Model` instance, '
          'got a `Sequential` instance instead:', model)
    # pylint: disable=protected-access
    if not model._is_graph_network:
      raise ValueError('Expected `model` argument '
                       'to be a functional `Model` instance, '
                       'but got a subclass model instead.')
    # pylint: enable=protected-access
    model = _set_mode(model, mode)
    new_model = _clone_model(model, input_tensors)

  if mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
    return _copy_weights(new_model, model)
  elif mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
    input_states, output_states = _get_input_output_states(new_model)
    all_inputs = new_model.inputs + input_states
    all_outputs = new_model.outputs + output_states
    new_streaming_model = tf.keras.Model(all_inputs, all_outputs)
    new_streaming_model.input_shapes = _get_state_shapes(all_inputs)
    new_streaming_model.output_shapes = _get_state_shapes(all_outputs)

    # inference streaming model with external states
    # has the same number of weights with
    # non streaming model so we can use set_weights directly
    new_streaming_model.set_weights(model.get_weights())
    return new_streaming_model
  elif mode == modes.Modes.NON_STREAM_INFERENCE:
    new_model.set_weights(model.get_weights())
    return new_model
  else:
    raise ValueError('non supported mode ', mode)


def to_streaming_inference(model_non_stream, flags, mode):
  """Convert non streaming trained model to inference modes.

  Args:
    model_non_stream: trained Keras model non streamable
    flags: settings with global data and model properties
    mode: it supports Non streaming inference, Streaming inference with internal
      states, Streaming inference with external states

  Returns:
    Keras inference model of inference_type
  """
  tf.keras.backend.set_learning_phase(0)
  input_data_shape = modes.get_input_data_shape(flags, mode)
  input_tensors = [
      tf.keras.layers.Input(
          shape=input_data_shape, batch_size=1, name='input_audio')
  ]
  model_inference = convert_to_inference_model(model_non_stream, input_tensors,
                                               mode)
  return model_inference


def model_to_tflite(sess,
                    model_non_stream,
                    flags,
                    mode=modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE,
                    save_model_path=None,
                    optimizations=None):
  """Convert non streaming model to tflite inference model.

  In this case inference graph will be stateless.
  But model can be streaming stateful with external state or
  non streaming statless (depending on input arg mode)

  Args:
    sess: tf session
    model_non_stream: Keras non streamable model
    flags: settings with global data and model properties
    mode: inference mode it can be streaming with external state or non
      streaming
    save_model_path: path to save intermediate model summary
    optimizations: list of optimization options

  Returns:
    tflite model
  """
  if mode not in (modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE,
                  modes.Modes.NON_STREAM_INFERENCE):
    raise ValueError('mode %s is not supported ' % mode)
  # convert non streaming Keras model to
  # Keras inference model (non streaming or streaming)
  model_stateless_stream = to_streaming_inference(model_non_stream, flags, mode)

  if save_model_path:
    save_model_summary(model_stateless_stream, save_model_path)

  # convert Keras inference model to tflite inference model
  converter = tf1.lite.TFLiteConverter.from_session(
      sess, model_stateless_stream.inputs, model_stateless_stream.outputs)
  converter.inference_type = tf1.lite.constants.FLOAT

  # this will enable audio_spectrogram and mfcc in TFLite
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
  converter.allow_custom_ops = True
  if optimizations:
    converter.optimizations = optimizations
  tflite_model = converter.convert()
  return tflite_model


# in below code .from_tensor() instead of tf.TensorSpec is adding TensorSpec
# which is not recognized here, so making TensorSpec visible too
TensorSpec = tf.TensorSpec


def model_to_saved(model_non_stream,
                   flags,
                   save_model_path,
                   mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE):
  """Convert Keras model to SavedModel.

  Depending on mode:
    1 Converted inference graph and model will be streaming statefull.
    2 Converted inference graph and model will be non streaming stateless.

  Args:
    model_non_stream: Keras non streamable model
    flags: settings with global data and model properties
    save_model_path: path where saved model representation with be stored
    mode: inference mode it can be streaming with external state or non
      streaming
  """

  if mode not in (modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
                  modes.Modes.NON_STREAM_INFERENCE):
    raise ValueError('mode %s is not supported ' % mode)

  if mode == modes.Modes.NON_STREAM_INFERENCE:
    model = model_non_stream
  else:
    # convert non streaming Keras model to Keras streaming model, internal state
    model = to_streaming_inference(model_non_stream, flags, mode)

  save_model_summary(model, save_model_path)
  model.save(save_model_path, include_optimizer=False, save_format='tf')


def parse(text):
  """Parse model parameters.

  Args:
    text: string with layer parameters: '128,128' or "'relu','relu'".

  Returns:
    list of parsed parameters
  """
  if not text:
    return []
  res = ast.literal_eval(text)
  if isinstance(res, tuple):
    return res
  else:
    return [res]


def next_power_of_two(x):
  """Calculates the smallest enclosing power of two for an input.

  Args:
    x: Positive float or integer number.

  Returns:
    Next largest power of two integer.
  """
  return 1 if x == 0 else 2**(int(x) - 1).bit_length()


def get_model_with_default_params(model_name, mode=None):
  """Creates a model with the params specified in HOTWORD_MODEL_PARAMS."""
  if model_name not in model_params.HOTWORD_MODEL_PARAMS:
    raise KeyError(
        "Expected 'model_name' to be one of "
        f"{model_params.HOTWORD_MODEL_PARAMS.keys} but got '{model_name}'.")
  params = model_params.HOTWORD_MODEL_PARAMS[model_name]
  params = model_flags.update_flags(params)
  model = kws_models.MODELS[params.model_name](params)
  if mode is not None:
    model = to_streaming_inference(model, flags=params, mode=mode)
  return model
