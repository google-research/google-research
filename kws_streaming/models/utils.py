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

"""Utility functions for operations on Model."""
import os.path
import tempfile
from typing import Sequence, Optional, List
from keras import models as models_utils
from keras.engine import functional
import numpy as np
from kws_streaming.layers import modes
from kws_streaming.layers import quantize
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.models import model_flags
from kws_streaming.models import model_params
from kws_streaming.models import model_utils
from kws_streaming.models import models as kws_models


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

  def _recursive_set_layer_mode(layer, mode):
    if isinstance(layer, tf.keras.layers.Wrapper):
      _recursive_set_layer_mode(layer.layer, mode)

    config = layer.get_config()
    # for every layer set mode, if it has it
    if 'mode' in config:
      layer.mode = mode
      # with any mode of inference - training is False
    if 'training' in config:
      layer.training = False
    if mode == modes.Modes.NON_STREAM_INFERENCE:
      if 'unroll' in config:
        layer.unroll = True

  for layer in model.layers:
    _recursive_set_layer_mode(layer, mode)
  return model


def _get_input_output_states(model):
  """Get input/output states of model with external states."""
  input_states = []
  output_states = []
  for i in range(len(model.layers)):
    config = model.layers[i].get_config()
    # input output states exist only in layers with property 'mode'
    if 'mode' in config:
      input_state = model.layers[i].get_input_state()
      if input_state not in ([], [None]):
        input_states.append(model.layers[i].get_input_state())
      output_state = model.layers[i].get_output_state()
      if output_state not in ([], [None]):
        output_states.append(output_state)
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

  model_config, created_layers = models_utils._clone_layers_and_model_config(
      model, new_input_layers, models_utils._clone_layer)
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


def get_stride(model):
  """Computes total stride of a model."""
  stride = 1
  for i in range(len(model.layers)):
    layer = model.layers[i]
    if hasattr(layer, 'stride'):
      stride = stride * layer.stride()
  return stride


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

  # get input data type and use it for input streaming type
  if isinstance(model_non_stream.input, (tuple, list)):
    dtype = model_non_stream.input[0].dtype
  else:
    dtype = model_non_stream.input.dtype

  input_tensors = [
      tf.keras.layers.Input(
          shape=input_data_shape, batch_size=1, dtype=dtype, name='input_audio')
  ]

  if (isinstance(model_non_stream.input, (tuple, list)) and
      len(model_non_stream.input) > 1):
    if len(model_non_stream.input) > 2:
      raise ValueError(
          'Maximum number of inputs supported is 2 (input_audio and '
          'cond_features), but got %d inputs' % len(model_non_stream.input))
    input_tensors.append(
        tf.keras.layers.Input(
            shape=flags.cond_shape,
            batch_size=1,
            dtype=model_non_stream.input[1].dtype,
            name='cond_features'))

  quantize_stream_scope = quantize.quantize_scope()
  with quantize_stream_scope:
    model_inference = convert_to_inference_model(model_non_stream,
                                                 input_tensors, mode)
  return model_inference


def model_to_tflite(
    sess,
    model_non_stream,
    flags,
    mode=modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE,
    save_model_path=None,
    optimizations=None,
    use_fp16=False,
    inference_type=tf1.lite.constants.FLOAT,
    experimental_new_quantizer=True,
    representative_dataset=None,
    inference_input_type=tf.float32,
    inference_output_type=tf.float32,
    supported_ops_override = None,
    allow_custom_ops = True):
  """Convert non streaming model to tflite inference model.

  If mode==modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE then inference graph
  will be stateless: all states will be managed outside of the model and
  will be passed to the model as additional inputs/outputs.
  If mode==modes.Modes.STREAM_INTERNAL_STATE_INFERENCE then inference graph
  will be stateful: all states will be part of the model - so model size
  can increase. Latest version of TFLite converter supports it, so
  conversion has to be done in eager mode.

  Args:
    sess: tf session, if None then eager mode is used
    model_non_stream: Keras non streamable model
    flags: settings with global data and model properties
    mode: inference mode it can be streaming with external state or non
      streaming
    save_model_path: path to save intermediate model summary
    optimizations: list of optimization options
    use_fp16: uses float16 post-training quantization in place for float.
      Only effective when `optimizations` is not None.
    inference_type: inference type, can be float or int8
    experimental_new_quantizer: enable new quantizer
    representative_dataset: function generating representative data sets
      for calibration post training quantizer
    inference_input_type: it can be used to quantize input data e.g. tf.int8
    inference_output_type: it can be used to quantize output data e.g. tf.int8
    supported_ops_override: explicitly set supported ops in converter.
    allow_custom_ops: explicitly set custom op usage.

  Returns:
    tflite model
  """
  if sess and mode not in (modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE,
                           modes.Modes.NON_STREAM_INFERENCE):
    raise ValueError('mode %s is not supported in session mode ' % mode)

  if not sess and mode not in (modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
                               modes.Modes.NON_STREAM_INFERENCE):
    raise ValueError('mode %s is not supported in eager mode ' % mode)

  # convert non streaming Keras model to
  # Keras inference model (non streaming or streaming)
  model_stream = to_streaming_inference(model_non_stream, flags, mode)

  if save_model_path:
    save_model_summary(model_stream, save_model_path)

  # Identify custom objects.
  with quantize.quantize_scope():
    if sess:
      # convert Keras inference model to tflite inference model
      converter = tf1.lite.TFLiteConverter.from_session(
          sess, model_stream.inputs, model_stream.outputs)
    else:
      if not save_model_path:
        save_model_path = tempfile.mkdtemp()
      tf.saved_model.save(model_stream, save_model_path)
      converter = tf.lite.TFLiteConverter.from_saved_model(save_model_path)

  converter.inference_type = inference_type
  converter.experimental_new_quantizer = experimental_new_quantizer
  converter.experimental_enable_resource_variables = True
  if representative_dataset is not None:
    converter.representative_dataset = representative_dataset
  if not supported_ops_override:
    # this will enable audio_spectrogram and mfcc in TFLite
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
    ]
  else:
    converter.target_spec.supported_ops = supported_ops_override
  converter.allow_custom_ops = allow_custom_ops
  converter.inference_input_type = inference_input_type
  converter.inference_output_type = inference_output_type
  if optimizations:
    converter.optimizations = optimizations
    if use_fp16:
      converter.target_spec.supported_types = [tf.float16]
      # pylint: disable=protected-access
      converter.target_spec._experimental_supported_accumulation_type = (
          tf.dtypes.float16)

  if hasattr(flags, 'quantize') and hasattr(flags, 'use_quantize_nbit'):
    if flags.quantize and flags.use_quantize_nbit:
      # pylint: disable=protected-access
      converter._experimental_low_bit_qat = True
      # pylint: enable=protected-access

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
  data_stride = params.data_stride
  params = model_flags.update_flags(params)
  params.data_stride = data_stride
  model = kws_models.MODELS[params.model_name](params)
  model.summary()
  if mode is not None:
    model = to_streaming_inference(model, flags=params, mode=mode)
  return model


def traverse_graph(prev_layer, layers):
  """Traverse keras sequential graph."""
  for layer in layers:
    if isinstance(layer, (tf.keras.Sequential, tf.keras.Model)):
      prev_layer = traverse_graph(prev_layer, layer.layers)
    else:
      prev_layer = layer(prev_layer)
  return prev_layer


def sequential_to_functional(model):
  """Converts keras sequential model to functional one."""
  input_layer = tf.keras.Input(
      batch_input_shape=model.layers[0].input_shape[0])
  prev_layer = input_layer
  prev_layer = traverse_graph(prev_layer, model.layers[1:])
  func_model = tf.keras.Model([input_layer], [prev_layer])
  return func_model


def ds_tc_resnet_model_params(use_tf_fft=False):
  """Generate parameters for ds_tc_resnet model."""

  # model parameters
  model_name = 'ds_tc_resnet'
  params = model_params.HOTWORD_MODEL_PARAMS[model_name]
  params.causal_data_frame_padding = 1  # causal padding on DataFrame
  params.clip_duration_ms = 160
  params.use_tf_fft = use_tf_fft
  params.mel_non_zero_only = not use_tf_fft
  params.feature_type = 'mfcc_tf'
  params.window_size_ms = 5.0
  params.window_stride_ms = 2.0
  params.wanted_words = 'a,b,c'
  params.ds_padding = "'causal','causal','causal','causal'"
  params.ds_filters = '4,4,4,2'
  params.ds_repeat = '1,1,1,1'
  params.ds_residual = '0,1,1,1'  # no residuals on strided layers
  params.ds_kernel_size = '3,3,3,1'
  params.ds_dilation = '1,1,1,1'
  params.ds_stride = '2,1,1,1'  # streaming conv with stride
  params.ds_pool = '1,2,1,1'  # streaming conv with pool
  params.ds_filter_separable = '1,1,1,1'

  # convert ms to samples and compute labels count
  params = model_flags.update_flags(params)

  # compute total stride
  pools = model_utils.parse(params.ds_pool)
  strides = model_utils.parse(params.ds_stride)
  time_stride = [1]
  for pool in pools:
    if pool > 1:
      time_stride.append(pool)
  for stride in strides:
    if stride > 1:
      time_stride.append(stride)
  total_stride = np.prod(time_stride)

  # override input data shape for streaming model with stride/pool
  params.data_stride = total_stride
  params.data_shape = (total_stride * params.window_stride_samples,)

  # set desired number of frames in model
  frames_number = 16
  frames_per_call = total_stride
  frames_number = (frames_number // frames_per_call) * frames_per_call
  # number of input audio samples required to produce one output frame
  framing_stride = max(
      params.window_stride_samples,
      max(0, params.window_size_samples -
          params.window_stride_samples))
  signal_size = framing_stride * frames_number

  # desired number of samples in the input data to train non streaming model
  params.desired_samples = signal_size
  params.batch_size = 1
  return params


def saved_model_to_tflite(saved_model_path,
                          optimizations=None,
                          inference_type=tf1.lite.constants.FLOAT,
                          experimental_new_quantizer=True,
                          representative_dataset=None,
                          inference_input_type=tf.float32,
                          inference_output_type=tf.float32,
                          use_quantize_nbit=0):
  """Convert saved_model to tflite.

  Args:
    saved_model_path: path to saved_model
    optimizations: list of optimization options
    inference_type: inference type, can be float or int8
    experimental_new_quantizer: enable new quantizer
    representative_dataset: function generating representative data sets
      for calibation post training quantizer
    inference_input_type: it can be used to quantize input data e.g. tf.int8
    inference_output_type: it can be used to quantize output data e.g. tf.int8
    use_quantize_nbit: adds experimental flag for default_n_bit precision.

  Returns:
    tflite model
  """

  # Identify custom objects.
  with quantize.quantize_scope():
    converter = tf.compat.v2.lite.TFLiteConverter.from_saved_model(
        saved_model_path)

  converter.inference_type = inference_type
  converter.experimental_new_quantizer = experimental_new_quantizer
  converter.experimental_enable_resource_variables = True
  converter.experimental_new_converter = True
  if representative_dataset is not None:
    converter.representative_dataset = representative_dataset

  # this will enable audio_spectrogram and mfcc in TFLite
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
  ]
  converter.allow_custom_ops = True

  converter.inference_input_type = inference_input_type
  converter.inference_output_type = inference_output_type
  if optimizations:
    converter.optimizations = optimizations
  if use_quantize_nbit:
    # pylint: disable=protected-access
    converter._experimental_low_bit_qat = True
    # pylint: enable=protected-access
  tflite_model = converter.convert()
  return tflite_model
