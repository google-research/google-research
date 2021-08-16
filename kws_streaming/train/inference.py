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

"""Inference utility functions."""
import numpy as np


def run_stream_inference(flags, model_stream, inp_audio):
  """Runs streaming inference.

  It is useful for speech filtering/enhancement
  Args:
    flags: model and data settings
    model_stream: tf model in streaming mode
    inp_audio: input audio data
  Returns:
    output sequence
  """

  step = flags.data_shape[0]
  start = 0
  end = step
  stream_out = None

  while end <= inp_audio.shape[1]:
    stream_update = inp_audio[:, start:end]
    stream_output_sample = model_stream.predict(stream_update)

    if stream_out is None:
      stream_out = stream_output_sample
    else:
      stream_out = np.concatenate((stream_out, stream_output_sample), axis=1)

    start = end
    end = start + step
  return stream_out


def run_stream_inference_classification(flags, model_stream, inp_audio):
  """Runs streaming inference classification with tf (with internal state).

  It is useful for testing streaming classification
  Args:
    flags: model and data settings
    model_stream: tf model in streaming mode
    inp_audio: input audio data
  Returns:
    last output
  """

  stream_step_size = flags.data_shape[0]
  start = 0
  end = stream_step_size
  while end <= inp_audio.shape[1]:
    # get overlapped audio sequence
    stream_update = inp_audio[:, start:end]

    # update indexes of streamed updates
    start = end
    end += stream_step_size

    # classification result of a current frame
    stream_output_prediction = model_stream.predict(stream_update)

  return stream_output_prediction


def run_stream_inference_classification_tflite(flags, interpreter, inp_audio,
                                               input_states):
  """Runs streaming inference classification with tflite (external state).

  It is useful for testing streaming classification
  Args:
    flags: model and data settings
    interpreter: tf lite interpreter in streaming mode
    inp_audio: input audio data
    input_states: input states
  Returns:
    last output
  """

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  if len(input_details) != len(output_details):
    raise ValueError('Number of inputs should be equal to the number of outputs'
                     'for the case of streaming model with external state')

  stream_step_size = flags.data_shape[0]
  start = 0
  end = stream_step_size
  while end <= inp_audio.shape[1]:
    stream_update = inp_audio[:, start:end]
    stream_update = stream_update.astype(np.float32)

    # update indexes of streamed updates
    start = end
    end += stream_step_size

    # set input audio data (by default input data at index 0)
    interpreter.set_tensor(input_details[0]['index'], stream_update)

    # set input states (index 1...)
    for s in range(1, len(input_details)):
      interpreter.set_tensor(input_details[s]['index'], input_states[s])

    # run inference
    interpreter.invoke()

    # get output: classification
    out_tflite = interpreter.get_tensor(output_details[0]['index'])

    # get output states and set it back to input states
    # which will be fed in the next inference cycle
    for s in range(1, len(input_details)):
      # The function `get_tensor()` returns a copy of the tensor data.
      # Use `tensor()` in order to get a pointer to the tensor.
      input_states[s] = interpreter.get_tensor(output_details[s]['index'])

  return out_tflite


def run_stream_inference_tflite(flags,
                                interpreter,
                                inp_audio,
                                input_states,
                                concat=True):
  """Runs streaming inference with tflite (external state).

  It is useful for testing streaming filtering
  Args:
    flags: model and data settings
    interpreter: tf lite interpreter in streaming mode
    inp_audio: input audio data
    input_states: input states
    concat: if True, it will concatenate outputs in dim 1, otherwise append them
  Returns:
    output sequence
  """

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  if len(input_details) != len(output_details):
    raise ValueError('Number of inputs should be equal to the number of outputs'
                     'for the case of streaming model with external state')

  step = flags.data_shape[0]
  start = 0
  end = step
  stream_out_tflite_external_st = np.array([])

  while end <= inp_audio.shape[1]:
    stream_update = inp_audio[:, start:end]
    stream_update = stream_update.astype(np.float32)

    # set input audio data (by default input data at index 0)
    interpreter.set_tensor(input_details[0]['index'], stream_update)

    # set input states (index 1...)
    for s in range(1, len(input_details)):
      interpreter.set_tensor(input_details[s]['index'], input_states[s])

    # run inference
    interpreter.invoke()

    # get output
    out_tflite = interpreter.get_tensor(output_details[0]['index'])

    # get output states and set it back to input states
    # which will be fed in the next inference cycle
    for s in range(1, len(input_details)):
      # The function `get_tensor()` returns a copy of the tensor data.
      # Use `tensor()` in order to get a pointer to the tensor.
      input_states[s] = interpreter.get_tensor(output_details[s]['index'])

    if concat:
      if stream_out_tflite_external_st.size == 0:
        stream_out_tflite_external_st = out_tflite
      else:
        stream_out_tflite_external_st = np.concatenate(
            (stream_out_tflite_external_st, out_tflite), axis=1)
    else:
      stream_out_tflite_external_st = np.append(stream_out_tflite_external_st,
                                                out_tflite)

    # update indexes of streamed updates
    start = end
    end = start + step

  return stream_out_tflite_external_st
