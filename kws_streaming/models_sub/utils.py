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

"""Utility functions."""

import tensorflow as tf


def method_exists(class_type, method_name):
  return hasattr(class_type, method_name) and callable(
      getattr(class_type, method_name))


def saved_model_to_tflite(saved_model_path,
                          tflite_model_path=None,
                          optimizations=None,
                          inference_type=tf.float32,
                          experimental_new_quantizer=False,
                          representative_dataset=None):
  """Converts saved_model to tflite.

  Args:
    saved_model_path: path to saved_model folder
    tflite_model_path: pathe to generated tflite module
    optimizations: tflite optimizations
    inference_type: type of the inference: tf.int8, tf.float32
    experimental_new_quantizer: apply new quantizer
    representative_dataset: function generating representative dataset which are
      used to calibrate post training quantization

  Returns:
    quantized layer
  """
  # optimizations = [tf.lite.Optimize.DEFAULT]

  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

  if optimizations:
    converter.optimizations = optimizations
  if inference_type:
    converter.inference_type = inference_type  # tf.int8

  converter.experimental_new_quantizer = experimental_new_quantizer
  if representative_dataset is not None:
    converter.representative_dataset = representative_dataset()
  quantized_model_tflite = converter.convert()

  if tflite_model_path is not None:
    with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
      f.write(quantized_model_tflite)
  return quantized_model_tflite


def run_stream_inference_classification(model, inp_audio):
  """Runs streaming inference classification with tf (with internal state).

  It is useful for testing streaming classification
  Args:
    model: keras model based on BaseModel
    inp_audio: input audio data
  Returns:
    last output
  """

  states = model.states()

  stream_step_size = model.stride
  start = 0
  end = stream_step_size
  while end <= inp_audio.shape[1]:
    # get new audio chunk
    stream_update = inp_audio[:, start:end]
    # update indexes of streamed updates
    start = end
    end += stream_step_size

    # classification result of a current frame
    outputs = model.stream_inference(stream_update, states)
    stream_output_prediction = outputs['output_0']
    for key in states.keys():
      states[key] = outputs[key]

  # return the last classification results
  return stream_output_prediction
