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

"""TFlite utils."""
# Lint as: python3
import os

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper  # pylint: disable=g-direct-tensorflow-import
from tensorflow.lite.tools import flatbuffer_utils  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tools import saved_model_utils  # pylint: disable=g-direct-tensorflow-import


def export_tflite_from_saved_model(saved_model_dir):
  """Convert SavedModel to TFlite graph."""
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
  converter.target_spec.supported_ops = supported_ops
  converter.allow_custom_ops = True
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()
  export_path = os.path.join(saved_model_dir, "model.tflite")
  with tf.io.gfile.GFile(export_path, "wb") as f:
    f.write(tflite_model)


def tflite_graph_rewrite(tflite_model_path,
                         saved_model_dir,
                         custom_op_registerers=None):
  """Rewrite TFLite graph to make inputs/outputs tensor name consistent.

  TF users do not have good control over outputs tensor names from
  get_concrete_function(), to maintain backward compatibility the tensor name
  in TFLite graph need to be meaningful and properly set. This function looks up
  the meaningful names from SavedModel signature meta data and rewrite it into
  TFlite graph.

  Arguments:
    tflite_model_path: The path to the exported TFLite graph, which will be
      overwrite after rewrite.
    saved_model_dir: Directory that stores SavedModelthat used for TFLite
    custom_op_registerers: list with custom op registers
      conversion.
  """
  # Find map from signature inputs/outputs name to tensor name in SavedModel.
  meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir,
                                                        "serve")
  signature_def = meta_graph_def.signature_def
  tensor_name_to_signature_name = {}
  for key, value in signature_def["serving_default"].inputs.items():
    tensor_name_to_signature_name[value.name] = key
  for key, value in signature_def["serving_default"].outputs.items():
    tensor_name_to_signature_name[value.name] = key

  # Find map from TFlite inputs/outputs index to tensor name in TFLite graph.
  with tf.io.gfile.GFile(tflite_model_path, "rb") as f:
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_content=f.read(),
        custom_op_registerers=custom_op_registerers)
  tflite_input_index_to_tensor_name = {}
  tflite_output_index_to_tensor_name = {}
  for idx, input_detail in enumerate(interpreter.get_input_details()):
    tflite_input_index_to_tensor_name[idx] = input_detail["name"]
  for idx, output_detail in enumerate(interpreter.get_output_details()):
    tflite_output_index_to_tensor_name[idx] = output_detail["name"]

  # Rewrite TFLite graph inputs/outputs name.
  mutable_fb = flatbuffer_utils.read_model_with_mutable_tensors(
      tflite_model_path)
  subgraph = mutable_fb.subgraphs[0]
  for input_idx, input_tensor_name in tflite_input_index_to_tensor_name.items():
    subgraph.tensors[subgraph.inputs[
        input_idx]].name = tensor_name_to_signature_name[input_tensor_name]
  for output_idx, output_tensor_name in tflite_output_index_to_tensor_name.items(
  ):
    subgraph.tensors[subgraph.outputs[
        output_idx]].name = tensor_name_to_signature_name[output_tensor_name]
  flatbuffer_utils.write_model(mutable_fb, tflite_model_path)


def get_tensor_name_to_tflite_input_index(details):
  tensor_name_to_tflite_input_index = {}
  for i, detail in enumerate(details):
    tensor_name_to_tflite_input_index[detail["name"]] = i
  return tensor_name_to_tflite_input_index


def run_stream_inference_classification_tflite(interpreter,
                                               inp_audio,
                                               input_states,
                                               stride,
                                               input_tensot_name="input_0",
                                               output_tensor_name="output_0"):
  """Runs streaming inference classification with tflite (external state).

  It is useful for testing streaming classification
  Args:
    interpreter: tf lite interpreter in streaming mode
    inp_audio: input audio data
    input_states: input states
    stride: total stride in the model
    input_tensot_name: input tensor name
    output_tensor_name: output tensor name
  Returns:
    last output
  """

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  inp_tensor_name_to_tflite_inp_index = get_tensor_name_to_tflite_input_index(
      input_details)
  out_tensor_name_to_tflite_out_index = get_tensor_name_to_tflite_input_index(
      output_details)

  start = 0
  end = stride
  while end <= inp_audio.shape[1]:
    # get new audio chunk
    stream_update = inp_audio[:, start:end]

    # update indexes of streamed updates
    start = end
    end += stride

    # classification result of a current frame
    interpreter.set_tensor(
        input_details[inp_tensor_name_to_tflite_inp_index[input_tensot_name]]
        ["index"], stream_update.astype(np.float32))

    # set states
    for key in input_states.keys():
      interpreter.set_tensor(
          input_details[inp_tensor_name_to_tflite_inp_index[key]]["index"],
          input_states[key].astype(np.float32))

    interpreter.invoke()

    # classification output
    tflite_output = interpreter.get_tensor(output_details[
        out_tensor_name_to_tflite_out_index[output_tensor_name]]["index"])

    # update states
    for key in input_states.keys():
      input_states[key] = interpreter.get_tensor(
          output_details[out_tensor_name_to_tflite_out_index[key]]["index"])

  # return the last classification results
  return tflite_output
