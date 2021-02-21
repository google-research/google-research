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

"""Tests for ConvModel."""
import os
import numpy as np
import tensorflow as tf

from kws_streaming.models_sub import conv_model
from kws_streaming.models_sub import tflite_utils
from kws_streaming.models_sub import utils


def prepare_calibration_data(model, input_data):
  """Prepares calibration data for post training calibration."""

  calibration_input_data = []
  calibration_states = []
  states = model.states(return_np=True)

  stream_step_size = model.stride
  start = 0
  end = stream_step_size
  while end <= input_data.shape[1]:
    # get new audio chunk
    stream_update = input_data[:, start:end]
    # update indexes of streamed updates
    start = end
    end += stream_step_size

    calibration_input_data.append(stream_update)
    calibration_states.append(states)

    # classification result of a current frame
    outputs = model.stream_inference(stream_update, states)

    for key in states.keys():
      states[key] = outputs[key]

  return calibration_input_data, calibration_states


class ConvModelTest(tf.test.TestCase):
  """End to end test ConvModel model.

  The model is streaming and qauntization aware.
  It will be trained quantized and tested in streaming and non streaming modes.
  """

  def test_conv_model_end_to_end(self):

    # prepare training and testing data
    # we will use mnist data to train a streaming and qauntization aware model
    # mnist data are 2d:
    # we can consider first dimension as time and second dimension as feature
    (train_images,
     train_labels), (test_images,
                     test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images = train_images.reshape(train_images.shape[0], 28, 28,
                                        1)[:200,]
    train_labels = train_labels[:200,]
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)[:50,]
    test_labels = test_labels[:50,]

    # create and train quantization aware model in non streaming mode
    model = conv_model.ConvModel(label_count=10, apply_quantization=True)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    model.fit(
        train_images,
        train_labels,
        epochs=1,
        validation_data=(test_images, test_labels))
    model.summary()

    # create streaming graph
    feature_dim = 28
    states = model.states()
    _ = model.stream_inference(
        tf.zeros((model.inference_batch_size, model.stride, feature_dim, 1),
                 dtype=tf.dtypes.float32,
                 name=model.input_tensor_name), states)

    # one test image
    train_image = train_images[:1,]

    # run tf non streaming inference
    non_stream_output_tf = model.predict(train_image)

    # run tf streaming inference
    stream_output_tf = utils.run_stream_inference_classification(
        model, train_image)

    # compare tf streaming and non streaming outputs
    self.assertAllClose(stream_output_tf, non_stream_output_tf, atol=1e-5)

    # save model in saved_model format
    states_signature = {}
    for state_name, state in states.items():
      states_signature[state_name] = tf.TensorSpec(
          state.shape, tf.float32, name=state_name)
    concrete_function = model.stream_inference.get_concrete_function(
        inputs=tf.TensorSpec(
            (model.inference_batch_size, model.stride, feature_dim, 1),
            dtype=tf.float32,
            name='input_0'),
        states=states_signature)
    saved_model_path = self.get_temp_dir()
    tf.saved_model.save(model, saved_model_path, signatures=concrete_function)

    calibration_input_data, calibration_states = prepare_calibration_data(
        model, train_image)

    def representative_dataset():
      def _representative_dataset_gen():
        for i in range(len(calibration_input_data)):
          yield [
              calibration_input_data[i].astype(np.float32),  # input audio
              calibration_states[i]
              [model.flatten.get_core_layer().name].numpy().astype(np.float32),
              calibration_states[i][
                  model.conv2.get_core_layer().name].numpy().astype(np.float32),
          ]
      return _representative_dataset_gen

    # convert saved_model to tflite with post training quantization
    tflite_model_path = os.path.join(saved_model_path, 'model.tflite')
    utils.saved_model_to_tflite(
        saved_model_path,
        tflite_model_path=tflite_model_path,
        optimizations=[tf.lite.Optimize.DEFAULT],
        inference_type=tf.int8,
        experimental_new_quantizer=True,
        representative_dataset=representative_dataset)

    # make inputs/outputs tensor name consistent between tflite and tf
    tflite_utils.tflite_graph_rewrite(tflite_model_path, saved_model_path)

    with tf.io.gfile.GFile(tflite_model_path, 'rb') as f:
      model_tflite = f.read()

    interpreter = tf.lite.Interpreter(model_content=model_tflite)
    interpreter.allocate_tensors()
    input_states = model.states(return_np=True)

    stream_output_tflite = (
        tflite_utils.run_stream_inference_classification_tflite(
            interpreter,
            train_image,
            input_states,
            model.stride,
            input_tensot_name=model.input_tensor_name,
            output_tensor_name=model.output_tensor_name))

    # compare tflite streaming and tf non streaming outputs
    self.assertAllClose(stream_output_tflite, non_stream_output_tf, atol=0.01)


if __name__ == '__main__':
  tf.test.main()
