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

"""Tests for cnn with streaming and quantization aware training."""
import numpy as np
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.layers.modes import Modes
from kws_streaming.models import utils
import kws_streaming.models.cnn as cnn
from kws_streaming.train import inference
from tensorflow_model_optimization.python.core.quantization.keras import quantize


def prepare_calibration_data(stream_model, total_stride, input_data):
  """Prepares calibration data for post training calibration."""

  calibration_data = []
  inputs = []
  for s in range(len(stream_model.inputs)):
    inputs.append(
        np.zeros(stream_model.inputs[s].shape, dtype=np.float32))

  # run streaming model and store all input data into calibration_data
  start = 0
  end = total_stride
  while end <= input_data.shape[1]:
    # get new chunk
    stream_update = input_data[:, start:end]
    # update indexes of streamed updates
    start = end
    end += total_stride

    # set input audio data (by default input data at index 0)
    inputs[0] = stream_update

    calibration_data.append(inputs)
    # run inference
    outputs = stream_model.predict(inputs)

    # get output states and set it back to input states
    # which will be fed in the next inference cycle
    for s in range(1, len(stream_model.inputs)):
      inputs[s] = outputs[s]
  return calibration_data


class CNNTest(tf.test.TestCase):
  """End to end test for CNN model.

  The model is streaming and quantization aware.
  It will be trained quantized and tested in streaming and non streaming modes.
  """

  def test_cnn_model_end_to_end(self):

    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf1.Session(config=config)
    tf1.keras.backend.set_session(sess)
    test_utils.set_seed(123)

    # data parameters
    num_time_bins = 12
    feature_size = 12

    # model params.
    total_stride = 2
    params = test_utils.Params([total_stride], 0)
    params.model_name = 'cnn'
    params.cnn_filters = '2'
    params.cnn_kernel_size = '(3,3)'
    params.cnn_act = "'relu'"
    params.cnn_dilation_rate = '(1,1)'
    params.cnn_strides = '(2,2)'
    params.dropout1 = 0.5
    params.units2 = ''
    params.act2 = ''

    params.label_count = 2
    params.return_softmax = True
    params.quantize = 1  # apply quantization aware training

    params.data_shape = (num_time_bins, feature_size)
    params.preprocess = 'custom'

    model = cnn.model(params)
    model.summary()

    # prepare training and testing data
    train_images, train_labels = test_utils.generate_data(
        img_size_y=num_time_bins, img_size_x=feature_size, n_samples=32)
    test_images = train_images
    test_labels = train_labels

    # create and train quantization aware model in non streaming mode
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

    # one test image
    train_image = train_images[:1,]

    # run tf non streaming inference
    non_stream_output_tf = model.predict(train_image)

    # specify input data shape for streaming mode
    params.data_shape = (total_stride, feature_size)
    # TODO(rybakov) add params structure for model with no feature extractor

    # prepare tf streaming model and use it to generate representative_dataset
    with quantize.quantize_scope():
      stream_quantized_model = utils.to_streaming_inference(
          model, params, Modes.STREAM_EXTERNAL_STATE_INFERENCE)

    calibration_data = prepare_calibration_data(stream_quantized_model,
                                                total_stride, train_image)

    def representative_dataset(dtype):
      def _representative_dataset_gen():
        for i in range(len(calibration_data)):
          yield [
              calibration_data[i][0].astype(dtype),  # input audio packet
              calibration_data[i][1].astype(dtype),  # conv state
              calibration_data[i][2].astype(dtype)  # flatten state
          ]

      return _representative_dataset_gen

    # Convert streaming quantization aware model to tflite
    # and apply post training quantization.
    # With below seetings, all inputs and outputs will be float.
    # They will be quantized and de-quantized inside of the model,
    # all conv ops will use quantized versions.
    # For full model quantization we need to set:
    # inference_input_type=tf.int8
    # inference_output_type=tf.int8
    # in this case all inputs and outputs including states will be quantized
    with quantize.quantize_scope():
      tflite_streaming_model = utils.model_to_tflite(
          sess, model, params,
          Modes.STREAM_EXTERNAL_STATE_INFERENCE,
          optimizations=[tf.lite.Optimize.DEFAULT],
          inference_type=tf.int8,
          experimental_new_quantizer=True,
          representative_dataset=representative_dataset(np.float32))

    # run tflite in streaming mode and compare output logits with tf
    interpreter = tf.lite.Interpreter(model_content=tflite_streaming_model)
    interpreter.allocate_tensors()
    input_states = []
    for detail in interpreter.get_input_details():
      input_states.append(np.zeros(detail['shape'], dtype=np.float32))
    stream_out_tflite = inference.run_stream_inference_classification_tflite(
        params, interpreter, train_image, input_states)
    self.assertAllClose(stream_out_tflite, non_stream_output_tf, atol=0.001)


if __name__ == '__main__':
  tf1.disable_eager_execution()
  tf.test.main()
