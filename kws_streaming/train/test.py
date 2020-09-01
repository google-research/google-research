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

"""Test utility functions."""
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import kws_streaming.data.input_data as input_data
from kws_streaming.layers.modes import Modes
from kws_streaming.models import models
from kws_streaming.models import utils


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


def tf_non_stream_model_accuracy(
    flags,
    folder,
    time_shift_samples=0,
    weights_name='best_weights',
    accuracy_name='tf_non_stream_model_accuracy.txt'):
  """Compute accuracy of non streamable model using TF.

  Args:
      flags: model and data settings
      folder: folder name where accuracy report will be stored
      time_shift_samples: time shift of audio data it will be applied in range:
        -time_shift_samples...time_shift_samples
        We can use non stream model for processing stream of audio.
        By default it will be slow, so to speed it up
        we can use non stream model on sampled audio data:
        for example instead of computing non stream model
        on every 20ms, we can run it on every 200ms of audio stream.
        It will reduce total latency by 10 times.
        To emulate sampling effect we use time_shift_samples.
      weights_name: file name with model weights
      accuracy_name: file name for storing accuracy in path + accuracy_name
  Returns:
    accuracy
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)

  audio_processor = input_data.AudioProcessor(flags)

  set_size = audio_processor.set_size('testing')
  tf.keras.backend.set_learning_phase(0)
  flags.batch_size = 100  # set batch size for inference
  set_size = int(set_size / flags.batch_size) * flags.batch_size
  model = models.MODELS[flags.model_name](flags)
  weights_path = os.path.join(flags.train_dir, weights_name)
  model.load_weights(weights_path).expect_partial()
  total_accuracy = 0.0
  count = 0.0
  for i in range(0, set_size, flags.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        flags.batch_size, i, flags, 0.0, 0.0, time_shift_samples, 'testing',
        0.0, 0.0, sess)

    predictions = model.predict(test_fingerprints)
    predicted_labels = np.argmax(predictions, axis=1)
    total_accuracy = total_accuracy + np.sum(
        predicted_labels == test_ground_truth)
    count = count + len(test_ground_truth)
  total_accuracy = total_accuracy / count

  logging.info('TF Final test accuracy on non stream model = %.2f%% (N=%d)',
               *(total_accuracy * 100, set_size))

  path = os.path.join(flags.train_dir, folder)
  if not os.path.exists(path):
    os.makedirs(path)

  fname_summary = 'model_summary_non_stream'
  utils.save_model_summary(model, path, file_name=fname_summary + '.txt')

  tf.keras.utils.plot_model(
      model,
      to_file=os.path.join(path, fname_summary + '.png'),
      show_shapes=True,
      expand_nested=True)

  with open(os.path.join(path, accuracy_name), 'wt') as fd:
    fd.write('%f on set_size %d' % (total_accuracy * 100, set_size))
  return total_accuracy * 100


def tf_stream_state_internal_model_accuracy(
    flags,
    folder,
    weights_name='best_weights',
    accuracy_name='tf_stream_state_internal_model_accuracy_sub_set.txt',
    max_test_samples=1000):
  """Compute accuracy of streamable model with internal state using TF.

  Testign model with batch size 1 can be slow, so accuracy is evaluated
  on subset of data with size max_test_samples
  Args:
      flags: model and data settings
      folder: folder name where accuracy report will be stored
      weights_name: file name with model weights
      accuracy_name: file name for storing accuracy in path + accuracy_name
      max_test_samples: max number of test samples. In this mode model is slow
        with TF because of batch size 1, so accuracy is computed on subset of
        testing data
  Returns:
    accuracy
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)

  logging.info('tf stream model state internal without state resetting'
               'between testing sequences')

  audio_processor = input_data.AudioProcessor(flags)
  set_size = audio_processor.set_size('testing')
  set_size = np.minimum(max_test_samples, set_size)
  inference_batch_size = 1
  tf.keras.backend.set_learning_phase(0)
  flags.batch_size = inference_batch_size  # set batch size
  model = models.MODELS[flags.model_name](flags)
  weights_path = os.path.join(flags.train_dir, weights_name)
  model.load_weights(weights_path).expect_partial()

  model_stream = utils.to_streaming_inference(
      model, flags, Modes.STREAM_INTERNAL_STATE_INFERENCE)

  total_accuracy = 0.0
  count = 0.0
  for i in range(0, set_size, inference_batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        inference_batch_size, i, flags, 0.0, 0.0, 0, 'testing', 0.0, 0.0, sess)

    if flags.preprocess == 'raw':
      start = 0
      end = flags.window_stride_samples
      while end <= test_fingerprints.shape[1]:
        # get overlapped audio sequence
        stream_update = test_fingerprints[:, start:end]

        # classification result of a current frame
        stream_output_prediction = model_stream.predict(stream_update)
        stream_output_arg = np.argmax(stream_output_prediction)

        # update indexes of streamed updates
        start = end
        end = start + flags.window_stride_samples
    else:
      # iterate over frames
      for t in range(test_fingerprints.shape[1]):
        # get new frame from stream of data
        stream_update = test_fingerprints[:, t, :]

        # [batch, time=1, feature]
        stream_update = np.expand_dims(stream_update, axis=1)

        # classification result of a current frame
        stream_output_prediction = model_stream.predict(stream_update)
        stream_output_arg = np.argmax(stream_output_prediction)

    total_accuracy = total_accuracy + (
        test_ground_truth[0] == stream_output_arg)
    count = count + 1
    if i % 200 == 0 and i:
      logging.info(
          'tf test accuracy, stream model state internal = %.2f%% %d out of %d',
          *(total_accuracy * 100 / count, i, set_size))

  total_accuracy = total_accuracy / count
  logging.info(
      'TF Final test accuracy of stream model state internal = %.2f%% (N=%d)',
      *(total_accuracy * 100, set_size))

  path = os.path.join(flags.train_dir, folder)
  if not os.path.exists(path):
    os.makedirs(path)

  fname_summary = 'model_summary_stream_state_internal'
  utils.save_model_summary(model_stream, path, file_name=fname_summary + '.txt')

  tf.keras.utils.plot_model(
      model_stream,
      to_file=os.path.join(path, fname_summary + '.png'),
      show_shapes=True,
      expand_nested=True)

  with open(os.path.join(path, accuracy_name), 'wt') as fd:
    fd.write('%f on set_size %d' % (total_accuracy * 100, set_size))
  return total_accuracy * 100


def tf_stream_state_external_model_accuracy(
    flags,
    folder,
    weights_name='best_weights',
    accuracy_name='stream_state_external_model_accuracy_sub_set.txt',
    reset_state=False,
    max_test_samples=1000):
  """Compute accuracy of streamable model with external state using TF.

  Args:
      flags: model and data settings
      folder: folder name where accuracy report will be stored
      weights_name: file name with model weights
      accuracy_name: file name for storing accuracy in path + accuracy_name
      reset_state: reset state between testing sequences.
        If True - then it is non streaming testing environment: state will be
          reseted on every test and will not be transferred to another one (as
          it is done in real streaming).
      max_test_samples: max number of test samples. In this mode model is slow
        with TF because of batch size 1, so accuracy is computed on subset of
        testing data
  Returns:
    accuracy
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)

  audio_processor = input_data.AudioProcessor(flags)
  set_size = audio_processor.set_size('testing')
  set_size = np.minimum(max_test_samples, set_size)
  inference_batch_size = 1
  tf.keras.backend.set_learning_phase(0)
  flags.batch_size = inference_batch_size  # set batch size
  model = models.MODELS[flags.model_name](flags)
  weights_path = os.path.join(flags.train_dir, weights_name)
  model.load_weights(weights_path).expect_partial()
  model_stream = utils.to_streaming_inference(
      model, flags, Modes.STREAM_EXTERNAL_STATE_INFERENCE)

  logging.info('tf stream model state external with reset_state %d',
               reset_state)

  inputs = []
  for s in range(len(model_stream.inputs)):
    inputs.append(np.zeros(model_stream.inputs[s].shape, dtype=np.float32))

  total_accuracy = 0.0
  count = 0.0
  inference_batch_size = 1
  for i in range(0, set_size, inference_batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        inference_batch_size, i, flags, 0.0, 0.0, 0, 'testing', 0.0, 0.0, sess)

    if reset_state:
      for s in range(len(model_stream.inputs)):
        inputs[s] = np.zeros(model_stream.inputs[s].shape, dtype=np.float32)

    if flags.preprocess == 'raw':
      start = 0
      end = flags.window_stride_samples
      # iterate over time samples with stride = window_stride_samples
      while end <= test_fingerprints.shape[1]:
        # get new frame from stream of data
        stream_update = test_fingerprints[:, start:end]

        # update indexes of streamed updates
        start = end
        end = start + flags.window_stride_samples

        # set input audio data (by default input data at index 0)
        inputs[0] = stream_update

        # run inference
        outputs = model_stream.predict(inputs)

        # get output states and set it back to input states
        # which will be fed in the next inference cycle
        for s in range(1, len(model_stream.inputs)):
          inputs[s] = outputs[s]

        stream_output_arg = np.argmax(outputs[0])
    else:
      # iterate over frames
      for t in range(test_fingerprints.shape[1]):
        # get new frame from stream of data
        stream_update = test_fingerprints[:, t, :]

        # [batch, time=1, feature]
        stream_update = np.expand_dims(stream_update, axis=1)

        # set input audio data (by default input data at index 0)
        inputs[0] = stream_update

        # run inference
        outputs = model_stream.predict(inputs)

        # get output states and set it back to input states
        # which will be fed in the next inference cycle
        for s in range(1, len(model_stream.inputs)):
          inputs[s] = outputs[s]

        stream_output_arg = np.argmax(outputs[0])
    total_accuracy = total_accuracy + (
        test_ground_truth[0] == stream_output_arg)
    count = count + 1
    if i % 200 == 0 and i:
      logging.info(
          'tf test accuracy, stream model state external = %.2f%% %d out of %d',
          *(total_accuracy * 100 / count, i, set_size))

  total_accuracy = total_accuracy / count
  logging.info(
      'TF Final test accuracy of stream model state external = %.2f%% (N=%d)',
      *(total_accuracy * 100, set_size))

  path = os.path.join(flags.train_dir, folder)
  if not os.path.exists(path):
    os.makedirs(path)

  fname_summary = 'model_summary_stream_state_external'
  utils.save_model_summary(model_stream, path, file_name=fname_summary + '.txt')

  tf.keras.utils.plot_model(
      model_stream,
      to_file=os.path.join(path, fname_summary + '.png'),
      show_shapes=True,
      expand_nested=True)

  with open(os.path.join(path, accuracy_name), 'wt') as fd:
    fd.write('%f on set_size %d' % (total_accuracy * 100, set_size))
  return total_accuracy * 100


def tflite_stream_state_external_model_accuracy(
    flags,
    folder,
    tflite_model_name='stream_state_external.tflite',
    accuracy_name='tflite_stream_state_external_model_accuracy.txt',
    reset_state=False):
  """Compute accuracy of streamable model with external state using TFLite.

  Args:
      flags: model and data settings
      folder: folder name where model is located
      tflite_model_name: file name with tflite model
      accuracy_name: file name for storing accuracy in path + accuracy_name
      reset_state: reset state between testing sequences.
        If True - then it is non streaming testing environment: state will be
          reseted in the beginning of every test sequence and will not be
          transferred to another one (as it is done in real streaming).
  Returns:
    accuracy
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)
  path = os.path.join(flags.train_dir, folder)

  logging.info('tflite stream model state external with reset_state %d',
               reset_state)

  audio_processor = input_data.AudioProcessor(flags)

  set_size = audio_processor.set_size('testing')

  interpreter = tf.lite.Interpreter(
      model_path=os.path.join(path, tflite_model_name))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  inputs = []
  for s in range(len(input_details)):
    inputs.append(np.zeros(input_details[s]['shape'], dtype=np.float32))

  total_accuracy = 0.0
  count = 0.0
  inference_batch_size = 1
  for i in range(0, set_size, inference_batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        inference_batch_size, i, flags, 0.0, 0.0, 0, 'testing', 0.0, 0.0, sess)

    # before processing new test sequence we can reset model state
    # if we reset model state then it is not real streaming mode
    if reset_state:
      for s in range(len(input_details)):
        inputs[s] = np.zeros(input_details[s]['shape'], dtype=np.float32)

    if flags.preprocess == 'raw':
      start = 0
      end = flags.window_stride_samples
      while end <= test_fingerprints.shape[1]:
        stream_update = test_fingerprints[:, start:end]
        stream_update = stream_update.astype(np.float32)

        # update indexes of streamed updates
        start = end
        end = start + flags.window_stride_samples

        # set input audio data (by default input data at index 0)
        interpreter.set_tensor(input_details[0]['index'], stream_update)

        # set input states (index 1...)
        for s in range(1, len(input_details)):
          interpreter.set_tensor(input_details[s]['index'], inputs[s])

        # run inference
        interpreter.invoke()

        # get output: classification
        out_tflite = interpreter.get_tensor(output_details[0]['index'])

        # get output states and set it back to input states
        # which will be fed in the next inference cycle
        for s in range(1, len(input_details)):
          # The function `get_tensor()` returns a copy of the tensor data.
          # Use `tensor()` in order to get a pointer to the tensor.
          inputs[s] = interpreter.get_tensor(output_details[s]['index'])

        out_tflite_argmax = np.argmax(out_tflite)
    else:
      for t in range(test_fingerprints.shape[1]):
        # get new frame from stream of data
        stream_update = test_fingerprints[:, t, :]
        stream_update = np.expand_dims(stream_update, axis=1)

        # [batch, time=1, feature]
        stream_update = stream_update.astype(np.float32)

        # set input audio data (by default input data at index 0)
        interpreter.set_tensor(input_details[0]['index'], stream_update)

        # set input states (index 1...)
        for s in range(1, len(input_details)):
          interpreter.set_tensor(input_details[s]['index'], inputs[s])

        # run inference
        interpreter.invoke()

        # get output: classification
        out_tflite = interpreter.get_tensor(output_details[0]['index'])

        # get output states and set it back to input states
        # which will be fed in the next inference cycle
        for s in range(1, len(input_details)):
          # The function `get_tensor()` returns a copy of the tensor data.
          # Use `tensor()` in order to get a pointer to the tensor.
          inputs[s] = interpreter.get_tensor(output_details[s]['index'])

        out_tflite_argmax = np.argmax(out_tflite)

    total_accuracy = total_accuracy + (
        test_ground_truth[0] == out_tflite_argmax)
    count = count + 1
    if i % 200 == 0 and i:
      logging.info(
          'tflite test accuracy, stream model state external = %f %d out of %d',
          *(total_accuracy * 100 / count, i, set_size))

  total_accuracy = total_accuracy / count
  logging.info(
      'tflite Final test accuracy, stream model state external = %.2f%% (N=%d)',
      *(total_accuracy * 100, set_size))

  with open(os.path.join(path, accuracy_name), 'wt') as fd:
    fd.write('%f on set_size %d' % (total_accuracy * 100, set_size))
  return total_accuracy * 100


def tflite_non_stream_model_accuracy(
    flags,
    folder,
    tflite_model_name='non_stream.tflite',
    accuracy_name='tflite_non_stream_model_accuracy.txt'):
  """Compute accuracy of non streamable model with TFLite.

  Model has to be converted to TFLite and stored in path+tflite_model_name
  Args:
      flags: model and data settings
      folder: folder name where model is located
      tflite_model_name: file name with tflite model
      accuracy_name: file name for storing accuracy in path + accuracy_name
  Returns:
    accuracy
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)
  path = os.path.join(flags.train_dir, folder)

  audio_processor = input_data.AudioProcessor(flags)

  set_size = audio_processor.set_size('testing')

  interpreter = tf.lite.Interpreter(
      model_path=os.path.join(path, tflite_model_name))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  inputs = []
  for s in range(len(input_details)):
    inputs.append(np.zeros(input_details[s]['shape'], dtype=np.float32))

  total_accuracy = 0.0
  count = 0.0
  inference_batch_size = 1
  for i in range(0, set_size, inference_batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        inference_batch_size, i, flags, 0.0, 0.0, 0, 'testing', 0.0, 0.0, sess)

    # set input audio data (by default input data at index 0)
    interpreter.set_tensor(input_details[0]['index'],
                           test_fingerprints.astype(np.float32))

    # run inference
    interpreter.invoke()

    # get output: classification
    out_tflite = interpreter.get_tensor(output_details[0]['index'])

    out_tflite_argmax = np.argmax(out_tflite)

    total_accuracy = total_accuracy + (
        test_ground_truth[0] == out_tflite_argmax)
    count = count + 1
    if i % 200 == 0 and i:
      logging.info(
          'tflite test accuracy, non stream model = %.2f%% %d out of %d',
          *(total_accuracy * 100 / count, i, set_size))

  total_accuracy = total_accuracy / count
  logging.info('tflite Final test accuracy, non stream model = %.2f%% (N=%d)',
               *(total_accuracy * 100, set_size))

  with open(os.path.join(path, accuracy_name), 'wt') as fd:
    fd.write('%f on set_size %d' % (total_accuracy * 100, set_size))
  return total_accuracy * 100


def convert_model_tflite(flags,
                         folder,
                         mode,
                         fname,
                         weights_name='best_weights',
                         optimizations=None):
  """Convert model to streaming and non streaming TFLite.

  Args:
      flags: model and data settings
      folder: folder where converted model will be saved
      mode: inference mode
      fname: file name of converted model
      weights_name: file name with model weights
      optimizations: list of optimization options
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)
  tf.keras.backend.set_learning_phase(0)
  flags.batch_size = 1  # set batch size for inference
  model = models.MODELS[flags.model_name](flags)
  weights_path = os.path.join(flags.train_dir, weights_name)
  model.load_weights(weights_path).expect_partial()
  # convert trained model to non streaming TFLite stateless
  # to finish other tests we do not stop program if exception happen here
  path_model = os.path.join(flags.train_dir, folder)
  if not os.path.exists(path_model):
    os.makedirs(path_model)
  try:
    with open(os.path.join(path_model, fname), 'wb') as fd:
      fd.write(
          utils.model_to_tflite(sess, model, flags, mode, path_model,
                                optimizations))
  except IOError as e:
    logging.warning('FAILED to write file: %s', e)
  except (ValueError, AttributeError, RuntimeError, TypeError) as e:
    logging.warning('FAILED to convert to mode %s, tflite: %s', mode, e)


def convert_model_saved(flags, folder, mode, weights_name='best_weights'):
  """Convert model to streaming and non streaming SavedModel.

  Args:
      flags: model and data settings
      folder: folder where converted model will be saved
      mode: inference mode
      weights_name: file name with model weights
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)
  tf.keras.backend.set_learning_phase(0)
  flags.batch_size = 1  # set batch size for inference
  model = models.MODELS[flags.model_name](flags)
  weights_path = os.path.join(flags.train_dir, weights_name)
  model.load_weights(weights_path).expect_partial()

  path_model = os.path.join(flags.train_dir, folder)
  if not os.path.exists(path_model):
    os.makedirs(path_model)
  try:
    # convert trained model to SavedModel
    utils.model_to_saved(model, flags, path_model, mode)
  except IOError as e:
    logging.warning('FAILED to write file: %s', e)
  except (ValueError, AttributeError, RuntimeError, TypeError,
          AssertionError) as e:
    logging.warning('WARNING: failed to convert to SavedModel: %s', e)
