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

"""Tests for supcon.inputs."""

from absl import flags
from absl.testing import parameterized
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from supcon import enums
from supcon import hparams
from supcon import inputs
from supcon import preprocessing

FLAGS = flags.FLAGS
ALL_MODES = (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL,
             tf.estimator.ModeKeys.PREDICT)


def make_params(input_fn):
  params = {}
  params['hparams'] = hparams.HParams()
  params['hparams'].input_data.input_fn = input_fn.__name__
  params['use_tpu'] = True
  params['data_dir'] = None
  return params


def input_class_params(input_class):
  params = (
      # pyformat: disable
      # (model_mode, image_size, max_samples)
      (enums.ModelMode.TRAIN, 32, None),
      (enums.ModelMode.TRAIN, 224, None),
      (enums.ModelMode.EVAL, 32, None),
      (enums.ModelMode.EVAL, 224, None),
      (enums.ModelMode.INFERENCE, 32, None),
      (enums.ModelMode.INFERENCE, 224, None),
      (enums.ModelMode.TRAIN, 32, 5),
      (enums.ModelMode.EVAL, 32, 5)
      # pyformat: enable
  )
  return [(input_class,) + p for p in params]


class InputsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(input_class_params('TfdsInput'))
  def test_input_class(self, input_class, model_mode, image_size, max_samples):
    split = 'train' if model_mode == enums.ModelMode.TRAIN else 'test'
    batch_size = 2
    dataset_size = 10
    expected_num_batches = dataset_size // batch_size
    if max_samples is not None and model_mode == enums.ModelMode.TRAIN:
      expected_num_batches = max_samples // batch_size

    params = {'batch_size': batch_size}
    if input_class == 'TfdsInput':
      with tfds.testing.mock_data(num_examples=dataset_size):
        data = inputs.TfdsInput(
            'cifar10',
            split,
            mode=model_mode,
            preprocessor=preprocessing.ImageToMultiViewedImagePreprocessor(
                is_training=model_mode == enums.ModelMode.TRAIN,
                preprocessing_options=hparams.ImagePreprocessing(
                    image_size=image_size, num_views=2),
                dataset_options=preprocessing.DatasetOptions(
                    decode_input=False)),
            max_samples=max_samples,
            num_classes=10).input_fn(params)
    else:
      raise ValueError(f'Unknown input class {input_class}')

    expected_num_channels = 3 if model_mode == enums.ModelMode.INFERENCE else 6
    expected_batch_size = (None if model_mode == enums.ModelMode.INFERENCE else
                           batch_size)

    if model_mode == enums.ModelMode.INFERENCE:
      self.assertIsInstance(data,
                            tf.estimator.export.TensorServingInputReceiver)
      image_shape = data.features.shape.as_list()
    else:
      self.assertIsInstance(data, tf.data.Dataset)
      shapes = tf.data.get_output_shapes(data)
      image_shape = shapes[0].as_list()
      label_shape = shapes[1].as_list()
      self.assertEqual([expected_batch_size], label_shape)
    self.assertEqual(
        [expected_batch_size, image_size, image_size, expected_num_channels],
        image_shape)

    if model_mode == enums.ModelMode.INFERENCE:
      return

    # Now extract the Tensors
    data = tf.data.make_one_shot_iterator(data).get_next()[0]

    with self.cached_session() as sess:
      for i in range(expected_num_batches + 1):
        if i == expected_num_batches and model_mode == enums.ModelMode.EVAL:
          with self.assertRaises(tf.errors.OutOfRangeError):
            sess.run(data)
          break
        else:
          sess.run(data)

  def verify_output_shapes(self, mode, input_fn, expected_image_size):
    params = make_params(input_fn)
    params['hparams'].input_data.preprocessing.image_size = (
        expected_image_size)
    expected_batch_size = {
        tf.estimator.ModeKeys.TRAIN: params['hparams'].bs,
        tf.estimator.ModeKeys.EVAL: params['hparams'].eval.batch_size,
        tf.estimator.ModeKeys.PREDICT: None,
    }[mode]
    params['batch_size'] = expected_batch_size
    input_data = input_fn(mode, params)
    expected_channels = 3 if mode == tf.estimator.ModeKeys.PREDICT else 6
    if mode == tf.estimator.ModeKeys.PREDICT:
      self.assertIsInstance(input_data,
                            tf.estimator.export.TensorServingInputReceiver)
      image_shape = input_data.features.shape.as_list()
    else:
      self.assertIsInstance(input_data, tf.compat.v2.data.Dataset)
      image_shape = tf.data.get_output_shapes(input_data)[0].as_list()
      labels_shape = tf.data.get_output_shapes(input_data)[1].as_list()
    self.assertEqual([
        expected_batch_size, expected_image_size, expected_image_size,
        expected_channels
    ], image_shape)
    if mode != tf.estimator.ModeKeys.PREDICT:
      self.assertEqual([expected_batch_size], labels_shape)

  @parameterized.parameters(*ALL_MODES)
  def test_imagenet(self, model_mode):
    self.verify_output_shapes(model_mode, inputs.imagenet, 224)

  @parameterized.parameters(*ALL_MODES)
  def test_cifar10(self, model_mode):
    self.verify_output_shapes(model_mode, inputs.cifar10, 32)

  @parameterized.parameters(
      (inputs.imagenet, -1, 1281167), (inputs.imagenet, 100, 100),
      (inputs.cifar10, -1, 50000), (inputs.cifar10, 100, 100))
  def test_get_num_train_images(self, input_fn, max_samples, expected_images):
    params = make_params(input_fn)
    params['hparams'].input_data.max_samples = max_samples

    self.assertEqual(
        inputs.get_num_train_images(params['hparams']), expected_images)

  @parameterized.parameters(
      (inputs.imagenet, 50000),
      (inputs.cifar10, 10000),
  )
  def test_get_num_eval_images(self, input_fn, expected_images):
    params = make_params(input_fn)

    self.assertEqual(
        inputs.get_num_eval_images(params['hparams']), expected_images)

  @parameterized.parameters(
      (inputs.imagenet, 1000),
      (inputs.cifar10, 10),
  )
  def test_get_num_classes(self, input_fn, expected_classes):
    params = make_params(input_fn)

    self.assertEqual(
        inputs.get_num_classes(params['hparams']), expected_classes)


if __name__ == '__main__':
  tf.test.main()
