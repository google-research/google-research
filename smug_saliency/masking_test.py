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

"""Tests for third_party.google_research.google_research.smug_saliency.masking."""
import os
import shutil
import tempfile
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile
import z3

from smug_saliency import masking
from smug_saliency import utils

FLAGS = flags.FLAGS

tf.disable_eager_execution()


def _get_z3_var(index):
  return z3.Int('z3var_' + str(index))


def _create_temporary_tf_graph_fully_connected(test_model_path):
  # Create a test model with 1 hiddenlayer with 4 nodes, and an output layer
  # with softmax activation function and 2 nodes.
  test_model = tf.keras.Sequential([
      tf.keras.layers.Dense(units=4, input_shape=(4,), activation='relu'),
      tf.keras.layers.Dense(units=2, activation='softmax')])
  test_model.compile(
      optimizer=tf.keras.optimizers.Adam(1e-3),
      loss=tf.keras.losses.CategoricalCrossentropy(),
      metrics=['accuracy'])
  # Name of the tensors in the graph:
  # input- dense_input:0
  # weights_first_layer - dense/MatMul/ReadVariableOp:0
  # biases_first_layer - dense/BiasAdd/ReadVariableOp:0
  # first_layer_input- dense/BiasAdd:0
  # first_layer_relu_output - dense/Relu:0
  # final_layer_input - dense_1/BiasAdd:0
  # final_layer_softmax_output - dense_1/Softmax:0
  tf.saved_model.simple_save(
      session=tf.keras.backend.get_session(),
      export_dir=test_model_path,
      inputs={'input': test_model.inputs[0]},
      outputs={'output': test_model.outputs[0]})

  weights = []
  biases = []
  weights_and_biases = test_model.get_weights()
  for i in range(len(weights_and_biases) // 2):
    weights.append(weights_and_biases[2 * i])
    biases.append(weights_and_biases[2 * i + 1])
  return weights, biases


def _create_temporary_tf_graph_cnn(test_model_path):
  # Create a test model with 1 conv layer with 3 kernels shaped (2, 2), and an
  # output layer with 4 nodes and softmax activation function.

  test_model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          3,
          kernel_size=(2, 2),
          strides=(1, 1),
          padding='same',
          activation='relu',
          input_shape=(4, 4, 3)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(4, activation='softmax')])
  test_model.compile(
      optimizer=tf.keras.optimizers.Adam(1e-3),
      loss=tf.keras.losses.CategoricalCrossentropy(),
      metrics=['accuracy'])
  # Name of the tensors in the graph:
  # input - conv2d_input:0
  # weights_first_layer - conv2d/Conv2D/ReadVariableOp:0
  # biases_first_layer - conv2d/bias/Read/ReadVariableOp:0
  # first_layer_input - conv2d/BiasAdd:0
  # first_layer_relu_output - conv2d/Relu:0
  # final_layer_input - dense/BiasAdd:0
  # final_layer_softmax_output - dense/Softmax:0
  tf.saved_model.simple_save(
      session=tf.keras.backend.get_session(),
      export_dir=test_model_path,
      inputs={'input': test_model.inputs[0]},
      outputs={'output': test_model.outputs[0]})


def _create_temporary_tf_graph_text_cnn(test_model_path):
  # Create a test model with 1 conv layer with 4 kernels shaped (3, 10),
  # and an output layer with 1 nodes and sigmoid activation function.
  test_model = tf.keras.Sequential([
      tf.keras.Input(shape=(5,)),  # max words = 5
      tf.keras.layers.Embedding(10, 10),  # num top words = 10
      tf.keras.layers.Conv1D(
          filters=4, strides=1, kernel_size=3, activation='relu'),
      tf.keras.layers.GlobalMaxPooling1D(),
      tf.keras.layers.Dense(1, activation='sigmoid')])
  test_model.compile(
      optimizer=tf.keras.optimizers.Adam(1e-3),
      loss=tf.keras.losses.CategoricalCrossentropy(),
      metrics=['accuracy'])

  # Name of the tensors in the graph:
  # input - input_1:0
  # embedding - embedding/embedding_lookup/Identity_1:0
  # weights_first_layer - conv1d/conv1d/ExpandDims_1:0
  # biases_first_layer - conv1d/BiasAdd/ReadVariableOp:0
  # first_layer_input - conv1d/BiasAdd:0
  # first_layer_relu_output - conv1d/Relu:0
  # final_layer_input - dense/BiasAdd:0
  # final_layer_softmax_output - dense/Sigmoid:0
  tf.saved_model.simple_save(
      session=tf.keras.backend.get_session(),
      export_dir=test_model_path,
      inputs={'input': test_model.inputs[0]},
      outputs={'output': test_model.outputs[0]})


class MaskingLibTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.test_model_path = os.path.join(
        tempfile.mkdtemp(dir=FLAGS.test_tmpdir), 'checkpoint')
    gfile.MakeDirs(self.test_model_path)

  def tearDown(self):
    shutil.rmtree(self.test_model_path)
    super().tearDown()

  @parameterized.parameters(1, 3)
  def test_encode_input(self, image_channels):
    # Creates a random image and checks if the encoded image, after multiplying
    # the mask bits (set to 1) is the same as the original image.
    image_edge_length = 2
    image = np.random.rand(image_edge_length, image_edge_length, image_channels)
    z3_var = _get_z3_var(index=0)

    # encoded_image has dimensions
    # (image_channels, image_edge_length, image_edge_length)
    encoded_image = masking._encode_input(
        image=image,
        z3_mask=[z3_var for _ in range(image_edge_length ** 2)],
        window_size=1)
    solver = z3.Solver()
    solver.add(z3_var == 1)
    # Swap the axes of the image so that it has the same dimensions as the
    # encoded image.
    image = masking._reorder(image).reshape(-1)
    encoded_image = utils.flatten_nested_lists(encoded_image)
    for i in range(image_channels * image_edge_length ** 2):
      solver.add(encoded_image[i] == image[i])

    self.assertEqual(str(solver.check()), 'sat')

  def test_formulate_smt_constraints_convolution_layer(self):
    with self.test_session():
      # Temporary graphs should be created inside a session. Notice multiple
      # graphs are being created in this particular code. So, if each graph
      # isn't created inside a separate session, the tensor names will have
      # unwanted integer suffices, which then would cause problems while
      # accessing tensors by name.
      _create_temporary_tf_graph_cnn(self.test_model_path)
    image_edge_length = 4
    image_channels = 3
    # The 1st convolution layer has 48 neurons.
    top_k = np.random.randint(low=1, high=48)
    image = np.ones((image_edge_length, image_edge_length, image_channels))
    tensor_names = {
        'input': 'conv2d_input:0',
        'first_layer': 'conv2d/BiasAdd:0',
        'first_layer_relu': 'conv2d/Relu:0',
        'logits': 'dense/BiasAdd:0',
        'softmax': 'dense/Softmax:0',
        'weights_layer_1': 'conv2d/Conv2D/ReadVariableOp:0',
        'biases_layer_1': 'conv2d/bias/Read/ReadVariableOp:0'}
    session = utils.restore_model(self.test_model_path)
    cnn_predictions = session.run(
        tensor_names,
        feed_dict={
            tensor_names['input']: image.reshape(
                (1, image_edge_length, image_edge_length, image_channels))})
    z3_mask = [_get_z3_var(index=i) for i in range(image_edge_length ** 2)]
    first_layer_activations = masking._reorder(masking._remove_batch_axis(
        cnn_predictions['first_layer'])).reshape(-1)
    masked_input = masking._encode_input(
        image=image, z3_mask=z3_mask, window_size=1)

    z3_optimizer = masking._formulate_smt_constraints_convolution_layer(
        z3_optimizer=utils.ImageOptimizer(
            z3_mask=z3_mask,
            window_size=1,
            edge_length=image_edge_length),
        kernels=masking._reorder(cnn_predictions['weights_layer_1']),
        biases=cnn_predictions['biases_layer_1'],
        chosen_indices=first_layer_activations.argsort()[-top_k:],
        conv_activations=first_layer_activations,
        input_activation_maps=masked_input,
        output_activation_map_shape=(image_edge_length, image_edge_length),
        strides=1,
        padding=(0, 1),
        gamma=0.5)
    mask, result = z3_optimizer.generate_mask()

    self.assertEqual(result, 'sat')
    self.assertEqual(mask.shape, (image_edge_length, image_edge_length))
    session.close()

  def test_formulate_smt_constraints_convolution_layer_text(self):
    with self.test_session():
      # Temporary graphs should be created inside a session. Notice multiple
      # graphs are being created in this particular code. So, if each graph
      # isn't created inside a separate session, the tensor names will have
      # unwanted integer suffices, which then would cause problems while
      # accessing tensors by name.
      _create_temporary_tf_graph_text_cnn(self.test_model_path)

    # The 1st convolution layer has 12 neurons.
    image = np.ones(5)
    tensor_names = {
        'input': 'input_1:0',
        'embedding': 'embedding/embedding_lookup/Identity_1:0',
        'first_layer': 'conv1d/BiasAdd:0',
        'first_layer_relu': 'conv1d/Relu:0',
        'logits': 'dense/BiasAdd:0',
        'softmax': 'dense/Sigmoid:0',
        'weights_layer_1': 'conv1d/conv1d/ExpandDims_1:0',
        'biases_layer_1': 'conv1d/BiasAdd/ReadVariableOp:0'}
    session = utils.restore_model(self.test_model_path)
    cnn_predictions = session.run(
        tensor_names, feed_dict={
            tensor_names['input']: image.reshape(1, 5)})
    text_embedding = masking._remove_batch_axis(cnn_predictions['embedding'])
    z3_mask = [z3.Int('mask_%d' % i) for i in range(text_embedding.shape[0])]
    masked_input = []
    for mask_bit, embedding_row in zip(z3_mask, text_embedding):
      masked_input.append([z3.ToReal(mask_bit) * i for i in embedding_row])
    first_layer_activations = masking._reorder(
        masking._remove_batch_axis(cnn_predictions['first_layer'])).reshape(-1)
    z3_optimizer = masking._formulate_smt_constraints_convolution_layer(
        z3_optimizer=utils.TextOptimizer(z3_mask=z3_mask),
        kernels=masking._reshape_kernels(
            kernels=cnn_predictions['weights_layer_1'],
            model_type='text_cnn'),
        biases=cnn_predictions['biases_layer_1'],
        chosen_indices=first_layer_activations.argsort()[-5:],
        conv_activations=first_layer_activations,
        input_activation_maps=[masked_input],
        output_activation_map_shape=masking._get_activation_map_shape(
            activation_maps_shape=cnn_predictions['first_layer'].shape,
            model_type='text_cnn'),
        strides=1,
        padding=(0, 0),
        gamma=0.5)
    mask, result = z3_optimizer.generate_mask()

    self.assertEqual(result, 'sat')
    self.assertEqual(mask.shape, (5,))
    session.close()

  def test_formulate_smt_constraints_fully_connected_layer(self):
    # For a neural network with 4 hidden nodes in the first layer, with the
    # original first layer activations = 1, and the SMT encoding of
    # the first hidden nodes- [mask_0, mask_1, mask_2, mask_3]. For
    # masked_activation > delta * original (k such constraints), only k mask
    # bits should be set to 1 and the others to 0.
    image_edge_length = 2
    top_k = np.random.randint(low=1, high=image_edge_length ** 2)
    z3_mask = [_get_z3_var(index=i) for i in range(image_edge_length ** 2)]
    smt_first_layer = [1 * z3.ToReal(i) for i in z3_mask]
    nn_first_layer = np.ones(len(smt_first_layer))

    z3_optimizer = utils.ImageOptimizer(
        z3_mask=z3_mask, window_size=1, edge_length=image_edge_length)
    z3_optimizer = masking._formulate_smt_constraints_fully_connected_layer(
        z3_optimizer=z3_optimizer,
        smt_first_layer=smt_first_layer,
        nn_first_layer=nn_first_layer,
        top_k=top_k,
        gamma=np.random.rand())
    mask, result = z3_optimizer._optimize()

    self.assertEqual(result, 'sat')
    self.assertEqual(np.sum(mask), top_k)

  def test_smt_constraints_final_layer(self):
    # The SMT encoding of the final layer - [mask_0, mask_1, mask_2, mask_3].
    # For logit_label_index > rest, the mask_bit at label_index should be set to
    # 1.
    image_edge_length = 2
    label_index = np.random.randint(low=0, high=image_edge_length ** 2)
    z3_mask = [_get_z3_var(index=i) for i in range(image_edge_length ** 2)]
    smt_output = [1 * z3.ToReal(i) for i in z3_mask]

    z3_optimizer = utils.ImageOptimizer(
        z3_mask=z3_mask, window_size=1, edge_length=image_edge_length)
    z3_optimizer = masking._formulate_smt_constraints_final_layer(
        z3_optimizer=z3_optimizer,
        smt_output=smt_output,
        delta=np.random.rand(),
        label_index=label_index)
    mask, result = z3_optimizer._optimize()

    self.assertEqual(result, 'sat')
    self.assertEqual(mask.reshape(-1)[label_index], 1)
    self.assertEqual(np.sum(mask), 1)

  def test_find_mask_first_layer(self):
    with self.test_session():
      # Temporary graphs should be created inside a session. Notice multiple
      # graphs are being created in this particular code. So, if each graph
      # isn't created inside a separate session, the tensor names will have
      # unwanted integer suffices, which then would cause problems while
      # accessing tensors by name.
      _create_temporary_tf_graph_fully_connected(self.test_model_path)
    result = masking.find_mask_first_layer(
        image=np.zeros((2, 2, 1)),
        run_params=masking.RunParams(
            **{
                'model_path': self.test_model_path,
                'tensor_names': {
                    'input': 'dense_input:0',
                    'first_layer': 'dense/BiasAdd:0',
                    'first_layer_relu': 'dense/Relu:0',
                    'softmax': 'dense_1/Softmax:0',
                    'logits': 'dense_1/BiasAdd:0',
                    'weights_layer_1': 'dense/MatMul/ReadVariableOp:0',
                    'biases_layer_1': 'dense/BiasAdd/ReadVariableOp:0'
                },
                'image_placeholder_shape': (1, 4),
                'model_type': 'fully_connected',
                'padding': (0, 0),
                'strides': 0,
                'activations': None,
            }),
        label_index=0,
        score_method='activations',
        window_size=1,
        top_k=4,
        gamma=0.5,
        timeout=600,
        num_unique_solutions=5)
    self.assertEqual(result['image'].shape, (4,))
    self.assertEqual(result['unmasked_logits'].shape, (2,))
    self.assertEqual(result['unmasked_first_layer'].shape, (4,))
    self.assertEqual(result['masks'][0].shape, (4,))
    self.assertLen(result['masks'], 5)
    self.assertLen(result['masked_first_layer'], 5)
    self.assertLen(result['inv_masked_first_layer'], 5)
    self.assertLen(result['masked_images'], 5)
    self.assertLen(result['inv_masked_images'], 5)
    self.assertLen(result['masked_logits'], 5)
    self.assertLen(result['inv_masked_logits'], 5)
    self.assertLen(result['solver_outputs'], 5)

  def test_find_mask_full_encoding(self):
    with self.test_session():
      # Temporary graphs should be created inside a session. Notice multiple
      # graphs are being created in this particular code. So, if each graph
      # isn't created inside a separate session, the tensor names will have
      # unwanted integer suffices, which then would cause problems while
      # accessing tensors by name.
      weights, biases = _create_temporary_tf_graph_fully_connected(
          self.test_model_path)
    result = masking.find_mask_full_encoding(
        image=np.zeros((2, 2, 1)),
        weights=weights,
        biases=biases,
        run_params=masking.RunParams(**{
            'model_path': self.test_model_path,
            'tensor_names': {
                'input': 'dense_input:0',
                'first_layer': 'dense/BiasAdd:0',
                'first_layer_relu': 'dense/Relu:0',
                'softmax': 'dense_1/Softmax:0',
                'logits': 'dense_1/BiasAdd:0',
                'weights_layer_1': 'dense/MatMul/ReadVariableOp:0',
                'biases_layer_1': 'dense/BiasAdd/ReadVariableOp:0'},
            'image_placeholder_shape': (1, 4),
            'model_type': 'fully_connected',
            'padding': (0, 0),
            'strides': 0,
            'activations': ['relu', 'linear'],
        }),
        window_size=1,
        label_index=0,
        delta=0,
        timeout=600,
        num_unique_solutions=5)
    self.assertEqual(result['image'].shape, (4,))
    self.assertEqual(result['unmasked_logits'].shape, (2,))
    self.assertEqual(result['unmasked_first_layer'].shape, (4,))
    self.assertEqual(result['masks'][0].shape, (4,))
    self.assertLen(result['masks'], 5)
    self.assertLen(result['masked_first_layer'], 5)
    self.assertLen(result['inv_masked_first_layer'], 5)
    self.assertLen(result['masked_images'], 5)
    self.assertLen(result['inv_masked_images'], 5)
    self.assertLen(result['masked_logits'], 5)
    self.assertLen(result['inv_masked_logits'], 5)
    self.assertLen(result['solver_outputs'], 5)

  def test_find_mask_first_layer_text_cnn(self):
    with self.test_session():
      # Temporary graphs should be created inside a session. Notice multiple
      # graphs are being created in this particular code. So, if each graph
      # isn't created inside a separate session, the tensor names will have
      # unwanted integer suffices, which then would cause problems while
      # accessing tensors by name.
      _create_temporary_tf_graph_text_cnn(self.test_model_path)
    result = masking.find_mask_first_layer(
        image=np.zeros(5),
        run_params=masking.RunParams(
            **{
                'model_path': self.test_model_path,
                'tensor_names': {
                    'input': 'input_1:0',
                    'embedding': 'embedding/embedding_lookup/Identity_1:0',
                    'first_layer': 'conv1d/BiasAdd:0',
                    'first_layer_relu': 'conv1d/Relu:0',
                    'logits': 'dense/BiasAdd:0',
                    'softmax': 'dense/Sigmoid:0',
                    'weights_layer_1': 'conv1d/conv1d/ExpandDims_1:0',
                    'biases_layer_1': 'conv1d/BiasAdd/ReadVariableOp:0',
                },
                'image_placeholder_shape': (1, 5),
                'model_type': 'text_cnn',
                'padding': (0, 0),
                'strides': 1,
                'activations': None,
            }),
        window_size=1,
        label_index=0,
        score_method='activations',
        top_k=4,
        gamma=0.5,
        timeout=600,
        num_unique_solutions=5)
    self.assertEqual(result['image'].shape, (5,))
    self.assertEqual(result['unmasked_logits'].shape, (1,))
    self.assertEqual(result['unmasked_first_layer'].shape, (12,))
    self.assertEqual(result['masks'][0].shape, (5,))
    self.assertLen(result['masks'], 5)
    self.assertLen(result['masked_first_layer'], 5)
    self.assertLen(result['inv_masked_first_layer'], 5)
    self.assertLen(result['masked_images'], 5)
    self.assertLen(result['inv_masked_images'], 5)
    self.assertLen(result['masked_logits'], 5)
    self.assertLen(result['inv_masked_logits'], 5)
    self.assertLen(result['solver_outputs'], 5)

  def test_find_mask_first_layer_cnn(self):
    with self.test_session():
      # Temporary graphs should be created inside a session. Notice multiple
      # graphs are being created in this particular code. So, if each graph
      # isn't created inside a separate session, the tensor names will have
      # unwanted integer suffices, which then would cause problems while
      # accessing tensors by name.
      _create_temporary_tf_graph_cnn(self.test_model_path)
    result = masking.find_mask_first_layer(
        image=np.zeros((4, 4, 3)),
        run_params=masking.RunParams(
            **{
                'model_path': self.test_model_path,
                'tensor_names': {
                    'input': 'conv2d_input:0',
                    'first_layer': 'conv2d/BiasAdd:0',
                    'first_layer_relu': 'conv2d/Relu:0',
                    'logits': 'dense/BiasAdd:0',
                    'softmax': 'dense/Softmax:0',
                    'weights_layer_1': 'conv2d/Conv2D/ReadVariableOp:0',
                    'biases_layer_1': 'conv2d/bias/Read/ReadVariableOp:0'
                },
                'image_placeholder_shape': (1, 4, 4, 3),
                'model_type': 'cnn',
                'padding': (0, 1),
                'strides': 1,
                'activations': None,
            }),
        window_size=1,
        label_index=0,
        score_method='activations',
        top_k=4,
        gamma=0.5,
        timeout=600,
        num_unique_solutions=5)
    self.assertEqual(result['image'].shape, (48,))
    self.assertEqual(result['unmasked_logits'].shape, (4,))
    self.assertEqual(result['unmasked_first_layer'].shape, (48,))
    self.assertEqual(result['masks'][0].shape, (48,))
    self.assertLen(result['masks'], 5)
    self.assertLen(result['masked_first_layer'], 5)
    self.assertLen(result['inv_masked_first_layer'], 5)
    self.assertLen(result['masked_images'], 5)
    self.assertLen(result['inv_masked_images'], 5)
    self.assertLen(result['masked_logits'], 5)
    self.assertLen(result['inv_masked_logits'], 5)
    self.assertLen(result['solver_outputs'], 5)

  @parameterized.parameters(
      ('get_saliency_map', 'activations'),
      ('get_saliency_map', 'integrated_gradients'),
      ('_get_gradients', 'gradients'),
      ('_get_gradients', 'blurred_gradients'),
  )
  def test_sort_indices(self, function_to_be_mocked, score_method):
    # priority array is reverse engineered such that,
    # the output of masking._sort_indices is [0, ..., num_hidden_nodes]
    priority = np.moveaxis(np.arange(48).reshape((1, 4, 4, 3)), 1, -1)
    with mock.patch.object(
        masking, function_to_be_mocked,
        return_value=priority), mock.patch.object(
            masking, '_apply_blurring', return_value=mock.MagicMock()):
      sorted_indices = masking._sort_indices(
          session=mock.MagicMock(),
          image=mock.MagicMock(),
          label_index=0,
          run_params=mock.MagicMock(),
          unmasked_predictions={
              'first_layer': priority,
              'first_layer_relu': priority,},
          score_method=score_method)

    np.testing.assert_array_equal(sorted_indices, np.arange(48))

  def test_get_gradients(self):
    with self.test_session() as session:
      _create_temporary_tf_graph_cnn(self.test_model_path)

      gradients = masking._get_gradients(
          session=session,
          graph=tf.get_default_graph(),
          features=np.ones((1, 4, 4, 3)),
          label_index=0,
          input_tensor_name='conv2d_input:0',
          output_tensor_name='dense/Softmax:0')

    self.assertEqual(gradients.shape, (1, 4, 4, 3))

  @parameterized.parameters(
      (('The input image should have 3 dimensions. Shape of the image: '
        r'\(4, 4\)'), np.ones((4, 4))),
      (('The input image should have height == width. '
        r'Shape of the input image: \(4, 5, 1\)'), np.ones((4, 5, 1))),
      (('The color channels of the input image has a value other than 1 or 3. '
        r'Shape of the image: \(4, 4, 2\)'), np.ones((4, 4, 2))),
  )
  def test_verify_image_dimensions(self, error, image):
    with self.assertRaisesRegex(ValueError, error):
      masking._verify_image_dimensions(image)

  @parameterized.parameters(
      (np.ones((4, 4)), 'text_cnn',
       r'Invalid mask shape: \(4, 4\). Expected a mask with 1 dimension.'),
      (np.ones(4), 'cnn',
       r'Invalid mask shape: \(4,\). Expected a mask with 2 equal dimensions.'),
      (np.ones(4), 'fully_connected',
       r'Invalid mask shape: \(4,\). Expected a mask with 2 equal dimensions.'),
  )
  def test_verify_mask_dimensions(self, mask, model_type, error):
    with self.assertRaisesRegex(ValueError, error):
      masking._verify_mask_dimensions(mask, model_type)

  def test_reorder(self):
    shape = tuple(np.random.randint(low=1, high=10, size=4))

    self.assertEqual(masking._reorder(np.ones(shape)).shape,
                     (shape[3], shape[0], shape[1], shape[2]))

  def test_remove_batch_axis(self):
    # The batch size has to be 1.
    shape = tuple(np.append([1], np.random.randint(low=1, high=10, size=3)))

    self.assertEqual(masking._remove_batch_axis(np.ones(shape)).shape,
                     (shape[1], shape[2], shape[3]))

  def test_remove_batch_axis_error(self):
    shape = tuple(np.random.randint(low=2, high=10, size=4))

    with self.assertRaisesRegex(
        ValueError, ('The array doesn\'t have the batch dimension as 1. '
                     'Received an array with length along the batch '
                     'dimension: %d' % shape[0])):
      masking._remove_batch_axis(np.ones(shape))

  def test_process_text_error(self):
    with self.assertRaisesRegex(ValueError,
                                ('The text input should be a 1D numpy array. '
                                 r'Shape of the received input: \(1, 500\)')):
      masking._process_text(image=np.ones((1, 500)), run_params=None)

  def test_get_hidden_node_location_image(self):
    num_channels = 64
    output_activation_map_size = 112
    flattened_indices = np.arange(
        num_channels * output_activation_map_size ** 2).reshape(
            num_channels, output_activation_map_size,
            output_activation_map_size)
    true_row = np.random.randint(low=0, high=output_activation_map_size)
    true_column = np.random.randint(low=0, high=output_activation_map_size)
    true_channel = np.random.randint(low=0, high=num_channels)

    (predicted_channel, predicted_row,
     predicted_column) = masking._get_hidden_node_location(
         flattened_index=flattened_indices[true_channel][true_row][true_column],
         num_rows=output_activation_map_size,
         num_columns=output_activation_map_size)

    self.assertEqual(true_channel, predicted_channel)
    self.assertEqual(true_row, predicted_row)
    self.assertEqual(true_column, predicted_column)

  def test_get_hidden_node_location_text(self):
    num_channels = 128
    output_activation_map_shape = (498, 1)
    flattened_indices = np.arange(
        num_channels * output_activation_map_shape[0]).reshape(
            num_channels, output_activation_map_shape[0],
            output_activation_map_shape[1])
    true_row = np.random.randint(low=0, high=output_activation_map_shape[0])
    true_channel = np.random.randint(low=0, high=num_channels)
    true_column = 0

    (predicted_channel, predicted_row,
     predicted_column) = masking._get_hidden_node_location(
         flattened_index=flattened_indices[true_channel][true_row][true_column],
         num_rows=output_activation_map_shape[0],
         num_columns=output_activation_map_shape[1])

    self.assertEqual(true_channel, predicted_channel)
    self.assertEqual(true_row, predicted_row)
    self.assertEqual(true_column, predicted_column)

  def test_get_activation_map_shape_image(self):
    activation_maps_shape = tuple(np.random.randint(low=1, high=10, size=4))

    self.assertEqual(
        masking._get_activation_map_shape(
            activation_maps_shape, model_type='cnn'),
        (activation_maps_shape[1], activation_maps_shape[2]))

  def test_get_activation_map_shape_text(self):
    activation_maps_shape = tuple(np.random.randint(low=1, high=10, size=3))

    self.assertEqual(
        masking._get_activation_map_shape(
            activation_maps_shape, model_type='text_cnn'),
        (activation_maps_shape[1], 1))

  @parameterized.parameters(
      (('Invalid model_type: text. Expected one of - '
        'fully_connected, cnn or text_cnn'), (1, 1, 1), 'text'),
      (r'Invalid activation_maps_shape: \(1, 1, 1\).Expected length 4.',
       (1, 1, 1), 'cnn'),
      (r'Invalid activation_maps_shape: \(1, 1, 1, 1\).Expected length 3.',
       (1, 1, 1, 1), 'text_cnn'),
      )
  def test_verify_activation_maps_shape(
      self, activation_maps_shape, model_type, error):
    with self.assertRaisesRegex(ValueError, error):
      masking._verify_activation_maps_shape(activation_maps_shape, model_type)

  @parameterized.parameters(('cnn', (3, 0, 1, 2)),
                            ('text_cnn', (3, 1, 2, 0)))
  def test_reshape_kernels(self, model_type, reshaped_dimensions):
    activation_maps_shape = tuple(np.random.randint(low=1, high=10, size=4))

    reshaped_kernel = masking._reshape_kernels(
        kernels=np.ones(activation_maps_shape),
        model_type=model_type)

    self.assertEqual(reshaped_kernel.shape,
                     (activation_maps_shape[reshaped_dimensions[0]],
                      activation_maps_shape[reshaped_dimensions[1]],
                      activation_maps_shape[reshaped_dimensions[2]],
                      activation_maps_shape[reshaped_dimensions[3]]))

  @parameterized.parameters(
      ('integrated_gradients', (4, 4, 3)),
      ('integrated_gradients_black_white_baselines', (4, 4, 3)),
      ('xrai', (4, 4)))
  def test_get_saliency_map(self, saliency_method, saliency_map_shape):
    with self.test_session():
      # Temporary graphs should be created inside a session. Notice multiple
      # graphs are being created in this particular code. So, if each graph
      # isn't created inside a separate session, the tensor names will have
      # unwanted integer suffices, which then would cause problems while
      # accessing tensors by name.
      _create_temporary_tf_graph_cnn(self.test_model_path)
    self.assertEqual(
        masking.get_saliency_map(
            session=utils.restore_model(self.test_model_path),
            features=np.random.rand(4, 4, 3),
            saliency_method=saliency_method,
            label=0,
            input_tensor_name='conv2d_input:0',
            output_tensor_name='dense/Softmax:0',
            ).shape,
        saliency_map_shape)

  def test_get_no_minimization_mask(self):
    mock_session = mock.MagicMock()
    mock_session.run.return_value = {
        # Every hidden node has a receptive field of 2 x 2
        'weights_layer_1': np.ones((1, 2, 2, 1)),
        'first_layer_relu': np.ones((1, 4, 4, 1)),
        'first_layer': np.ones((1, 4, 4, 1)),
    }
    mock_run_params = mock.MagicMock()
    mock_run_params.strides = 1
    mock_run_params.padding = (1, 1)
    mock_run_params.image_placeholder_shape = (4, 4)
    mock_run_params.model_type = 'cnn'
    with mock.patch.object(
        utils, 'restore_model',
        return_value=mock_session), mock.patch.object(
            masking, 'get_saliency_map',
            return_value=mock.MagicMock()), mock.patch.object(
                masking, '_reorder',
                return_value=np.ones(32)), mock.patch.object(
                    masking, '_sort_indices',
                    return_value=[0, 21, 10]), mock.patch.object(
                        masking, '_remove_batch_axis',
                        return_value=mock.MagicMock()):
      mask = masking.get_no_minimization_mask(
          image=np.ones((4, 4)),
          label_index=0,
          top_k=4,
          run_params=mock_run_params,
          sum_attributions=False)
    # The receptive field of hidden node indexed 0 on the padded image -
    # 1 1 0 0 0 0
    # 1 1 0 0 0 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    #
    # The receptive field of hidden node indexed 21 on the padded image -
    # 0 0 0 0 0 0
    # 0 1 1 0 0 0
    # 0 1 1 0 0 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    #
    # The receptive field of hidden node indexed 10 on the padded image -
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    # 0 0 1 1 0 0
    # 0 0 1 1 0 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    #
    # Union of all these masks gives us the no minimisation mask.
    # Then, the padding (1, 1) is removed from the padded image and we the
    # output.
    # becomes -
    # 1 1 0 0
    # 1 1 1 0
    # 0 1 1 0
    # 0 0 0 0

    np.testing.assert_allclose(mask, [[1, 1, 0, 0],
                                      [1, 1, 1, 0],
                                      [0, 1, 1, 0],
                                      [0, 0, 0, 0]])

  def test_get_no_minimization_mask_text(self):
    mock_session = mock.MagicMock()
    mock_session.run.return_value = {
        # Every hidden node has a receptive field of 3
        'weights_layer_1': np.ones((1, 3, 10, 12)),
        'first_layer_relu': np.ones((1, 11, 1)),
        'first_layer': np.ones((1, 11, 1)),
    }
    mock_run_params = mock.MagicMock()
    mock_run_params.strides = 1
    mock_run_params.padding = (1, 2)
    mock_run_params.image_placeholder_shape = (1, 10)
    mock_run_params.model_type = 'text_cnn'
    with mock.patch.object(
        utils, 'restore_model',
        return_value=mock_session), mock.patch.object(
            masking, 'get_saliency_map',
            return_value=mock.MagicMock()), mock.patch.object(
                masking, '_reorder',
                return_value=np.ones(20)), mock.patch.object(
                    masking, '_sort_indices',
                    return_value=[0, 16]), mock.patch.object(
                        masking, '_remove_batch_axis',
                        return_value=mock.MagicMock()):
      mask = masking.get_no_minimization_mask(
          image=np.ones(10),
          label_index=0,
          top_k=4,
          run_params=mock_run_params,
          sum_attributions=False)
    # The receptive field of hidden node indexed 0 on the text -
    # 1 1 1 0 0 0 0 0 0 0 0 0 0
    # The receptive field of hidden node indexed 16
    # (activation map indexed 1, 5th position) on the text -
    # 0 0 0 0 0 1 1 1 0 0 0 0 0
    # We get the below result after taking their union and removing the padding.
    np.testing.assert_allclose(mask, [1, 1, 0, 0, 1, 1, 1, 0, 0, 0])

if __name__ == '__main__':
  absltest.main()
