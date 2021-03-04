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

"""Tests for third_party.google_research.google_research.smug_saliency.utils."""
import os
import tempfile
from unittest import mock
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf
import z3
from smug_saliency import utils

FLAGS = flags.FLAGS

tf.disable_eager_execution()


def _construct_optimizer(z3_mask, optimizer_type):
  if optimizer_type == 'image':
    return utils.ImageOptimizer(
        z3_mask=z3_mask, window_size=1, edge_length=2)
  else:
    return utils.TextOptimizer(z3_mask=z3_mask)


class UtilsTest(parameterized.TestCase):

  @parameterized.parameters('image', 'text')
  def test_optimizer_sat(self, optimizer_type):
    # Here we test if the smt solver can find a solution to the equation-
    # 2*mask_0 - mask_1 - mask_2 + 2*mask_3 > 3.
    # With binary masking variables, there exists only one solution
    # where mask_0, mask_3 = 1 and mask_2, mask_3 = 0
    z3_mask = [z3.Int('mask_0'), z3.Int('mask_1'),
               z3.Int('mask_2'), z3.Int('mask_3')]
    optimizer_instance = _construct_optimizer(
        z3_mask=z3_mask, optimizer_type=optimizer_type)

    optimizer_instance.solver.add(
        2 * z3_mask[0] - z3_mask[1] - z3_mask[2] + 2 * z3_mask[3] > 3)
    learnt_mask, result = optimizer_instance.generate_mask()

    self.assertEqual(result, 'sat')
    np.testing.assert_allclose(learnt_mask.reshape(-1), [1., 0., 0., 1.])

  @parameterized.parameters('image', 'text')
  def test_optimizer_unsat(self, optimizer_type):
    # Here we test if the smt solver can prove the following equation has no
    # solution- 2*mask_0 - mask_1 - mask_2 + 2*mask_3 < -3.
    # With binary masking variables, there doesn't exist a solution.
    z3_mask = [z3.Int('mask_0'), z3.Int('mask_1'),
               z3.Int('mask_2'), z3.Int('mask_3')]
    optimizer_instance = _construct_optimizer(
        z3_mask=z3_mask, optimizer_type=optimizer_type)

    optimizer_instance.solver.add(
        2 * z3_mask[0] - z3_mask[1] - z3_mask[2] + 2 * z3_mask[3] < -3)
    learnt_mask, result = optimizer_instance.generate_mask()

    self.assertEqual(result, 'unsat')
    np.testing.assert_allclose(learnt_mask.reshape(-1), [0., 0., 0., 0.])

  @parameterized.parameters('image', 'text')
  def test_generator(self, optimizer_type):
    # Here we test if the smt solver can find a solution to the equation-
    # 2*mask_0 - mask_1 - mask_2 + 2*mask_3 > 3.
    # With binary masking variables, there exists only one solution
    # where mask_0, mask_3 = 1 and mask_2, mask_3 = 0
    z3_mask = [z3.Int('mask_0'), z3.Int('mask_1'),
               z3.Int('mask_2'), z3.Int('mask_3')]
    optimizer_instance = _construct_optimizer(
        z3_mask=z3_mask, optimizer_type=optimizer_type)
    optimizer_instance.solver.add(
        2 * z3_mask[0] - z3_mask[1] - z3_mask[2] + 2 * z3_mask[3] > 3)

    generator = optimizer_instance.generator(3)

    learnt_mask, result = next(generator)
    self.assertEqual(result, 'sat')
    np.testing.assert_allclose(learnt_mask.reshape(-1), [1., 0., 0., 1.])

    # Since there is only one unique solution so the rest of the solutions are
    # unsat.
    learnt_mask, result = next(generator)
    self.assertEqual(result, 'unsat')
    np.testing.assert_allclose(learnt_mask.reshape(-1), [0., 0., 0., 0.])

    learnt_mask, result = next(generator)
    self.assertEqual(result, 'unsat')
    np.testing.assert_allclose(learnt_mask.reshape(-1), [0., 0., 0., 0.])

    with self.assertRaises(StopIteration):
      next(generator)

  def _assert_z3_constraint_sat(self, constraint, z3_var):
    solver = z3.Solver()
    solver.add(z3_var == 1)
    solver.add(constraint)
    self.assertEqual(str(solver.check()), 'sat')

  @parameterized.parameters(
      (1, 1),
      (-1, 0)
  )
  def test_z3_relu(self, input_val, true_val):
    z3_var = z3.Int('var')
    solver = z3.Solver()
    solver.add(utils.z3_relu(z3_var) == true_val)
    solver.add(z3_var == input_val)
    self.assertEqual(str(solver.check()), 'sat')

  def test_smt_forward_equals_nn_forward(self):
    weights = [np.random.random_sample((3, 3)), np.random.random_sample((1, 3))]
    biases = [np.random.random_sample(3), np.random.random_sample(1)]
    activations = ['relu', 'linear']
    z3_var = z3.Int('var')
    nn_output, _ = utils.nn_forward(np.ones(3), weights, biases, activations)
    smt_output, _ = utils.smt_forward(
        [z3.ToReal(z3_var), z3.ToReal(z3_var), z3.ToReal(z3_var)],
        weights, biases, activations)
    self._assert_z3_constraint_sat(
        constraint=z3.And(smt_output[0] - nn_output[0] < 1e-4,
                          smt_output[0] - nn_output[0] > -1e-4), z3_var=z3_var)

  def test_nn_forward(self):
    weights = [np.asarray([[1, 1], [-1, -1]]), np.asarray([[1, 1]])]
    biases = [np.asarray([0, 0]), np.asarray([-1])]
    activations = ['relu', 'linear']
    output, hidden_weighted_sum = utils.nn_forward(
        np.ones(2), weights, biases, activations)
    self.assertLen(hidden_weighted_sum, 2)
    np.testing.assert_allclose(output, [1.])
    np.testing.assert_allclose(hidden_weighted_sum[0], [2., -2.])
    np.testing.assert_allclose(hidden_weighted_sum[1], [1.])

  def test_smt_forward(self):
    z3_var = z3.Int('var')
    weights = [np.asarray([[1, 1], [-1, -1]]), np.asarray([[1, 1]])]
    biases = [np.asarray([0, 0]), np.asarray([-1])]
    activations = ['relu', 'linear']
    output, hidden_weighted_sum = utils.smt_forward(
        [z3_var, z3_var], weights, biases, activations)
    self.assertLen(output, 1)
    self.assertLen(hidden_weighted_sum, 2)
    self._assert_z3_constraint_sat(constraint=output[0] == 1, z3_var=z3_var)
    self._assert_z3_constraint_sat(
        constraint=hidden_weighted_sum[0][0] == 2, z3_var=z3_var)
    self._assert_z3_constraint_sat(
        constraint=hidden_weighted_sum[0][1] == -2, z3_var=z3_var)
    self._assert_z3_constraint_sat(
        constraint=hidden_weighted_sum[1][0] == 1, z3_var=z3_var)

  def test_smt_forward_invalid_input_1(self):
    z3_var = z3.Int('var')
    weights = [np.asarray([[1, 1], [-1, -1]]), np.asarray([[1, 1]])]
    biases = [np.asarray([0, 0]), np.asarray([-1])]
    activations = ['relu', 'linear']
    with self.assertRaisesRegex(
        ValueError, 'Lengths of weights, biases and activations should be the '
        'same, but got weights with length 2 biases with length 2 '
        'activations with length 1'):
      utils.smt_forward(
          features=[z3_var, z3_var],
          weights=weights,
          biases=biases,
          activations=activations[:-1])

  def test_smt_forward_invalid_input_2(self):
    z3_var = z3.Int('var')
    weights = [np.asarray([[1, 1], [-1, -1]]), np.asarray([[1, 1]])]
    biases = [np.asarray([0, 0]), np.asarray([-1])]
    activations = ['relu', 'linear']
    with self.assertRaisesRegex(
        ValueError, 'Lengths of weights, biases and activations should be the '
        'same, but got weights with length 2 biases with length 2 '
        'activations with length 1'):
      utils.smt_forward(
          features=[z3_var, z3_var],
          weights=weights,
          biases=biases,
          activations=activations[:-1])

  def test_smt_forward_invalid_input_3(self):
    z3_var = z3.Int('var')
    weights = [np.asarray([[1, 1], [-1, -1]]), np.asarray([[1, 1]])]
    biases = [np.asarray([0, 0]), np.asarray([-1])]
    activations = ['relu', 'linear']
    with self.assertRaisesRegex(
        ValueError, 'Lengths of weights, biases and activations should be the '
        'same, but got weights with length 2 biases with length 2 '
        'activations with length 1'):
      utils.smt_forward(
          features=[z3_var, z3_var],
          weights=weights,
          biases=biases,
          activations=activations[:-1])

  def test_nn_forward_invalid_input_1(self):
    weights = [np.asarray([[1, 1], [-1, -1]]), np.asarray([[1, 1]])]
    biases = [np.asarray([0, 0]), np.asarray([-1])]
    activations = ['relu', 'linear']
    with self.assertRaisesRegex(
        ValueError, 'Lengths of weights, biases and activations should be the '
        'same, but got weights with length 1 biases with length 2 '
        'activations with length 2'):
      utils.smt_forward(
          features=np.ones(2),
          weights=weights[:-1],
          biases=biases,
          activations=activations)

  def test_nn_forward_invalid_input_2(self):
    weights = [np.asarray([[1, 1], [-1, -1]]), np.asarray([[1, 1]])]
    biases = [np.asarray([0, 0]), np.asarray([-1])]
    activations = ['relu', 'linear']
    with self.assertRaisesRegex(
        ValueError, 'Lengths of weights, biases and activations should be the '
        'same, but got weights with length 2 biases with length 1 '
        'activations with length 2'):
      utils.smt_forward(
          features=np.ones(2),
          weights=weights,
          biases=biases[:-1],
          activations=activations)

  def test_nn_forward_invalid_input_3(self):
    weights = [np.asarray([[1, 1], [-1, -1]]), np.asarray([[1, 1]])]
    biases = [np.asarray([0, 0]), np.asarray([-1])]
    activations = ['relu', 'linear']
    with self.assertRaisesRegex(
        ValueError, 'Lengths of weights, biases and activations should be the '
        'same, but got weights with length 2 biases with length 2 '
        'activations with length 1'):
      utils.smt_forward(
          features=np.ones(2),
          weights=weights,
          biases=biases,
          activations=activations[:-1])

  @parameterized.parameters(
      (0, (0, 0)),
      (1, (0, 1)),
      (2, (1, 0)),
      (3, (1, 1)),
  )
  def test_convert_pixel_to_2d_indices(
      self, flattened_pixel_index, expected_indices):
    self.assertEqual(utils.convert_pixel_to_2d_indices(
        edge_length=2, flattened_pixel_index=flattened_pixel_index),
                     expected_indices)

  @parameterized.parameters(
      # 1D array is some array of length 4
      # 2D array:
      # [[0., 0., 1., 1.],
      #  [0., 0., 1., 1.],
      #  [2., 2., 3., 3.],
      #  [2., 2., 3., 3.]]
      # with 0, 1, 2, 3 representing the indices of the 1D array.
      (0, 0),
      (1, 0),
      (2, 1),
      (3, 1),
      (4, 0),
      (5, 0),
      (6, 1),
      (7, 1),
      (8, 2),
      (9, 2),
      (10, 3),
      (11, 3),
      (12, 2),
      (13, 2),
      (14, 3),
      (15, 3),
  )
  def test_convert_pixel_to_mask_index(
      self, flattened_pixel_index, expected_mask_index):
    self.assertEqual(
        utils.convert_pixel_to_mask_index(
            edge_length=4,
            window_size=2,
            flattened_pixel_index=flattened_pixel_index),
        expected_mask_index)

  def test_zero_pad(self):
    image = [[1., 2., 3.],
             [4., 5., 6.]]

    np.testing.assert_allclose(
        utils.zero_pad(activation_map=image, padding=(2, 3)),
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1., 2., 3., 0, 0, 0],
         [0, 0, 4., 5., 6., 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

  def test_dot_product(self):
    kernel = np.random.rand(3, 3)

    self.assertAlmostEqual(
        utils.dot_product(
            input_activation_map=np.ones((4, 4)),
            input_activation_map_row=0,
            input_activation_map_column=0,
            sliced_kernel=kernel),
        np.sum(kernel))

  def test_dot_product_text(self):
    kernel = np.random.rand(3, 5)

    self.assertAlmostEqual(
        utils.dot_product(
            input_activation_map=np.ones((5, 5)),
            input_activation_map_row=0,
            input_activation_map_column=0,
            sliced_kernel=kernel),
        np.sum(kernel))

  def test_flatten_nested_lists(self):
    image_edge_length = 2
    activation_maps = np.random.rand(3, image_edge_length, image_edge_length)

    flattened_convolutions = utils.flatten_nested_lists(
        activation_maps=activation_maps)

    np.testing.assert_allclose(
        np.asarray(flattened_convolutions), activation_maps.reshape(-1))

  def test_smt_convolution_invalid_input_kernel_input_channels(self):
    with self.assertRaisesRegex(
        ValueError,
        'Input channels in inputs and kernels are not equal. Number of input '
        'channels in input: 3 and kernels: 30'):
      utils.smt_convolution(
          input_activation_maps=np.random.rand(3, 10, 10),
          kernels=np.random.rand(1, 1, 30, 5),
          kernel_biases=np.zeros(5),
          padding=1,
          strides=1)

  def test_smt_convolution_invalid_padding(self):
    with self.assertRaisesRegex(
        ValueError,
        'Padding should be a tuple with 2 dimensions. Input padding: 1'):
      utils.smt_convolution(
          input_activation_maps=np.random.rand(3, 10, 10),
          kernels=np.random.rand(1, 1, 3, 5),
          kernel_biases=np.zeros(5),
          padding=1,
          strides=1)

  def test_smt_convolution_invalid_input_bias_output_channels(self):
    with self.assertRaisesRegex(
        ValueError,
        'Output channels in kernels and biases are not equal. Number of output '
        'channels in kernels: 5 and biases: 50'):
      utils.smt_convolution(
          input_activation_maps=np.random.rand(3, 10, 10),
          kernels=np.random.rand(1, 1, 3, 5),
          kernel_biases=np.zeros(50),
          padding=(1, 1),
          strides=1)

  def test_restore(self):
    # Create a test model.
    test_model_path = os.path.join(
        tempfile.mkdtemp(dir=FLAGS.test_tmpdir), 'checkpoint')
    tf.keras.backend.set_learning_phase(0)
    test_model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=3, input_shape=(3,), activation='relu')])
    test_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])
    # The model saved using simple_save is used to create the frozen_graph
    tf.saved_model.simple_save(
        session=tf.keras.backend.get_session(),
        export_dir=test_model_path,
        inputs={'input': test_model.inputs[0]},
        outputs={'output': test_model.outputs[0]})

    session = utils.restore_model(model_path=os.path.join(test_model_path))
    np.testing.assert_allclose(
        session.run('dense/BiasAdd:0',
                    feed_dict={'dense_input:0': np.ones((1, 3))}).shape, (1, 3))
    session.close()

  def test_calculate_auc_score(self):
    self.assertAlmostEqual(
        utils.calculate_auc_score([0, 1, 1, 0], [0, 0.3, 0.2, 1]), 0.5)

  def test_calculate_max_f1_score(self):
    self.assertAlmostEqual(
        utils.calculate_max_f1_score([0, 1, 1, 0], [0, 0.3, 0.2, 1]), 0.8)

  def test_calculate_min_mae_score(self):
    self.assertAlmostEqual(
        # A threshold below 0.3 binarizes [0, 0.3, 0.2, 1] as [0, 1, 1, 1].
        # Hence, the MAE score = (0 + 0 + 0 + 1) / 4 = 0.25. The function
        # computes multiple MAE scores and returns the best solution it finds.
        utils.calculate_min_mae_score([0, 1, 1, 0], [0, 0.3, 0.2, 1]), 0.25)

  def test_get_tightest_crop(self):
    crop_params, cropped_mask = utils._get_tightest_crop(
        saliency_map=np.asarray([[0, 0, 0, -1e-6],
                                 [0, 1, 2, -1e-6],
                                 [0, 0, 0, -1e-6]]), threshold=0)
    np.testing.assert_allclose(cropped_mask,
                               [[0, 0, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 0, 0]])
    self.assertEqual(crop_params['top'], 1)
    self.assertEqual(crop_params['bottom'], 2)
    self.assertEqual(crop_params['left'], 1)
    self.assertEqual(crop_params['right'], 3)

  def test_check_dimensions_invalid_image(self):
    with self.assertRaisesRegex(
        ValueError,
        'Image should have 3 dimensions. '
        r'Shape of the supplied image: \(2, 2\)'):
      utils._check_dimensions(
          image=np.ones((2, 2)).astype(np.float),
          saliency_map=np.ones((2, 2)).astype(np.float),
          model_type='cnn')

  def test_check_dimensions_invalid_saliency_map(self):
    with self.assertRaisesRegex(
        ValueError,
        'Saliency map should have 2 dimensions. '
        r'Shape of the supplied Saliency map: \(2, 2, 3\)'):
      utils._check_dimensions(
          image=np.ones((2, 2, 2)).astype(np.float),
          saliency_map=np.ones((2, 2, 3)).astype(np.float),
          model_type='cnn')

  def test_check_dimensions_invalid_text(self):
    with self.assertRaisesRegex(
        ValueError,
        'The text input should be a 1D numpy array. '
        r'Shape of the supplied image: \(2, 2\)'):
      utils._check_dimensions(
          image=np.ones((2, 2)).astype(np.float),
          saliency_map=np.ones(10).astype(np.float),
          model_type='text_cnn')

  def test_check_dimensions_invalid_saliency_map_text(self):
    with self.assertRaisesRegex(
        ValueError,
        'The text saliency map should be a 1D numpy array. '
        r'Shape of the supplied Saliency map: \(2, 2, 3\)'):
      utils._check_dimensions(
          image=np.ones(10).astype(np.float),
          saliency_map=np.ones((2, 2, 3)).astype(np.float),
          model_type='text_cnn')

  def test_calculate_saliency_score_valid(self):
    mock_session = mock.MagicMock()
    mock_session.run.return_value = {'softmax': (np.ones(2), np.ones(2))}
    with mock.patch.object(
        utils, 'restore_model', return_value=mock_session):
      output = utils.calculate_saliency_score(
          run_params=mock.MagicMock(),
          image=np.ones((2, 2, 2)).astype(np.float),
          saliency_map=np.random.rand(2, 2),
          area_threshold=0.05)
    self.assertCountEqual(
        output.keys(),
        ['true_label', 'true_confidence', 'cropped_label', 'cropped_confidence',
         'crop_mask', 'saliency_map', 'image', 'saliency_score'])

  def test_calculate_saliency_score_invalid(self):
    mock_session = mock.MagicMock()
    mock_session.run.return_value = {'softmax': (np.ones(2), np.ones(2))}
    with mock.patch.object(
        utils, 'restore_model', return_value=mock_session):
      output = utils.calculate_saliency_score(
          run_params=mock.MagicMock(),
          image=np.ones((2, 2, 2)),
          saliency_map=np.zeros((2, 2)),
          area_threshold=0.05)
    self.assertIsNone(output)

  def test_brute_force_fast_saliency_evaluate_masks(self):
    mock_session = mock.MagicMock()
    mock_session.run.return_value = {'softmax': (np.ones(2), np.ones(2))}

    with mock.patch.object(
        utils, 'restore_model', return_value=mock_session):
      output = utils.brute_force_fast_saliency_evaluate_masks(
          run_params=mock.MagicMock(),
          image=np.ones((2, 2, 3)),
          grid_size=2,
          area_threshold=0.05)

    self.assertCountEqual(
        output.keys(),
        ['true_label', 'true_confidence', 'cropped_label', 'cropped_confidence',
         'crop_mask', 'saliency_map', 'image', 'saliency_score'])

  def test_evaluate_cropped_image(self):
    mock_session = mock.MagicMock()
    mock_session.run.return_value = {'softmax': (np.ones(2), np.ones(2))}
    mock_run_params = mock.MagicMock()
    mock_run_params.model_type = 'cnn'

    output = utils._evaluate_cropped_image(
        session=mock_session,
        run_params=mock_run_params,
        crop_mask=np.asarray([[1, 1], [0, 0]]).astype(np.float),
        image=np.ones((2, 2)).astype(np.float),
        processed_image=np.ones((2, 2)).astype(np.float),
        saliency_map=np.ones((2, 2)).astype(np.float),
        area_threshold=0)

    self.assertCountEqual(
        output.keys(),
        ['true_label', 'true_confidence', 'cropped_label', 'cropped_confidence',
         'crop_mask', 'saliency_map', 'image', 'saliency_score'])
    # sparsity of the mask = 0.5, confidence of the classifier = 1
    # => saliency_score = log(0.5) - log(1)
    self.assertAlmostEqual(output['saliency_score'], np.log(0.5))

  def test_calculate_saliency_score_text(self):
    mock_session = mock.MagicMock()
    mock_session.run.return_value = {'softmax': (np.ones(2), np.ones(2))}
    mock_run_params = mock.MagicMock()
    mock_run_params.model_type = 'text_cnn'

    with mock.patch.object(
        utils, 'restore_model', return_value=mock_session):
      output = utils.calculate_saliency_score(
          run_params=mock_run_params,
          image=np.asarray([0, 1, 1, 1]).astype(np.float),
          saliency_map=np.asarray([0, 0, 1, 1]).astype(np.float),
          area_threshold=0)
    self.assertCountEqual(
        output.keys(),
        ['true_label', 'true_confidence', 'cropped_label', 'cropped_confidence',
         'crop_mask', 'saliency_map', 'image', 'saliency_score'])

  def test_evaluate_cropped_image_text(self):
    mock_session = mock.MagicMock()
    mock_session.run.return_value = {'softmax': (np.ones(2), np.ones(2))}
    mock_run_params = mock.MagicMock()
    mock_run_params.model_type = 'text_cnn'

    output = utils._evaluate_cropped_image(
        session=mock_session,
        run_params=mock_run_params,
        crop_mask=np.asarray([0, 0, 1, 1]).astype(np.float),
        image=np.asarray([0, 1, 1, 1]).astype(np.float),
        processed_image=np.asarray([0, 0, 1, 1]).astype(np.float),
        saliency_map=np.asarray([0, 0, 1, 1]).astype(np.float),
        area_threshold=0)

    self.assertCountEqual(
        output.keys(),
        ['true_label', 'true_confidence', 'cropped_label', 'cropped_confidence',
         'crop_mask', 'saliency_map', 'image', 'saliency_score'])
    # sparsity of the mask = 2 / 3, confidence of the classifier = 1
    # => saliency_score = log(2 / 3) - log(1)
    self.assertAlmostEqual(output['saliency_score'], np.log(2 / 3))

  def test_verify_saliency_map_shape(self):
    with self.assertRaisesRegex(ValueError,
                                'The saliency map should be a 2D numpy array '
                                r'but the received shape is \(2, 2, 2\)'):
      utils._verify_saliency_map_shape(np.ones((2, 2, 2)))

  def test_normalize_array(self):
    np.testing.assert_allclose(
        utils.normalize_array(
            array=np.asarray([[1, 5], [5, 9]]), percentile=50),
        [[0, 1], [1, 2]])

  @parameterized.parameters('ig', 'smug')
  def test_scale_saliency_map(self, method):
    saliency_map = utils.scale_saliency_map(
        saliency_map=np.random.random((2, 2)), method=method)

    np.testing.assert_allclose(saliency_map.shape, (2, 2))


class SmtConvolutionTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(1, 3, 10)
  def test_smt_convolution(self, input_activation_map_channels):
    activation_map_size = 224
    image = np.random.rand(
        activation_map_size, activation_map_size, input_activation_map_channels)
    kernels = np.random.rand(7, 7, input_activation_map_channels, 2)

    with self.test_session() as sess:
      tf_convolution = sess.run(
          tf.nn.convolution(
              input=np.expand_dims(image, axis=0),
              filter=kernels,
              strides=2,
              padding='SAME',
              data_format=None,
              dilations=None,
              name=None))

    np.testing.assert_allclose(
        np.moveaxis(tf_convolution, -1, 0).reshape(-1),
        utils.flatten_nested_lists(
            utils.smt_convolution(
                input_activation_maps=np.moveaxis(np.copy(image), -1, 0),
                kernels=kernels,
                kernel_biases=np.zeros(kernels.shape[-1]),
                padding=(2, 3),
                strides=2)))

  @parameterized.parameters('train', 'test')
  def test_get_mnist_dataset(self, split):
    with self.test_session():
      # This function has to be called in a separate session or else it throws
      # the following error - Cannot add function 'cond_png_false_94'
      # because a different function with the same name already exists.
      data = utils.get_mnist_dataset(num_datapoints=1, split=split)
    self.assertLen(data['images'], 1)
    self.assertLen(data['image_ids'], 1)
    self.assertLen(data['labels'], 1)
    np.testing.assert_allclose(data['images'][0].shape, (28, 28, 1))
    self.assertEqual(data['image_ids'][0], 0)
    self.assertLessEqual(np.max(data['images'][0]), 1.0)
    self.assertGreaterEqual(np.min(data['images'][0]), 0.0)

if __name__ == '__main__':
  absltest.main()
