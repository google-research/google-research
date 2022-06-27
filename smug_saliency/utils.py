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

"""Functions for the forward pass (symbolic and decimal) of a neural network.

Given an image and a trained neural network this code does an smt encoding of
the forward pass of the neural network and further, employs z3 solver to
learn a mask for the inputs given the weights.
"""
import collections
import io
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage.draw as draw
import sklearn.metrics as metrics
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import z3


tf.disable_eager_execution()


class OptimizerBase:
  """Creates a solver by using z3 solver.

  Attributes:
    z3_mask: list, contains mask bits as z3 vars.
    mask_sum: z3.ExprRef, sum of boolean mask bits.
    minimal_mask_sum: int, the minimum value of mask_sum which satisfying the
        smt constraints.
    solver: z3.Optimize, minimizes a mask_sum wrt smt constraints.

  Subclasses should define the generate_mask method.
  """

  def __init__(self, z3_mask):
    """Initializer.

    Args:
      z3_mask: list, contains mask bits as z3 vars.
    """
    self.z3_mask = z3_mask
    self.mask_sum = 0
    self.solver = z3.Optimize()
    for mask in self.z3_mask:
      self.solver.add(z3.Or(mask == 1, mask == 0))
      self.mask_sum += mask
    self.minimal_mask_sum = self.solver.minimize(self.mask_sum)

  def _optimize(self):
    """Solves the SMT constraints and returns the solution as a numpy array.

    Returns:
      z3_mask: float numpy array with shape (num_mask_variables,).
      result: string, returns one of the following: 'sat', 'unsat' or 'unknown'.
    """
    result = str(self.solver.check())
    z3_mask = np.zeros(len(self.z3_mask))
    if result != 'unknown':
      z3_assignment = self.solver.model()
      for var in z3_assignment.decls():
        z3_mask[int(str(var).split('_')[1])] = int(str(z3_assignment[var]))

      # Block the currently found solution so that for every call of optimize,
      # a unique mask is found.
      block = [var() != z3_assignment[var] for var in z3_assignment]
      self.solver.add(z3.Or(block))
    return z3_mask, result

  def generate_mask(self):
    """Constructs the mask with the same shape as that of data.

    Returns:
      mask: float numpy array.
      result: string, returns one of the following: 'sat', 'unsat' or 'unknown'.
    """
    raise NotImplementedError('Must be implemented by subclass.')

  def generator(self, num_unique_solutions):
    """Generates solutions from the optimizer.

    If the number of unique solutions is smaller than num_unique_solutions,
    the rest of the solutions are unsat.

    Args:
      num_unique_solutions: int, number of unique solutions you want to sample.

    Yields:
      mask: float numpy array.
      result: string, returns one of the following: 'sat', 'unsat' or 'unknown'.
    """
    for _ in range(num_unique_solutions):
      yield self.generate_mask()


class TextOptimizer(OptimizerBase):
  """Creates a solver for text by using z3 solver.
  """

  def __init__(self, z3_mask):
    """Initializer.

    Args:
      z3_mask: list, contains mask bits as z3 vars.
    """
    super().__init__(z3_mask=z3_mask)

  def generate_mask(self):
    """Constructs the mask with the same shape as that of data.

    Returns:
      mask: float numpy array with shape (num_mask_variables,).
      result: string, returns one of the following: 'sat', 'unsat' or 'unknown'.
    """
    # This method explicitly requires a masking variable for each input word
    # to the neural network. If a mask bit covers multiple words, then the
    # function has to be appropriately modified.
    return self._optimize()


class ImageOptimizer(OptimizerBase):
  """Creates a solver by using z3 solver.

  Attributes:
    edge_length: int, side length of the 2D array (image) whose pixels are to
        be masked.
    window_size: int, side length of the square mask.
  """

  def __init__(self, z3_mask, window_size, edge_length):
    """Initializer.

    Args:
      z3_mask: list, contains mask bits as z3 vars.
      window_size: int, side length of the square mask.
      edge_length: int, side length of the 2D array (image) whose pixels are to
          be masked.
    """
    super().__init__(z3_mask=z3_mask)
    self.edge_length = edge_length
    self.window_size = window_size

  def generate_mask(self):
    """Constructs a 2D mask with the same shape as that of image.

    Returns:
      mask: float numpy array with shape (edge_length, edge_length).
      result: string, returns one of the following: 'sat', 'unsat' or 'unknown'.
    """
    z3_mask, result = self._optimize()
    mask = np.zeros((self.edge_length, self.edge_length))
    num_masks_along_row = math.ceil(self.edge_length / self.window_size)
    for row in range(self.edge_length):
      for column in range(self.edge_length):
        mask_id = (
            num_masks_along_row * (row // self.window_size)) + (
                column // self.window_size)
        mask[row][column] = z3_mask[mask_id]
    return mask, result


def restore_model(model_path):
  """Restores a frozen tensorflow model into a tf session and returns it.

  Args:
    model_path: string, path to a tensorflow frozen graph.

  Returns:
    A tensorflow session.
  """
  session = tf.Session()
  tf.saved_model.loader.load(session, ['serve'], model_path)
  return session


def zero_pad(activation_map, padding):
  """Appends vectors of zeros on all the 4 sides of the image.

  Args:
    activation_map: list of list of z3.ExprRef, activation map to be 0-padded.
    padding: tuple, number of layers 0 padded vectors on top / left side of the
        image, number of layers 0 padded vectors on bottom / right side of the
        image.

  Returns:
    list of list of z3.ExrRef, 0 padded activation map.
  """
  num_rows = len(activation_map)
  num_columns = len(activation_map[0])

  # padded_activation_map has a shape - (num_padded_rows, num_padded_columns)
  padded_activation_map = []
  for _ in range(num_rows + padding[0] + padding[1]):
    padded_activation_map.append([0] * (num_columns + padding[0] + padding[1]))

  for i in range(num_rows):
    for j in range(num_columns):
      padded_activation_map[padding[0] + i][padding[0] +
                                            j] = activation_map[i][j]
  return padded_activation_map


def dot_product(input_activation_map, input_activation_map_row,
                input_activation_map_column, sliced_kernel):
  """Convolution operation for a convolution kernel and a patch in the image.

  Performs convolution on a square patch of the input_activation_map with
  (input_activation_map_row, input_activation_map_column) and
  (input_activation_map_row + kernel_rows - 1,
   input_activation_map_column + kernel_columns - 1) as the diagonal vertices.

  Args:
    input_activation_map: list of list of z3.ExprRef with dimensions
        (input_activation_map_size, input_activation_map_size).
    input_activation_map_row: int, row in the activation map for which the
        convolution is being performed.
    input_activation_map_column: int, column in the activation map for which
        convolution is being performed.
    sliced_kernel: numpy array with shape (kernel_rows, kernel_columns),
        2d slice of a kernel along input_channel.

  Returns:
    z3.ExprRef, dot product of the convolution kernel and a patch in the image.
  """
  convolution = 0
  for i in range(sliced_kernel.shape[0]):
    for j in range(sliced_kernel.shape[1]):
      convolution += (
          input_activation_map
          [input_activation_map_row + i][input_activation_map_column + j]
          * sliced_kernel[i][j])
  return convolution


def smt_convolution(input_activation_maps, kernels, kernel_biases, padding,
                    strides):
  """Performs convolution on symbolic inputs.

  Args:
    input_activation_maps: list of list of z3.ExprRef with dimensions
        (input_channels, input_activation_map_size, input_activation_map_size),
        input activation maps.
    kernels: numpy array with shape
        (kernel_size, kernel_size, input_channels, output_channels),
        weights of the convolution layer.
    kernel_biases: numpy array with shape (output_channels,), biases of the
        convolution layer.
    padding: tuple, number of layers 0 padded vectors on top/left side of the
        image.
    strides: int, number of pixel shifts over the input matrix.

  Returns:
    list of list of list of z3.ExprRef with dimensions (output_channels,
        output_activation_map_size, output_activation_map_size), convolutions.

  Raises:
    ValueError: If input_channels is inconsistent across
        input_activation_maps and kernels, or output_channels is inconsistent
        across kernels and kernel_biases, or padding is not a tuple, or padding
        isn't a tuple of size 2.
  """
  if len(input_activation_maps) != kernels.shape[2]:
    raise ValueError(
        'Input channels in inputs and kernels are not equal. Number of input '
        'channels in input: %d and kernels: %d' % (
            len(input_activation_maps), kernels.shape[2]))
  if not isinstance(padding, tuple) or len(padding) != 2:
    raise ValueError(
        'Padding should be a tuple with 2 dimensions. Input padding: %s' %
        padding)
  if kernels.shape[3] != kernel_biases.shape[0]:
    raise ValueError(
        'Output channels in kernels and biases are not equal. Number of output '
        'channels in kernels: %d and biases: %d' % (
            kernels.shape[3], kernel_biases.shape[0]))
  padded_input_activation_maps = []

  # reshape the kernels to
  # (output_channels, kernel_size, kernel_size, input_channels)
  kernels = np.moveaxis(kernels, -1, 0)
  for input_activation_map in input_activation_maps:
    padded_input_activation_maps.append(
        zero_pad(
            # (input_activation_map_size, input_activation_map_size)
            activation_map=input_activation_map,
            padding=padding))
  output_activation_maps = []
  output_activation_map_size = len(input_activation_maps[0]) // strides
  # Iterate over output_channels.
  for kernel, kernel_bias in zip(kernels, kernel_biases):
    output_activation_map = np.full(
        (output_activation_map_size, output_activation_map_size),
        kernel_bias).tolist()
    for i in range(output_activation_map_size):
      for j in range(output_activation_map_size):
        for channel_in in range(kernel.shape[-1]):
          output_activation_map[i][j] += dot_product(
              input_activation_map=padded_input_activation_maps[channel_in],
              input_activation_map_row=strides * i,
              input_activation_map_column=strides * j,
              sliced_kernel=kernel[:, :, channel_in])
    output_activation_maps.append(output_activation_map)
  return output_activation_maps


def flatten_nested_lists(activation_maps):
  """Flattens a nested list of depth 3 in a row major order.

  Args:
    activation_maps: list of list of list of z3.ExprRef with dimensions
        (channels, activation_map_size, activation_map_size), activation_maps.

  Returns:
    list of z3.ExprRef.
  """
  flattened_activation_maps = []
  for activation_map in activation_maps:
    for activation_map_row in activation_map:
      flattened_activation_maps.extend(activation_map_row)
  return flattened_activation_maps


def z3_relu(x):
  """Relu activation function.

  max(0, x).

  Args:
    x: z3.ExprRef, z3 Expression.

  Returns:
    z3.ExprRef.
  """
  return z3.If(x > 0, x, 0)


def _verify_lengths(weights, biases, activations):
  """Verifies the lengths of the weights, biases, and activations are equal.

  Args:
    weights: list of float numpy array with shape (output_dim, input_dim) and
        length num_layers, weights of the neural network.
    biases: list of float numpy array with shape (output_dim,) and length
        num_layers, biases of the neural network.
    activations: list of string with length num_layers, activations for each
        hidden layer.

  Raises:
    ValueError: If lengths of weights, biases, and activations are not equal.
  """
  if not len(weights) == len(biases) == len(activations):
    raise ValueError('Lengths of weights, biases and activations should be the '
                     'same, but got weights with length %d biases with length '
                     '%d activations with length %d' % (
                         len(weights), len(biases), len(activations)))


def smt_forward(features, weights, biases, activations):
  """Forward pass of a neural network with the inputs being symbolic.

  Computes the forward pass of a neural network by looping through the weights
  and the biases in a layerwise manner.

  Args:
    features: list of z3.ExprRef, contains a z3 instance corresponding
        to each pixel of a flattened image.
    weights: list of float numpy array with shape (output_dim, input_dim) and
        length num_layers, weights of the neural network.
    biases: list of float numpy array with shape (output_dim,) and length
        num_layers, biases of the neural network.
    activations: list of string with length num_layers, activations for each
        hidden layer.

  Returns:
    logits: list of z3.ExprRef, output logits.
    hidden_nodes: list of list of list of z3.ExprRef with dimensions
        (num_layers, output_dim, input_dim),
        weighted sum at every hidden neuron.
  """
  _verify_lengths(weights, biases, activations)
  layer_features = [i for i in features]
  hidden_nodes = []
  for layer_weights, layer_bias, layer_activation in zip(
      weights, biases, activations):
    # Values of hidden nodes after activation.
    layer_output = []
    # Values of hidden nodes before activation.
    layer_weighted_sums = []
    for weight_row, bias in zip(layer_weights, layer_bias):
      # Iterating over output_dim
      intermediate_sum = bias
      for x, weight in zip(layer_features, weight_row):
        # Iterating over input_dim
        intermediate_sum += weight * x
      layer_weighted_sums.append(intermediate_sum)
      # Apply relu or linear activation function
      if layer_activation == 'relu':
        layer_output.append(z3_relu(intermediate_sum))
      else:
        layer_output.append(intermediate_sum)
    hidden_nodes.append(layer_weighted_sums)
    layer_features = layer_output
  return layer_features, hidden_nodes


def nn_forward(features, weights, biases, activations):
  """Forward pass of a neural network using matrix multiplication.

  Computes the forward pas of a neural network using matrix multiplication and
  addition by looping through the weights and the biases.

  Args:
    features: float numpy array with shape (num_input_features,),
        image flattened as a 1D vector.
    weights: list of float numpy array with shape (output_dim, input_dim) and
        length num_layers, weights of the neural network .
    biases: list of float numpy array with shape (output_dim,) and length
        num_layers, biases of the neural network.
    activations: list of strings with length num_layers,
        activations for each hidden layer.

  Returns:
    logits: float numpy array with shape (num_labels,).
    hidden_nodes: list of numpy array with shape (output_dim,) and
        length num_layers.
  """
  _verify_lengths(weights, biases, activations)
  hidden_nodes = []
  layer_features = np.copy(features)
  for layer_weights, layer_bias, layer_activation in zip(
      weights, biases, activations):
    layer_output = np.matmul(
        layer_features, layer_weights.transpose()) + layer_bias
    hidden_nodes.append(layer_output)
    if layer_activation == 'relu':
      layer_output = layer_output * (layer_output > 0)
    layer_features = layer_output
  return layer_features, hidden_nodes


def convert_pixel_to_2d_indices(edge_length, flattened_pixel_index):
  """Maps an index of an array to its reshaped 2D matrix's rows and columns.

  This function maps the index of an array with length edge_length ** 2 to the
  rows and columns of its reshaped 2D matrix with shape
  (edge_length, edge_length).

  Args:
    edge_length: int, side length of the 2D array (image) whose pixels are to be
        masked.
    flattened_pixel_index: int, flattened pixel index in the image in
        a row major order.
  Returns:
    row_index: int, row index of the 2D array
    column_index: int, column index of the 2D array
  """
  return (
      flattened_pixel_index // edge_length, flattened_pixel_index % edge_length)


def convert_pixel_to_mask_index(
    edge_length, window_size, flattened_pixel_index):
  """Maps flattened pixel index to the flattened index of its mask.

  Args:
    edge_length: int, side length of the 2D array (image).
    window_size: int, side length of the square mask.
    flattened_pixel_index: int, flattened pixel index in the image in
        a row major order.

  Returns:
    int, the index of the mask bit in the flattened mask array.
  """
  num_masks_along_row = edge_length // window_size
  num_pixels_per_mask_row = edge_length * window_size
  return (
      num_masks_along_row * (flattened_pixel_index // num_pixels_per_mask_row)
      + (flattened_pixel_index % edge_length) // window_size)


def calculate_auc_score(ground_truth, attribution_map):
  """Calculates the auc of roc curve of the attribution map wrt ground truth.

  Args:
    ground_truth: float numpy array, ground truth values.
    attribution_map: float numpy array, attribution map.

  Returns:
    float, AUC of the roc curve.
  """
  return metrics.roc_auc_score(ground_truth, attribution_map)


def calculate_min_mae_score(ground_truth, attribution_map):
  """Calculates the mean absolute error of the attribution map wrt ground truth.

  Converts the continuous valued attribution maps to binary valued by
  choosing multiple thresholds. Entries above the threshold are set to 1 and
  below are set to 0. Then, it computes MAE for each such mask and returns
  the best score.

  Args:
    ground_truth: int numpy array, ground truth values.
    attribution_map: float numpy array, attribution map.

  Returns:
    float, the mean absolute error.
  """
  thresholds = np.unique(attribution_map)
  thresholds = np.append(
      thresholds[::max(int(round(len(thresholds) / 1000)), 1)], thresholds[-1])
  mae_score = np.inf
  for threshold in thresholds:
    thresholded_attributions = np.zeros_like(attribution_map, dtype=np.int8)
    thresholded_attributions[attribution_map >= threshold] = 1
    mae_score = min(
        mae_score,
        metrics.mean_absolute_error(ground_truth, thresholded_attributions))
  return mae_score


def calculate_max_f1_score(ground_truth, attribution_map):
  """Calculates the F1 score of the attribution map wrt the ground truth.

  Computes f1 score for a continuous valued attribution map. First,
  it computes precision and recall at multiple thresholds using
  sklearn.precision_recall_curve(). Then it computes f1 scores for each
  precision and recall score and returns the max.

  Args:
    ground_truth: int numpy array, ground truth values.
    attribution_map: float numpy array, attribution map.

  Returns:
    float, the F1 score.
  """
  precision, recall, _ = metrics.precision_recall_curve(
      ground_truth, attribution_map)
  # Sklearn's f1_score metric requires both the ground_truth and the
  # attribution_map to be binary valued. So, we compute the precision and
  # recall scores at multiple thresholds and report the best f1 score.
  return np.nanmax(list(
      map(lambda p, r: 2 * (p * r) / (p + r), precision, recall)))




def get_mnist_dataset(num_datapoints, split='test'):
  """Loads the MNIST dataset.

  Args:
    num_datapoints: int, number of images to load.
    split: str, One of {'train', 'test'} representing train and test data
      respectively.

  Returns:
    dict,
      * image_ids: list of int, the serial number of each image serialised
          accoriding to its position in the dataset.
      * labels: list of int, inception logit indices of each image.
      * images: list of float numpy array with shape (28, 28, 1),
          MNIST images with values between [0, 1].
  """
  builder = tfds.builder('mnist')
  builder.download_and_prepare()
  dataset = builder.as_dataset()
  data = collections.defaultdict(list)
  for image_id, datapoint in enumerate(tfds.as_numpy(dataset[split])):
    data['images'].append(datapoint['image'] / 255.0)
    data['labels'].append(datapoint['label'])
    data['image_ids'].append(image_id)
    if image_id == num_datapoints - 1:
      break
  return data


def _get_tightest_crop(saliency_map, threshold):
  """Finds the tightest bounding box for a given saliency map.

  For a continuous valued saliency map, finds the tightest bounding box by
  all the attributions outside the bounding box have a score less than the
  threshold.

  Args:
    saliency_map: float numpy array with shape (rows, columns), saliency map.
    threshold: float, attribution threshold.

  Returns:
    crop parameters: dict,
      * left: int, index of the left most column of the bounding box.
      * right: int, index of the right most column of the bounding box + 1.
      * top: int, index of the top most row of the bounding box.
      * bottom: int, index of the bottom most row of the bounding box + 1.
    cropped mask: int numpy array with shape (rows, columns), the values within
      the bounding set to 1.
  """
  non_zero_rows, non_zero_columns = np.asarray(
      saliency_map > threshold).nonzero()
  top = np.min(non_zero_rows)
  bottom = np.max(non_zero_rows) + 1
  left = np.min(non_zero_columns)
  right = np.max(non_zero_columns) + 1
  cropped_mask = np.zeros_like(saliency_map)
  cropped_mask[top: bottom, left: right] = 1
  return {
      'left': left,
      'right': right,
      'top': top,
      'bottom': bottom,
  }, cropped_mask


def _check_dimensions(image, saliency_map, model_type):
  """Verifies the image and saliency map dimensions have proper dimensions.

  Args:
    image: If model_type = 'cnn', float numpy array with shape (rows, columns,
      channels), image. Otherwise, float numpy array with shape
      (num_zero_padded_words,), text.
    saliency_map: If model_type = 'cnn', float numpy array with shape (rows,
      columns, channels). Otherwise, float numpy array with shape
      (num_zero_padded_words,), saliency_map.
    model_type: str, One of {'cnn', 'text_cnn'}, model type.

  Raises:
    ValueError:
      If model_type is 'text_cnn' and image isn't a 3D array or the saliency map
      isn't a 2D array. Or,
      if the model_type is 'cnn' and the image isn't a 1D array or the saliency
      map isn't a 1D array.
  """
  if model_type == 'text_cnn':
    if image.ndim != 1:
      raise ValueError('The text input should be a 1D numpy array. '
                       'Shape of the supplied image: {}'.format(image.shape))
    if saliency_map.ndim != 1:
      raise ValueError(
          'The text saliency map should be a 1D numpy array. '
          'Shape of the supplied Saliency map: {}'.format(saliency_map.shape))
  else:
    if image.ndim != 3:
      raise ValueError(
          'Image should have 3 dimensions. '
          'Shape of the supplied image: {}'.format(image.shape))
    if saliency_map.ndim != 2:
      raise ValueError(
          'Saliency map should have 2 dimensions. '
          'Shape of the supplied Saliency map: {}'.format(saliency_map.shape))


def calculate_saliency_score(
    run_params, image, saliency_map, area_threshold=0.05, session=None):
  """Computes the score for an image using the saliency metric.

  For a continuous valued saliency map, tighest bounding box is found at
  multiple threhsolds and the best score is returned.
  The saliency metric is defined as score(a, p) = log(a') - log(p),
  where a = fraction of the image area occupied by the mask,
        p = confidence of the classifier on the cropped and rescaled image.
        a' = max(area_threshold, a)
  Reference: https://arxiv.org/pdf/1705.07857.pdf

  Args:
    run_params: RunParams with model_path, model_type and tensor_names.
    image: If model_type = 'cnn', float numpy array with shape (rows, columns,
      channels) with pixel values between [0, 255], image. Otherwise, float
      numpy array with shape (num_zero_padded_words,), text.
    saliency_map: If model_type = 'cnn', float numpy array with shape (rows,
      columns, channels). Otherwise, float numpy array with shape
      (num_zero_padded_words,), saliency_map.
    area_threshold: float, area_threshold used in the metric.
    session: (default: None) tensorflow session.

  Returns:
    if a the saliency_map has all 0s returns None
    else dict,
      * true_label: int, True label of the image.
      * true_confidence: float, Confidence of the classifier on the image.
      * cropped_label: int, Predicted label of the classifier on the cropped
          image.
      * cropped_confidence: float, Confidence of the classifier on the cropped
          image for the true label.
      * crop_mask: int numpy array with shape (rows, columns), the values
          within the bounding set to 1.
      * saliency_map: float numpy array with shape (rows, columns),
          saliency map.
      * image: float numpy array with shape (rows, columns), image.
      * saliency_score: float, saliency score.
  """
  _check_dimensions(image=image, saliency_map=saliency_map,
                    model_type=run_params.model_type)
  if session is None:
    session = restore_model(run_params.model_path)
  # Sometimes the saliency map consists of all 1s. Hence, a threshold = 0
  # should be present.
  thresholds = np.append(0, np.unique(saliency_map))
  min_score = None
  record = None
  steps = max(int(round(thresholds.size / 100)), 1)
  if run_params.model_type == 'text_cnn':
    steps = 1
  for threshold in thresholds[::steps]:
    if np.sum(saliency_map > threshold) == 0:
      # A bounding box doesn't exist.
      continue
    crop_mask, processed_image = _crop_and_process_image(
        image=image,
        saliency_map=saliency_map,
        threshold=threshold,
        model_type=run_params.model_type)
    eval_record = _evaluate_cropped_image(
        session=session,
        run_params=run_params,
        crop_mask=crop_mask,
        image=image,
        processed_image=processed_image,
        saliency_map=saliency_map,
        area_threshold=area_threshold)
    if min_score is None or eval_record['saliency_score'] < min_score:
      min_score = eval_record['saliency_score']
      record = eval_record
  session.close()
  return record


def _crop_and_process_image(image, saliency_map, threshold, model_type):
  """Crops the image and returns the processed image.

  Args:
    image: If model_type = 'cnn', float numpy array with shape (rows, columns,
      channels) with pixel values between [0, 255], image. Otherwise, float
      numpy array with shape (num_zero_padded_words,), text.
    saliency_map: If model_type = 'cnn', float numpy array with shape (rows,
      columns, channels). Otherwise, float numpy array with shape
      (num_zero_padded_words,), saliency_map.
    threshold: float, saliency threshold.
    model_type: str, One of 'cnn' for image or 'text_cnn' for text input.

  Returns:
    crop_mask: If model_type = 'cnn',
        float numpy array with shape (rows, columns, channels), image.
        Otherwise,
        float numpy array with shape (num_zero_padded_words,), text.
    processed_image: If model_type = 'cnn',
        float numpy array with shape (rows, columns, channels), image.
        Otherwise,
        float numpy array with shape (num_zero_padded_words,), text.
  """
  if model_type == 'text_cnn':
    crop_mask = (saliency_map > threshold).astype(int)
    return crop_mask, saliency_map * crop_mask
  else:
    image_shape_original = (image.shape[0], image.shape[1])
    crop_params, crop_mask = _get_tightest_crop(saliency_map=saliency_map,
                                                threshold=threshold)
    cropped_image = image[crop_params['top']:crop_params['bottom'],
                          crop_params['left']:crop_params['right'], :]
    return crop_mask, np.array(
        Image.fromarray(cropped_image.astype(np.uint8)).resize(
            image_shape_original, resample=Image.BILINEAR))


def process_model_input(image, pixel_range):
  """Scales the input image's pixels to make it within pixel_range."""
  # pixel values are between [0, 1]
  image = normalize_array(image, percentile=100)
  min_pixel_value, max_pixel_value = pixel_range
  # pixel values are within pixel_range
  return image * (max_pixel_value - min_pixel_value) + min_pixel_value


def _evaluate_cropped_image(session, run_params, crop_mask, image,
                            processed_image, saliency_map, area_threshold):
  """Computes the saliency metric for a given resized image.

  Args:
    session: tf.Session, tensorflow session.
    run_params: RunParams with tensor_names and pixel_range.
    crop_mask: int numpy array with shape (rows, columns), the values within the
      bounding set to 1.
    image: If model_type = 'cnn', float numpy array with shape (rows, columns,
      channels) with pixel values between [0, 255], image. Otherwise, float
      numpy array with shape (num_zero_padded_words,), text.
    processed_image: float numpy array with shape (cropped_rows,
        cropped_columns, channels), cropped image.
    saliency_map:
      * None if brute_force_fast_saliency_evaluate_masks is using this function.
      * otherwise, float numpy array with shape (rows, columns), saliency map.
    area_threshold: float, area threshold in the saliency metric.

  Returns:
    dict,
      * true_label: int, True label of the image.
      * true_confidence: float, Confidence of the classifier on the image.
      * cropped_label: int, Predicted label of the classifier on the cropped
          image.
      * cropped_confidence: float, Confidence of the classifier on the cropped
          image for the true label.
      * crop_mask: int numpy array with shape (rows, columns), the values
          within the bounding set to 1.
      saliency_map:
        * None if brute_force_fast_saliency_evaluate_masks is using this
            function.
        * otherwise, float numpy array with shape (rows, columns), saliency map.
      * image: float numpy array with shape (rows, columns), image.
      * saliency_score: float, saliency score.
  """
  if run_params.model_type == 'cnn':
    image = process_model_input(image, run_params.pixel_range)
    processed_image = process_model_input(processed_image,
                                          run_params.pixel_range)
  true_softmax, cropped_softmax = session.run(
      run_params.tensor_names,
      feed_dict={
          run_params.tensor_names['input']: [image, processed_image]}
      )['softmax']
  true_label = np.argmax(true_softmax)
  cropped_confidence = cropped_softmax[true_label]
  if run_params.model_type == 'text_cnn':
    # Sparsity is defined as words in the mask / words in the sentence.
    # Hence, to ignore zero padding we only account for non-zero entries in the
    # input.
    sparsity = np.sum(crop_mask) / np.sum(image != 0)
  else:
    sparsity = np.sum(crop_mask) / crop_mask.size
  score = np.log(max(area_threshold, sparsity)) - np.log(cropped_confidence)
  return {
      'true_label': true_label,
      'true_confidence': np.max(true_softmax),
      'cropped_label': np.argmax(cropped_softmax),
      'cropped_confidence': cropped_confidence,
      'crop_mask': crop_mask,
      'saliency_map': saliency_map,
      'image': image,
      'saliency_score': score,
  }


def _generate_cropped_image(image, grid_size):
  """Generates crop mask and cropped images by dividing the image into a grid.

  Args:
    image: float numpy array with shape (rows, columns, channels), image.
    grid_size: int, size of the grid.

  Yields:
    crop_mask: int numpy array with shape (rows, columns), the values
        within the bounding set to 1.
    image: float numpy array with shape (cropped_rows, cropped_columns,
        channels), cropped image.
  """
  image_edge_length = image.shape[0]
  scale = image_edge_length / grid_size
  for row_top in range(grid_size):
    for column_left in range(grid_size):
      for row_bottom in range(row_top + 2, grid_size + 1):
        # row_bottom starts from row_top + 2 so that while slicing, we don't
        # end up with a null array.
        for column_right in range(column_left + 2, grid_size + 1):
          crop_mask = np.zeros((image_edge_length, image_edge_length))
          row_slice = slice(int(scale * row_top), int(scale * row_bottom))
          column_slice = slice(int(scale * column_left),
                               int(scale * column_right))
          crop_mask[row_slice, column_slice] = 1
          yield crop_mask, image[row_slice, column_slice, :]


def brute_force_fast_saliency_evaluate_masks(run_params,
                                             image,
                                             grid_size=10,
                                             area_threshold=0.05,
                                             session=None):
  """Finds the best bounding box in an image that optimizes the saliency metric.

  Divides the image into (grid_size x grid_size) grid. Then evaluates all
  possible bounding boxes formed by choosing any 2 grid points as opposite
  ends of its diagonal.

  Args:
    run_params: RunParams with model_path and tensor_names.
    image: float numpy array with shape (rows, columns, channels) and pixel
        values between [0, 255], image.
    grid_size: int, size of the grid.
    area_threshold: float, area_threshold used in the saliency metric.
    session: tf.Session, (default None) tensorflow session with the loaded
        neural network.

  Returns:
    dict,
      * true_label: int, True label of the image.
      * true_confidence: float, Confidence of the classifier on the image.
      * cropped_label: int, Predicted label of the classifier on the cropped
          image.
      * cropped_confidence: float, Confidence of the classifier on the cropped
          image for the true label.
      * crop_mask: int numpy array with shape (rows, columns), the values
          within the bounding set to 1.
      * saliency_map: None.
      * image: float numpy array with shape (rows, columns), image.
      * saliency_score: float, saliency score.
  """
  if session is None:
    session = restore_model(run_params.model_path)
  min_score = None
  for crop_mask, cropped_image in _generate_cropped_image(image, grid_size):
    eval_record = _evaluate_cropped_image(
        session=session,
        run_params=run_params,
        crop_mask=crop_mask,
        image=image,
        processed_image=np.array(
            Image.fromarray(cropped_image.astype(np.uint8)).resize(
                run_params.image_placeholder_shape[1:-1],
                resample=Image.BILINEAR)),
        saliency_map=None,
        area_threshold=area_threshold)
    if min_score is None or eval_record['saliency_score'] < min_score:
      min_score = eval_record['saliency_score']
      record = eval_record
  session.close()
  return record


def remove_ticks():
  """Removes ticks from the axes."""
  plt.tick_params(
      axis='both',  # changes apply to the x-axis
      which='both',  # both major and minor ticks are affected
      bottom=False,  # ticks along the bottom edge are off
      top=False,  # ticks along the top edge are off
      left=False,  # ticks along the left edge are off
      right=False,  # ticks along the right edge are off
      labelbottom=False,
      labelleft=False)


def show_bounding_box(mask, left_offset=0, top_offset=0, linewidth=3,
                      edgecolor='lime'):
  """Given a mask, shows the tightest rectangle capturing it.

  Args:
    mask: numpy array with shape (rows, columns), a mask.
    left_offset: int, shift the bounding box left by these many pixels.
    top_offset: int, shift the bounding box top by these many pixels.
    linewidth: int, line width the of the bounding box.
    edgecolor: string, color of the bounding box.
  """
  ax = plt.gca()
  params, _ = _get_tightest_crop(mask, 0)
  ax.add_patch(patches.Rectangle(
      (params['left'] - left_offset, params['top'] - top_offset),
      params['right'] - params['left'],
      params['bottom'] - params['top'],
      linewidth=linewidth, edgecolor=edgecolor, facecolor='none'))


def normalize_array(array, percentile=99):
  """Normalizes saliency maps for visualization.

  Args:
    array: numpy array, a saliency map.
    percentile: int, the minimum value and the value with this percentile in x
      are scaled between 0 and 1.

  Returns:
    numpy array with same shape as input array, the normalized saliency map.
  """
  return (array - array.min()) / (
      np.percentile(array, percentile) - array.min())


def _verify_saliency_map_shape(saliency_map):
  """Checks if the shape of the saliency map is a 2D array.

  Args:
    saliency_map: numpy array with shape (rows, columns), a saliency map.

  Raises:
    ValueError: If the saliency map isn't a 2D array.
  """
  if saliency_map.ndim != 2:
    raise ValueError('The saliency map should be a 2D numpy array '
                     'but the received shape is {}'.format(saliency_map.shape))


def scale_saliency_map(saliency_map, method):
  """Scales saliency maps for visualization.

  For smug and smug base the saliency map is scaled such that the positive
  scores are scaled between 0.5 and 1 (99th percentile maps to 1).
  For other methods the saliency map is scaled between 0 and 1
  (99th percentile maps to 1).

  Args:
    saliency_map: numpy array with shape (rows, columns), a saliency map.
    method: str, saliency method.

  Returns:
    numpy array with shape (rows, columns), the normalized saliency map.
  """
  _verify_saliency_map_shape(saliency_map)
  saliency_map = normalize_array(saliency_map)
  if 'smug' in method:
    # For better visualization, the smug_saliency_map and the
    # no_minimization_saliency_map are scaled between [0.5, 1] instead of the
    # usual [0, 1]. Note that doing such a scaling doesn't affect the
    # saliency score in any way as the relative ordering between the pixels
    # is preserved.
    saliency_map[saliency_map > 0] = 0.5 + 0.5 * saliency_map[saliency_map > 0]
  return saliency_map


def visualize_saliency_map(saliency_map, title=''):
  """Grayscale visualization of the saliency map.

  Args:
    saliency_map: numpy array with shape (rows, columns), a saliency map.
    title: str, title of the saliency map.
  """
  _verify_saliency_map_shape(saliency_map)
  plt.imshow(saliency_map, cmap=plt.cm.gray, vmin=0, vmax=1)
  plt.title(title)
  remove_ticks()
