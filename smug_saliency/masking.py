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

# Lint as: python3
"""Find the minimal mask for a given image and a trained neural network.

Given an image and a trained neural network this code constructs an smt encoding
of the forward pass of the neural network and further, employs z3 solver to
learn a mask for the inputs given the weights. The smt constraints can be
either formulated by constraining the activations of the first hidden node or
the final layer.
"""

import collections
import time
import numpy as np
from saliency import integrated_gradients
from saliency import xrai
import scipy.ndimage as scipy_ndimage
import tensorflow.compat.v1 as tf
import z3
from smug_saliency import utils

tf.disable_eager_execution()


class RunParams(
    collections.namedtuple(
        '_RunParams', [
            'model_type', 'model_path', 'image_placeholder_shape', 'padding',
            'strides', 'tensor_names', 'activations'
        ])):
  """Run parameters for a particular dataset and neural network model pair.

  The named tuple contains:
    model_type: str, type of the model for which a mask is being found.
    model_path: str, path to the saved model.
    image_placeholder_shape: tuple, input placeholder of the neural
      network for a single image.
    padding: tuple of length 2, takes the form (a, b) where a is the
      number of 0 padded vectors on the top and left side of the image;
      b is the number of 0 padded vectors on the bottom and right side of the
      image.
    strides: int, number of pixel shifts over the input matrix.
    tensor_names: dict,
      * input: str, name of input tensor in tf graph.
      * first_layer: str, name of the first layer pre-relu activation
          tensor in the tf graph.
      * first_layer_relu: str, name of the first layer relu activation
          tensor in the tf graph.
      * logits: str, name of the logits tensor in the tf graph.
      * softmax: str, name of the softmax tensor in the tf graph.
      * weights_layer_1: str, name of the layer 1 fc / conv weights.
      * biases_layer_1: str, name of the layer 1 biases.
      * (text only) embedding: str, name of the embedding layer.
    activations (full encoding only): list of str, activation functions of
      all the hidden layers and the final layer. Each activation takes a
      one of the following values {'relu', 'linear'}. The default value of
      activations is set to None.
  """


def _encode_input(image, z3_mask, window_size):
  """Encodes the image pixels by multiplying them with masking variables.

  Converts the pixels into z3.ExprRef by multiplying them with their
  corresponding masking variable. For an image with pixels with the same
  spatial dimensions are multiplied with the same masking variable.

  Args:
    image: float numpy array with shape
        (image_edge_length, image_edge_length, image_channels), image.
    z3_mask: list of z3.ExprRef with length (edge_length // window_size) ** 2,
        unique masking variables.
    window_size: int, side length of the square mask.

  Returns:
    list of list of list of z3.ExprRef with dimensions
        (image_channels, image_edge_length, image_edge_length), encoded input.
  """
  image_edge_length, _, image_channels = image.shape
  encoded_input = []
  for channel in range(image_channels):
    # Slicing the image across each channel
    encoded_input_per_channel = []
    for image_row in range(image_edge_length):
      encoded_input_row = []
      for image_column in range(image_edge_length):
        index = utils.convert_pixel_to_mask_index(
            edge_length=image_edge_length,
            window_size=window_size,
            flattened_pixel_index=image_row * image_edge_length + image_column)
        encoded_input_row.append(
            z3.ToReal(z3_mask[index]) * image[image_row][image_column][channel])
      encoded_input_per_channel.append(encoded_input_row)
    encoded_input.append(encoded_input_per_channel)
  return encoded_input


def _formulate_smt_constraints_final_layer(
    z3_optimizer, smt_output, delta, label_index):
  """Formulates smt constraints using the logits in the final layer.

  Generates constraints by setting the logit corresponding to label_index to be
  more than the rest of the logits by an amount delta for the forward pass of a
  masked image.

  Args:
    z3_optimizer: instance of z3.Optimizer, z3 optimizer.
    smt_output: list of z3.ExprRef with length num_output.
    delta: float, masked logit for the label is greater than the rest of the
        masked logits by delta.
    label_index: int, index of the label of the training image.

  Returns:
    z3 optimizer with added smt constraints.
  """
  for i, output in enumerate(smt_output):
    if i != label_index:
      z3_optimizer.solver.add(smt_output[label_index] - output > delta)
  return z3_optimizer


def _get_hidden_node_location(flattened_index, num_rows, num_columns):
  """Converts the flattened index of a hidden node to its index in the 3D array.

  Converts the index of a hidden node in the first convolution layer (flattened)
  into its location- row, column, and channel in the 3D activation map. The
  3D activation map has dimensions: (num_channels, num_rows, num_columns).

  Args:
    flattened_index: int, index of a hidden node in the first convolution
        layer after it is flattened.
    num_rows: int, number of rows in the activation map produced by each
        kernel.
    num_columns: int, number of columns in the activation map produced by each
        kernel.

  Returns:
    channel: int, channel number of the activation map to which the hidden node
        belongs to.
    row: int, row number of the hidden node in the activation map.
    column: int, column number of the hidden node in the activation map.
  """
  total = num_rows * num_columns
  output_activation_map_row = (flattened_index % total) // num_columns
  output_activation_map_column = (flattened_index % total) % num_columns
  return (flattened_index // total,
          output_activation_map_row,
          output_activation_map_column)


def _formulate_smt_constraints_convolution_layer(
    z3_optimizer, kernels, biases, padding, strides, gamma, chosen_indices,
    output_activation_map_shape, conv_activations, input_activation_maps):
  """Formulates the smt constraints for a convolution layer.

  Formulates the smt constraints by performing convolutions for the activations
  whose indices are specified in chosen_indices.

  Args:
    z3_optimizer: instance of z3.Optimizer, z3 optimizer.
    kernels: numpy array with shape
        (output_channels, kernel_size, kernel_size, input_channels),
        weights of the convolution layer.
    biases: numpy array with shape (output_channels,), biases of the
        convolution layer.
    padding: tuple, number of layers 0 padded vectors on top/left side of the
        image.
    strides: int, number of pixel shifts over the input matrix.
    gamma: float, masked activation is greater than gamma times the unmasked
        activation. Its value is always between [0,1).
    chosen_indices: list of int, indices (after flattening the activation maps)
        of the hidden node activations for which the minimisation is being done.
    output_activation_map_shape: tuple of length 2, shape of the activation
        map of the form (num_rows, num_columns).
    conv_activations: float numpy array, flattened convolution layer
        activations.
    input_activation_maps: list of z3.ExprRef of depth 3, padded_image.

  Returns:
    an instance of z3.Optimizer with added smt constraints.
  """
  padded_input_activation_maps = []
  for input_activation_map in input_activation_maps:
    padded_input_activation_maps.append(
        utils.zero_pad(activation_map=input_activation_map, padding=padding))
  for index in chosen_indices:
    (hidden_node_channel, hidden_node_row,
     hidden_node_column) = _get_hidden_node_location(
         flattened_index=index,
         num_rows=output_activation_map_shape[0],
         num_columns=output_activation_map_shape[1])
    # Perform convolution
    smt_equation = biases[hidden_node_channel]
    # kernels.shape[-1] represents the number of input channels in the kernel.
    for image_channel in range(kernels.shape[-1]):
      smt_equation += utils.dot_product(
          input_activation_map=padded_input_activation_maps[image_channel],
          # hidden_node_row * strides is the starting row of the convolution
          # patch in the input image. Similarly, hidden_node_column * strides
          # is the starting column of the convolution patch.
          input_activation_map_row=hidden_node_row * strides,
          input_activation_map_column=hidden_node_column * strides,
          sliced_kernel=kernels[hidden_node_channel][:, :, image_channel])

    # Add constraint to the solver
    if conv_activations[index] > 0:
      # we constrain only those nodes whose activations are positive.
      # In future we might handle nodes with negative values as well.
      z3_optimizer.solver.add(
          smt_equation > gamma * conv_activations[index])
  return z3_optimizer


def _formulate_smt_constraints_fully_connected_layer(
    z3_optimizer, nn_first_layer, smt_first_layer, top_k, gamma):
  """Formulates smt constraints using first layer activations.

  Generates constraints for the top_k nodes in the first hidden layer by setting
  the masked activation to be greater than that of the unmasked activation.

  Args:
    z3_optimizer: instance of z3.Optimizer, z3 optimizer.
    nn_first_layer: numpy array with shape (num_hidden_nodes_first_layer,)
    smt_first_layer: list of z3.ExprRef with length
        num_hidden_nodes_first_layer.
    top_k: int, constrain the nodes with top k activations in the first hidden
        layer.
    gamma: float, masked activation is greater than gamma times the unmasked
        activation. Its value is always between [0,1).

  Returns:
    z3 optimizer with added smt constraints.
  """
  for index in nn_first_layer.argsort()[-top_k:]:
    if nn_first_layer[index] > 0:
      # we constrain only those nodes whose activations are positive.
      # In future we might handle nodes with negative values as well.
      z3_optimizer.solver.add(
          smt_first_layer[index] > gamma * nn_first_layer[index])
  return z3_optimizer


def _verify_mask_dimensions(mask, model_type):
  """Checks if the mask dimensions are valid for a given model_type.

  Args:
    mask: For
      * image - numpy array with shape (image_edge_length, image_edge_length),
        binary mask of the image.
      * text - numpy array with shape (num_words,), binary mask of the text.
    model_type: str, type of the model for which a mask is being found.
        Takes one of the following values: {'cnn', 'text_cnn',
        'fully_connected'}.

  Raises:
    ValueError: If -
      * model_type is 'text_cnn' and mask isn't a 1D numpy array, or
      * model_type is 'fully_connected', cnn' and mask isn't a 2D numpy array
          with num rows equal to num columns.
      raises a value error.
  """
  if model_type == 'text_cnn' and mask.ndim != 1:
    raise ValueError('Invalid mask shape: {}. Expected a mask '
                     'with 1 dimension.'.format(mask.shape))

  if model_type == 'cnn' or model_type == 'fully_connected':
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
      raise ValueError('Invalid mask shape: {}. Expected a mask '
                       'with 2 equal dimensions.'.format(mask.shape))


def _record_solution(result, mask, solver_output, image, session, run_params):
  """Stores the activations and logits of the masked and the inv-masked images.

  Using the image and its mask, this function generates the masked image and
  the inv-masked image. Then, it does a forward pass to find the pre-relu
  activations of the first hidden layer, and the logits and then stores them
  in the result dictionary.

  Args:
    result: defaultdict,
      * image: float numpy array with shape
          (image_edge_length * image_edge_length * image_channels,)
      * combined_solver_runtime: float, time taken by the solver to find all
          the solutions.
      * unmasked_logits: float numpy array with shape (num_outputs,)
      * unmasked_first_layer: float numpy array with shape
          (num_hidden_nodes_first_layer,)
      * masked_first_layer: list with length num_sols, contains float numpy
          array with shape (num_hidden_nodes_first_layer,)
      * inv_masked_first_layer: list with length num_sols, contains float numpy
          array with shape (num_hidden_nodes_first_layer,)
      * masks: list with length num_sols, contains float numpy array
          with shape (image_edge_length ** 2,)
      * masked_images: list with length num_sols, contains float numpy array
          with shape (image_edge_length ** 2,)
      * inv_masked_images: list with length num_sols, contains float numpy
          array with shape (image_edge_length ** 2,)
      * masked_logits: list with length num_sols, contains float numpy array
          with shape (num_outputs,)
      * inv_masked_logits: list with length num_sols, contains float numpy
          array with shape (num_outputs,)
      * solver_outputs: list with length num_sols, contains strings
          corresponding to every sampled solution saying 'sat', 'unsat' or
          'unknown'.
    mask: For
      * image - numpy array with shape (image_edge_length, image_edge_length),
        binary mask of the image.
      * text - numpy array with shape (num_words,), binary mask of the text.
    solver_output: string, takes the value 'sat', 'unsat', or 'unknown'.
    image: numpy array with shape (image_edge_length, image_edge_length,
        image_channels), image for which the mask was found.
    session: tensorflow session with loaded graph.
    run_params: RunParams with model_type, image_placeholder_shape,
        tensor_names.
  """
  _verify_mask_dimensions(mask, run_params.model_type)
  tensor_names = run_params.tensor_names
  image_placeholder_shape = run_params.image_placeholder_shape

  if run_params.model_type != 'text_cnn':
    mask = np.repeat(mask[:, :, np.newaxis], image.shape[2], axis=2)

  masked_image = image * mask
  masked_predictions = session.run(
      tensor_names,
      feed_dict={
          tensor_names['input']: masked_image.reshape(image_placeholder_shape)})

  inv_masked_image = image * (1 - mask)
  inv_masked_predictions = session.run(
      tensor_names,
      feed_dict={
          tensor_names['input']:
              inv_masked_image.reshape(image_placeholder_shape)})
  result['masks'].append(mask.reshape(-1))
  result['masked_images'].append(masked_image.reshape(-1))
  result['masked_logits'].append(masked_predictions['logits'].reshape(-1))
  # masked_first_layer is stored even in the case of full_encoding to study the
  # first layer activations.
  result['masked_first_layer'].append(
      masked_predictions['first_layer'].reshape(-1))
  result['solver_outputs'].append(solver_output.encode('utf-8'))
  result['inv_masked_logits'].append(
      inv_masked_predictions['first_layer'].reshape(-1))
  result['inv_masked_images'].append(inv_masked_image.reshape(-1))
  result['inv_masked_first_layer'].append(
      inv_masked_predictions['logits'].reshape(-1))


def _verify_image_dimensions(image):
  """Verifies if the input image has the correct shape.

  Args:
    image: float numpy array with shape (image_edge_length, image_edge_length,
        image_channels), image to be masked.

  Raises:
    ValueError: The input image should be of the shape- (height, width,
        channels). Raises an error if the image doesn't have 3 dimensions,
        or height != width, or if channels has a value other than
        1 (black and white image) and 3 (rgb image).
  """
  if np.ndim(image) != 3:
    raise ValueError('The input image should have 3 dimensions. Shape of the '
                     'image: %s' % str(image.shape))
  if image.shape[0] != image.shape[1]:
    raise ValueError('The input image should have height == width. Shape of '
                     'the input image: %s' % str(image.shape))
  if image.shape[2] != 1 and image.shape[2] != 3:
    raise ValueError('The color channels of the input image has a value other '
                     'than 1 or 3. Shape of the image: %s' % str(image.shape))


def find_mask_full_encoding(image,
                            weights,
                            biases,
                            run_params,
                            window_size,
                            label_index,
                            delta=0,
                            timeout=600,
                            num_unique_solutions=1):
  """Finds a binary mask for a given image and a trained Neural Network.

  Args:
    image: float numpy array with shape (image_edge_length, image_edge_length,
        image_channels), image to be masked. For MNIST, the pixel values are
        between [0, 1] and for Imagenet, the pixel values are between
        [-117, 138].
    weights: list of num_layers float numpy arrays with shape
        (output_dim, input_dim), weights of the neural network.
    biases: list of num_layers float numpy arrays with shape (output_dim,),
        biases of the neural network.
    run_params: RunParams with model_type, model_path, image_placeholder_shape,
        activations, tensor_names.
    window_size: int, side length of the square mask.
    label_index: int, index of the label of the training image.
    delta: float, logit of the correct label is greater than the rest of the
        logit by an amount delta. Its value is always >= 0. It is only used when
        constrain_final_layer is True.
    timeout: int, solver timeout in seconds.
    num_unique_solutions: int, number of unique solutions you want to sample.

  Returns:
    result: dictionary,
      * image: float numpy array with shape
          (image_edge_length * image_edge_length * image_channels,)
      * combined_solver_runtime: float, time taken by the solver to find all
          the solutions.
      * unmasked_logits: float numpy array with shape (num_outputs,)
      * unmasked_first_layer: float numpy array with shape
          (num_hidden_nodes_first_layer,)
      * masked_first_layer: list with length num_sols, contains float numpy
          array with shape (num_hidden_nodes_first_layer,)
      * inv_masked_first_layer: list with length num_sols, contains float numpy
          array with shape (num_hidden_nodes_first_layer,)
      * masks: list with length num_sols, contains float numpy array
          with shape (image_edge_length ** 2,)
      * masked_images: list with length num_sols, contains float numpy array
          with shape (image_edge_length ** 2,)
      * inv_masked_images: list with length num_sols, contains float numpy
          array with shape (image_edge_length ** 2,)
      * masked_logits: list with length num_sols, contains float numpy array
          with shape (num_outputs,)
      * inv_masked_logits: list with length num_sols, contains float numpy
          array with shape (num_outputs,)
      * solver_outputs: list with length num_sols, contains strings
          corresponding to every sampled solution saying 'sat', 'unsat' or
          'unknown'.
  """
  _verify_image_dimensions(image)
  image_placeholder_shape = run_params.image_placeholder_shape
  tensor_names = run_params.tensor_names
  # z3's timeout is in milliseconds
  z3.set_option('timeout', timeout * 1000)
  image_edge_length, _, _ = image.shape
  num_masks_along_row = image_edge_length // window_size
  session = utils.restore_model(run_params.model_path)

  z3_mask = [z3.Int('mask_%d' % i) for i in range(num_masks_along_row ** 2)]

  unmasked_predictions = session.run(
      tensor_names,
      feed_dict={
          tensor_names['input']: image.reshape(image_placeholder_shape)})

  smt_output, _ = utils.smt_forward(
      features=utils.flatten_nested_lists(_encode_input(
          image=image,
          z3_mask=z3_mask,
          window_size=window_size)),
      weights=weights,
      biases=biases,
      activations=run_params.activations)

  z3_optimizer = _formulate_smt_constraints_final_layer(
      z3_optimizer=utils.ImageOptimizer(
          z3_mask=z3_mask,
          window_size=window_size,
          edge_length=image_edge_length),
      smt_output=smt_output,
      delta=delta,
      label_index=label_index)
  solver_start_time = time.time()
  result = collections.defaultdict(list)

  # All the masks found in each call of z3_optimizer.generator() is guarranteed
  # to be unique since duplicated solutions are blocked. For more details
  # refer z3_optimizer.generator().
  for mask, solver_output in z3_optimizer.generator(num_unique_solutions):
    _record_solution(result=result,
                     mask=mask,
                     solver_output=solver_output,
                     image=image,
                     session=session,
                     run_params=run_params)
  result.update({
      'image': image.reshape(-1),
      'combined_solver_runtime': time.time() - solver_start_time,
      'unmasked_logits': np.squeeze(unmasked_predictions['logits']),
      'unmasked_first_layer': np.squeeze(unmasked_predictions['first_layer'])})
  session.close()
  return result


def _verify_activation_maps_shape(activation_maps_shape, model_type):
  """Checks the activation_maps_shape is valid given the model_type.

  Args:
    activation_maps_shape: tuple of length 4, shape of the activation map.
    model_type: str, type of the model for which a mask is being found.
        Takes one of the following values: {'cnn', 'text_cnn'}.

  Raises:
    ValueError: If -
      * the model_type is not one of 'cnn' or 'text_cnn', or
      * model_type is 'cnn' and activation_maps_shape doesn't have a
          length 4, or
      * model_type is 'text_cnn' and activation_maps_shape doesn't have a
          length 3,
      raises a value error.
  """
  if model_type != 'cnn' and model_type != 'text_cnn':
    raise ValueError('Invalid model_type: {}. Expected one of - '
                     'cnn or text_cnn'.format(model_type))

  if model_type == 'cnn' and len(activation_maps_shape) != 4:
    raise ValueError('Invalid activation_maps_shape: {}. '
                     'Expected length 4.'.format(activation_maps_shape))

  if model_type == 'text_cnn' and len(activation_maps_shape) != 3:
    raise ValueError('Invalid activation_maps_shape: {}. '
                     'Expected length 3.'.format(activation_maps_shape))


def _get_activation_map_shape(activation_maps_shape, model_type):
  """Returns the shape of the activation map produced by each conv kernel.

  Args:
    activation_maps_shape: For model_type,
      * 'cnn' - tuple representing (batch_size, num_rows, num_columns,
           num_output_channels), shape of the 1st layer activation maps.
      * 'text_cnn' - tuple representing (batch_size, num_rows,
           num_output_channels), shape of the 1st layer activation maps.
    model_type: str, type of the model for which a mask is being found.
        Takes one of the following values: {'cnn', 'text_cnn'}.

  Returns:
    tuple of length 2, shape of the activation map per kernel of the form
        (num_rows, num_columns).
  """
  _verify_activation_maps_shape(
      activation_maps_shape=activation_maps_shape, model_type=model_type)
  if model_type == 'cnn':
    # In case of CNNs trained on image having 2D conv operations,
    # the shape is (batch_size, num_rows, num_columns,
    # num_output_channels).
    return activation_maps_shape[1], activation_maps_shape[2]
  else:
    # In case of CNNs trained on text having 1D conv operations,
    # the shape is (batch_size, num_rows, num_output_channels).
    return activation_maps_shape[1], 1


def get_no_minimization_mask(image, label_index, top_k, run_params,
                             sum_attributions=False):
  """Generates a no minimization mask for a given image.

  Args:
    image:
      * image: float numpy array with shape (image_edge_length,
          image_edge_length, image_channels), image to be masked. For MNIST,
          the pixel values are between [0, 1] and for Imagenet, the pixel
          values are between [-117, 138].
      * text: float numpy array with shape (num_words,), text to be masked.
    label_index: int, index of the label of the training image.
    top_k: int, constrain the nodes with top k activations in the first hidden
        layer.
    run_params: RunParams with model_type, model_path, image_placeholder_shape,
        padding, strides, tensor_names.
    sum_attributions: bool, If true, the attribution of a pixel is the sum of
        all the attributions of the equations in which the masking variable
        appears. Otherwise, it is the max of all the attributions of the
        equations in which the masking variable appears.

  Returns:
    float numpy array with shape (num_rows, num_cols), no minimization mask.
  """
  session = utils.restore_model(run_params.model_path)
  unmasked_predictions = session.run(
      run_params.tensor_names,
      feed_dict={run_params.tensor_names['input']: image.reshape(
          run_params.image_placeholder_shape)})

  chosen_indices = _sort_indices(
      unmasked_predictions=unmasked_predictions,
      score_method='integrated_gradients',
      session=session,
      image=image,
      run_params=run_params,
      label_index=label_index)

  scores = _reorder(get_saliency_map(
      session=session,
      features=_remove_batch_axis(
          unmasked_predictions['first_layer_relu']),
      saliency_method='integrated_gradients',
      input_tensor_name=run_params.tensor_names['first_layer_relu'],
      output_tensor_name=run_params.tensor_names['softmax'],
      label=label_index)).reshape(-1)
  if run_params.model_type == 'text_cnn':
    mask = np.zeros(image.shape[0] + sum(run_params.padding))
  else:
    mask = np.zeros((image.shape[0] + sum(run_params.padding),
                     image.shape[1] + sum(run_params.padding)))
  num_rows, num_columns = _get_activation_map_shape(
      activation_maps_shape=unmasked_predictions['first_layer'].shape,
      model_type=run_params.model_type)
  kernel_size = unmasked_predictions['weights_layer_1'].shape[1]
  for index in chosen_indices[-top_k:]:
    _, row, column = _get_hidden_node_location(
        flattened_index=index, num_rows=num_rows, num_columns=num_columns)
    top = row * run_params.strides
    left = column * run_params.strides
    if sum_attributions:
      if run_params.model_type == 'text_cnn':
        mask[top: top + kernel_size] += scores[index]
      else:
        mask[top: top + kernel_size, left: left + kernel_size] += scores[index]
    else:
      # The importance of a masking variable is always over-written with
      # a higher value as the indices are sorted.
      if run_params.model_type == 'text_cnn':
        mask[top: top + kernel_size] = scores[index]
      else:
        mask[top: top + kernel_size, left: left + kernel_size] = scores[index]
  session.close()

  left_padding = run_params.padding[0]
  if run_params.model_type == 'text_cnn':
    return mask[left_padding: image.shape[0] + left_padding]
  else:
    return mask[left_padding: image.shape[0] + left_padding,
                left_padding: image.shape[1] + left_padding]


def _remove_batch_axis(array):
  """Removes the batch dimension (axis 0) from the array.

  This function is preferred over np.squeeze(array, axis=0) because the
  later only works as expected with numpy arrays. In case the array is a list
  of numpy array, np.squeeze removes all the dimensions with size 1 instead
  of just the 0th dimension. To avoid this undesirable effect, we create a
  separate function.

  Args:
    array: A list of nd numpy arrays or an nd numpy array.

  Returns:
    numpy array with one dimension less than the input.

  Raises:
    ValueError: Raises an error if the array / list doesn't have
        a length 1 along the first (batch) dimension.
  """
  if len(array) != 1:
    raise ValueError('The array doesn\'t have the batch dimension as 1. '
                     'Received an array with length along the batch '
                     'dimension: %d' % len(array))
  return array[0]


def _reorder(array):
  """Reorders an nd array by moving the last axis to the first.

  This function is used to reorder the following 4D numpy arrays -
  'conv_kernels', 'first_layer_activations', and 'first_layer_saliency_maps'.
  These arrays have num_output_channels as their last dimension, which is
  made their first.

  Args:
    array: 4d float numpy array, array to be reordered.

  Returns:
    4d float numpy array, reorderd array.
  """
  return np.moveaxis(array, -1, 0)


def _reshape_kernels(kernels, model_type):
  """Reshapes the kernel by moving axes.

  The kernel is reshaped into (num_output_channels, num_rows, num_columns,
  num_input_channels).

  Args:
    kernels: float numpy array with 4 dimensions,
        weights of the convolution layer.
    model_type: str, type of the model for which a mask is being found.
        Takes one of the following values: {'cnn', 'text_cnn'}.

  Returns:
    float numpy array with shape (num_output_channels, num_rows, num_columns,
        num_input_channels), reshaped kernel weights.
  """
  if model_type == 'cnn':
    # In case of CNNs trained on image having 2D conv operations,
    # the shape of the kernel is (num_rows, num_columns, num_input_channels,
    # num_output_channels).
    return _reorder(kernels)
  else:
    # In case of CNNs trained on text having 1D conv operations,
    # the shape of the kernel is (1, num_rows, num_columns,
    # num_output_channels).
    return np.swapaxes(kernels, 0, 3)


def get_saliency_map(session, features, saliency_method, label,
                     input_tensor_name, output_tensor_name):
  """Generates a saliency map for the input features.

  Args:
    session: tensorflow session.
    features: numpy array, the features for which a saliency map is to be
        computed.
    saliency_method: string, the saliency method to be used. Takes one of the
        following values: {'integrated_gradients',
        'integrated_gradients_black_white_baselines', 'xrai'}.
    label: int, label of the image which also corresponds to the logit
        associated with the image in the final layer.
    input_tensor_name: str, name of input tensor in tf graph.
    output_tensor_name: str, name of the output tensor in the tf graph.

  Returns:
    A saliency map.
  """
  graph = tf.get_default_graph()
  label_placeholder = tf.placeholder(tf.int32)
  output_tensor = graph.get_tensor_by_name(
      output_tensor_name)[0][label_placeholder]
  input_tensor = graph.get_tensor_by_name(input_tensor_name)
  if saliency_method == 'integrated_gradients':
    # Integrated Gradients is used on the first layer activations.
    # We run IG for 200 steps because empirically we find with these many steps,
    # the IG scores converges.
    return integrated_gradients.IntegratedGradients(
        graph=graph, session=session, y=output_tensor, x=input_tensor).GetMask(
            x_value=features, feed_dict={label_placeholder: label}, x_steps=200)
  elif saliency_method == 'integrated_gradients_black_white_baselines':
    # Integrated Gradients (Black + White baselines) is used on the input.
    # Computes 2 saliency maps using a black image and a white image as a
    # baseline separately and returns their mean average.
    # We run IG for 200 steps because empirically we find with these many steps,
    # the IG scores converges.
    saliency_maps = []
    for baseline in [
        np.min(features) * np.ones_like(features),  # black baseline
        np.max(features) * np.ones_like(features),  # white baseline
    ]:
      saliency_maps.append(
          integrated_gradients.IntegratedGradients(
              graph=graph, session=session, y=output_tensor,
              x=input_tensor).GetMask(
                  x_value=features,
                  x_baseline=baseline,
                  feed_dict={label_placeholder: label},
                  x_steps=200))
    return np.mean(saliency_maps, axis=0)
  elif saliency_method == 'xrai':
    return xrai.XRAI(
        graph=graph, session=session, y=output_tensor, x=input_tensor).GetMask(
            x_value=features, feed_dict={label_placeholder: label})


def _get_gradients(session, graph, features, label_index, input_tensor_name,
                   output_tensor_name):
  """Computes gradient wrt an input tensor.

  Args:
    session: an instance of tf.Session, tensorflow session.
    graph: an instance of tf.Graph, tensorflow graph.
    features: float numpy array, value of the input tensor for which gradient is
        to be computed.
    label_index: int, index of the label of the training image.
    input_tensor_name: str, tensor name of the input for which the gradient is
        being computed.
    output_tensor_name: str, tensor name of the output layer.

  Returns:
    float numpy array, gradients.
  """
  gradient = tf.gradients(
      ys=graph.get_tensor_by_name(output_tensor_name)[0][label_index],
      xs=graph.get_tensor_by_name(input_tensor_name))
  return _remove_batch_axis(session.run(
      gradient, feed_dict={input_tensor_name: features}))


def _apply_blurring(image, sigma):
  """Applies a gaussian blur to the image with the specified sigma.

  Args:
    image: float numpy array with shape (image_edge_length, image_edge_length,
        image_channels), image.
    sigma: float, variance of the gaussian kernel used for blurring.

  Returns:
    float numpy array with shape (image_edge_length, image_edge_length,
        image_channels), blurred image.
  """
  return scipy_ndimage.gaussian_filter(
      image, sigma=[sigma, sigma, 0], mode='constant')


def _sort_indices(session, image, label_index, run_params, unmasked_predictions,
                  score_method):
  """Sorts the indices of the first layer on the basis of priority.

  First, the function assigns a score to the hidden activations using
  one of the following methods -
    1. magnitude of the first layer activations.
    2. gradients of the first layer.
    3. (image only) Blurring the input, in case of image, and computing
       the 1st layer gradients wrt to the blurred image.
    4. integrated_gradients with a black baseline on the first layer.
    5. integrated_gradients with a black + white baseline on the first layer.

  Then, if the input is -
    1. Image - The first layer activations, because of 2D convolutions,
       have the shape (batch_size, output_activation_map_size,
       output_activation_map_size, output_activation_map_channels). After
       computing the priority, the priority array is reshaped into
       (output_activation_map_channels, output_activation_map_size,
       output_activation_map_size) and flattened.
    2. Text - The first layer activations, because of 1D convolutions,
       have the shape (batch_size, activations_per_kernel, num_kernels).
       After computing the priority, priority array is reshaped into
       (num_kernels, activations_per_kernel) and flattened.

  Finally, the corresponding arguments are sorted on the basis of assigned
  importance.

  Args:
    session: instance of tf.Session(), tensorflow session.
    image: float numpy array with shape (image_edge_length, image_edge_length,
        image_channels), image to be masked. For MNIST, the pixel values are
        between [0, 1] and for Imagenet, the pixel values are between
        [-117, 138].
    label_index: int, index of the label of the training image.
    run_params: RunParams with image_placeholder_shape and tensor_names.
    unmasked_predictions: dict,
      * input: float numpy array, the input tensor to the neural network.
      * first_layer: float numpy array, the first layer tensor in the neural
          network.
      * first_layer_relu: str, the first layer relu activation
          tensor in the neural network.
      * logits: str, the logits tensor in the neural network.
      * softmax: float numpy array, the softmax tensor in the neural network.
      * weights_layer_1: float numpy array, the first layer fc / conv weights.
      * biases_layer_1: float numpy array, the first layer fc / conv biases.
      * (text only) embedding: float numpy array with shape (num_words,
          num_latent_dimensions), the embedding layer.
    score_method: str, assigns scores to hidden nodes, and nodes with the
        top_k scores are chosen. Takes a value -
        {'activations', 'blurred_gradients', 'gradients',
        'integrated_gradients', 'integrated_gradients_black_white_baselines'}.

  Returns:
    int numpy array with shape (num_first_layer_activations,), sorted indices
        of the first layer.
  """
  tensor_names = run_params.tensor_names
  if score_method == 'activations':
    first_layer_priority = _reorder(
        _remove_batch_axis(unmasked_predictions['first_layer'])).reshape(-1)
  elif score_method == 'integrated_gradients':
    first_layer_priority = _reorder(
        get_saliency_map(
            session=session,
            features=_remove_batch_axis(
                unmasked_predictions['first_layer_relu']),
            saliency_method='integrated_gradients',
            input_tensor_name=tensor_names['first_layer_relu'],
            output_tensor_name=tensor_names['softmax'],
            label=label_index)).reshape(-1)
  elif score_method == 'gradients':
    first_layer_priority = _get_gradients(
        session=session,
        graph=tf.get_default_graph(),
        # features should also account for the batch size.
        features=unmasked_predictions['first_layer_relu'],
        label_index=label_index,
        input_tensor_name=tensor_names['first_layer_relu'],
        output_tensor_name=tensor_names['softmax'])
    first_layer_priority = _reorder(
        _remove_batch_axis(first_layer_priority)).reshape(-1)
  elif score_method == 'blurred_gradients':
    blurred_predictions = session.run(
        tensor_names,
        feed_dict={
            tensor_names['input']:
                _apply_blurring(image, sigma=3).reshape(
                    run_params.image_placeholder_shape)})
    first_layer_priority = _get_gradients(
        session=session,
        graph=tf.get_default_graph(),
        # features should also account for the batch size.
        features=blurred_predictions['first_layer_relu'],
        label_index=label_index,
        input_tensor_name=tensor_names['first_layer_relu'],
        output_tensor_name=tensor_names['softmax'])
    first_layer_priority = _reorder(
        _remove_batch_axis(first_layer_priority)).reshape(-1)
  return first_layer_priority.argsort()


def _process_image(image, run_params, window_size):
  """Generates the masked input and does a forward pass of the image.

  Args:
    image: float numpy array with shape (image_edge_length,
          image_edge_length, image_channels), image to be masked. For MNIST,
          the pixel values are between [0, 1] and for Imagenet, the pixel
          values are between [-117, 138].
    run_params: RunParams with model_type, model_path, image_placeholder_shape,
        activations, tensor_names, input, first_layer, logits.
    window_size: int, side length of the square mask.

  Returns:
    masked_input: nested list of z3.ExprRef with dimensions
      (image_channels, image_edge_length, image_edge_length)
    unmasked_predictions: dict,
      * input: float numpy array, the input tensor to the neural network.
      * first_layer: float numpy array, the first layer tensor in the neural
          network.
      * first_layer_relu: str, the first layer relu activation
          tensor in the neural network.
      * logits: str, the logits tensor in the neural network.
      * softmax: float numpy array, the softmax tensor in the neural network.
      * weights_layer_1: float numpy array, the first layer fc / conv weights.
      * biases_layer_1: float numpy array, the first layer fc / conv biases.
    session: tf.Session, tensorflow session with the loaded neural network.
    optimizer: utils.ImageOptimizer, z3 optimizer for image.
  """
  _verify_image_dimensions(image)
  image_edge_length, _, _ = image.shape
  num_masks_along_row = image_edge_length // window_size
  # We always find a 2d mask irrespective of the number of image channels.
  z3_mask = [z3.Int('mask_%d' % i) for i in range(num_masks_along_row ** 2)]

  session = utils.restore_model(run_params.model_path)
  unmasked_predictions = session.run(
      run_params.tensor_names,
      feed_dict={run_params.tensor_names['input']: image.reshape(
          run_params.image_placeholder_shape)})

  # _encode_input generates a masked_input with a shape
  # (image_channels, image_edge_length, image_edge_length)
  return (_encode_input(image=image, z3_mask=z3_mask, window_size=window_size),
          unmasked_predictions, session,
          utils.ImageOptimizer(z3_mask=z3_mask, window_size=window_size,
                               edge_length=image_edge_length))


def _process_text(image, run_params):
  """Generates the masked embedding and does a forward pass of the image.

  Args:
    image: float numpy array with shape (num_words,), text to be masked.
    run_params: RunParams with model_type, model_path, image_placeholder_shape,
        activations, tensor_names, input, first_layer, logits.

  Returns:
    masked_input: nested list of z3.ExprRef with dimensions
      (1, num_words, num_latent_dimensions).
    unmasked_predictions: dict,
      * input: float numpy array, the input tensor to the neural network.
      * first_layer: float numpy array, the first layer tensor in the neural
          network.
      * first_layer_relu: str, the first layer relu activation
          tensor in the neural network.
      * logits: str, the logits tensor in the neural network.
      * softmax: float numpy array, the softmax tensor in the neural network.
      * weights_layer_1: float numpy array, the first layer fc / conv weights.
      * biases_layer_1: float numpy array, the first layer fc / conv biases.
      * (text only) embedding: float numpy array with shape (num_words,
          num_latent_dimensions), the embedding layer.
    session: tf.Session, tensorflow session with the loaded neural network.
    optimizer: utils.TextOptimizer, z3 optimizer for image.
  Raises:
    ValueError: Raises an error if the text isn't a 1D array.
  """
  if image.ndim != 1:
    raise ValueError('The text input should be a 1D numpy array. '
                     'Shape of the received input: %s' % str(image.shape))
  session = utils.restore_model(run_params.model_path)
  unmasked_predictions = session.run(
      run_params.tensor_names, feed_dict={
          run_params.tensor_names['input']: image.reshape(
              run_params.image_placeholder_shape)})

  text_embedding = _remove_batch_axis(unmasked_predictions['embedding'])
  # text_embedding has a shape (num_words, num_latent_dimensions)
  z3_mask = [z3.Int('mask_%d' % i) for i in range(text_embedding.shape[0])]

  # masked_input has a shape (num_words, num_latent_dimensions)
  masked_input = []
  for mask_bit, embedding_row in zip(z3_mask, text_embedding):
    masked_input.append([z3.ToReal(mask_bit) * i for i in embedding_row])

  return ([masked_input], unmasked_predictions, session,
          utils.TextOptimizer(z3_mask=z3_mask))


def find_mask_first_layer(image,
                          label_index,
                          run_params,
                          window_size,
                          score_method,
                          top_k=None,
                          gamma=None,
                          timeout=600,
                          num_unique_solutions=1):
  """Finds a binary mask for a given image and a trained Neural Network.

  Args:
    image:
      * image: float numpy array with shape (image_edge_length,
          image_edge_length, image_channels), image to be masked. For MNIST,
          the pixel values are between [0, 1] and for Imagenet, the pixel
          values are between [-117, 138].
      * text: float numpy array with shape (num_words,), text to be masked.
    label_index: int, index of the label of the training image.
    run_params: RunParams with model_type, model_path, image_placeholder_shape,
        padding, strides, tensor_names.
    window_size: int, side length of the square mask.
    score_method: str, assigns scores to hidden nodes, and nodes with the
        top_k scores are chosen. Takes a value -
        {'activations', 'blurred_gradients', 'gradients',
        'integrated_gradients', 'integrated_gradients_black_white_baselines'}.
    top_k: int, constrain the nodes with top k activations in the first hidden
        layer. It is only used when constrain_final_layer is false.
    gamma: float, masked activation is greater than gamma times the unmasked
        activation. Its value is always between [0,1).
    timeout: int, solver timeout in seconds.
    num_unique_solutions: int, number of unique solutions you want to sample.

  Returns:
    result: dictionary,
      * image: float numpy array with shape
          (image_edge_length * image_edge_length * image_channels,)
      * combined_solver_runtime: float, time taken by the solver to find all
          the solutions.
      * masked_first_layer: list with length num_unique_solutions,
          contains float list with length (num_hidden_nodes_first_layer,)
      * inv_masked_first_layer: list with length num_unique_solutions,
          contains float list with length (num_hidden_nodes_first_layer,)
      * masks: list with length num_unique_solutions, contains float numpy array
          with shape (image_edge_length ** 2,)
      * masked_images: list with length num_unique_solutions, contains float
          numpy array with shape (image_channels * image_edge_length ** 2,)
      * inv_masked_images: list with length num_unique_solutions,
          contains float numpy array with shape (image_edge_length ** 2,)
      * masked_logits: list with length num_unique_solutions,
          contains float list with length (num_outputs,)
      * inv_masked_logits: list with length num_unique_solutions, contains float
          list with length (num_outputs,)
      * solver_outputs: list with length num_unique_solutions, contains strings
          corresponding to every sampled solution saying 'sat', 'unsat' or
          'unknown'.
      * chosen_indices
  """
  # z3's timeout is in milliseconds
  z3.set_option('timeout', timeout * 1000)
  model_type = run_params.model_type
  result = collections.defaultdict(list)

  if model_type == 'text_cnn':
    # For text data, window size is always 1 i.e. 1 masking variable per word.
    masked_input, unmasked_predictions, session, z3_optimizer = _process_text(
        image, run_params)
  else:
    masked_input, unmasked_predictions, session, z3_optimizer = _process_image(
        image, run_params, window_size)

  if model_type == 'fully_connected':
    _, smt_hidden_input = utils.smt_forward(
        features=utils.flatten_nested_lists(masked_input),
        weights=[unmasked_predictions['weights_layer_1']],
        biases=[unmasked_predictions['biases_layer_1']],
        activations=['relu'])
    # assign first layer pre-relu activations to smt_hidden_input
    z3_optimizer = _formulate_smt_constraints_fully_connected_layer(
        z3_optimizer=z3_optimizer,
        nn_first_layer=unmasked_predictions['first_layer'].reshape(-1),
        smt_first_layer=smt_hidden_input[0],
        gamma=gamma,
        top_k=top_k)
  else:
    chosen_indices = _sort_indices(
        unmasked_predictions=unmasked_predictions,
        score_method=score_method,
        session=session,
        image=image,
        run_params=run_params,
        label_index=label_index)
    result.update({'chosen_indices': chosen_indices})
    z3_optimizer = _formulate_smt_constraints_convolution_layer(
        z3_optimizer=z3_optimizer,
        kernels=_reshape_kernels(
            kernels=unmasked_predictions['weights_layer_1'],
            model_type=model_type),
        biases=unmasked_predictions['biases_layer_1'],
        chosen_indices=chosen_indices[-top_k:],  # pylint: disable=invalid-unary-operand-type
        # unmasked_predictions['first_layer'] has the shape
        # (batch_size, output_activation_map_size, output_activation_map_size,
        # output_activation_map_channels). This is reshaped into
        # (output_activation_map_channels, output_activation_map_size,
        # output_activation_map_size) and then flattened.
        conv_activations=_reorder(_remove_batch_axis(
            unmasked_predictions['first_layer'])).reshape(-1),
        input_activation_maps=masked_input,
        output_activation_map_shape=_get_activation_map_shape(
            activation_maps_shape=unmasked_predictions['first_layer'].shape,
            model_type=model_type),
        strides=run_params.strides,
        padding=run_params.padding,
        gamma=gamma)

  solver_start_time = time.time()
  # All the masks found in each call of z3_optimizer.generator() is guaranteed
  # to be unique since duplicated solutions are blocked. For more details
  # refer z3_optimizer.generator().
  for mask, solver_output in z3_optimizer.generator(num_unique_solutions):
    _record_solution(result=result,
                     mask=mask,
                     solver_output=solver_output,
                     image=image,
                     session=session,
                     run_params=run_params)
  result.update({
      'image': image.reshape(-1),
      'combined_solver_runtime': time.time() - solver_start_time,
      'unmasked_logits': unmasked_predictions['logits'].reshape(-1),
      'unmasked_first_layer': unmasked_predictions['first_layer'].reshape(-1)})
  session.close()
  return result
