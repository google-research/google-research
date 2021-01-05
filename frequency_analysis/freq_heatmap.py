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

"""Frequency analysis of image classification models.

Add frequency basis vectors as perturbations to the inputs of image
classification models. Analyze how the output changes across different frequency
components.
"""

from typing import List, Optional, Tuple

import numpy as np
from six.moves import map
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf


def _get_symmetric_pos(dims, pos):
  """Compute the symmetric position of the point in 2D FFT.

  Args:
    dims: a tuple of 2 positive integers, dimensions of the 2D array.
    pos: a tuple of 2 integers, coordinate of the query point.

  Returns:
    a numpy array of shape [2], the coordinate of the symmetric point of the
      query point.
  """
  x = np.array(dims)
  p = np.array(pos)
  return np.where(np.mod(x, 2) == 0, np.mod(x - p, x), x - 1 - p)


def _get_fourier_basis(x, y):
  """Compute real-valued basis vectors of 2D discrete Fourier transform.

  Args:
    x: first dimension of the 2D numpy array.
    y: second dimension of the 2D numpy array.

  Returns:
    fourier_basis, a real-valued numpy array of shape (x, y, x, y).
      Each 2D slice of the last two dimensions of fourier_basis,
      i.e., fourier_basis[i, j, :, :] is a 2D array with unit norm,
      such that its Fourier transform, when low frequency located
      at the center, is supported at (i, j) and its symmetric position.
  """
  # If Fourier basis at (i, j) is generated, set marker[i, j] = 1.
  marker = np.zeros([x, y], dtype=np.uint8)
  fourier_basis = np.zeros([x, y, x, y], dtype=np.float32)
  for i in range(x):
    for j in range(y):
      if marker[i, j] > 0:
        continue
      freq = np.zeros([x, y], dtype=np.complex64)
      sym = _get_symmetric_pos((x, y), (i, j))
      sym_i = sym[0]
      sym_j = sym[1]
      if (sym_i, sym_j) == (i, j):
        freq[i, j] = 1.0
        marker[i, j] = 1
      else:
        freq[i, j] = 0.5 + 0.5j
        freq[sym_i, sym_j] = 0.5 - 0.5j
        marker[i, j] = 1
        marker[sym_i, sym_j] = 1
      basis = np.fft.ifft2(np.fft.ifftshift(freq))
      basis = np.sqrt(x * y) * np.real(basis)
      fourier_basis[i, j, :, :] = basis
      if (sym_i, sym_j) != (i, j):
        fourier_basis[sym_i, sym_j, :, :] = basis
  return fourier_basis


def _get_sum_of_norms(a_list):
  """Compute sum of row norms of the reshaped 2D arrays in a_list.

  For each numpy array in a_list, this function first reshape the array to 2D
  with shape (batch_size, ?), then compute the l_2 norm of each row, i.e.,
  norm of [i, :] for i in range(batch_size), and then compute the sum of the row
  norms. This is equivalent to computing the l_(2,1) norm of the transpose of
  the 2D matrices.

  Args:
    a_list: a list of numpy arrays, each with shape (batch_size, ...)

  Returns:
    a list of sum of l_2 row norms.
  """
  sum_of_norms = []
  for each_array in a_list:
    array_2d = each_array.reshape((each_array.shape[0], -1))
    norms = np.linalg.norm(array_2d, axis=1)
    sum_of_norms.append(np.sum(norms))
  return sum_of_norms


def _generate_perturbed_images(images,
                               perturb_basis,
                               perturb_norm = 1.0,
                               clip_min = None,
                               clip_max = None,
                               rand_flip = False):
  """Generate perturbed images given a perturbation basis.

  Args:
    images: numpy array, clean images before perturbation. The shape of this
      array should be [batch_size, height, width] or
      [batch_size, height, width, num_channels].
    perturb_basis: numpy array basis matrix for perturbation. The shape of this
      array should be [height, width]. It is assumed to have unit l_2 norm.
    perturb_norm: the l_2 norm of the Fourier-basis perturbations.
    clip_min: lower bound for clipping operation after adding perturbation. If
      None, no lower clipping.
    clip_max: upper bound for clipping operation after adding perturbation. If
      None, no upper clipping.
    rand_flip: whether or not to randomly flip the sign of basis vectors.

  Returns:
    perturbed images, numpy array with the same shape as clean_images.
  """
  if len(images.shape) != 3 and len(images.shape) != 4:
    raise ValueError('Incorrect shape of clean images.')

  if len(images.shape) == 3:
    clean_images = np.expand_dims(images, axis=3)
  elif len(images.shape) == 4:
    clean_images = images
  else:
    raise ValueError('Unexpected number of dimensions: %d' % len(images.shape))

  batch_size = clean_images.shape[0]
  num_channels = clean_images.shape[3]

  if not rand_flip:
    # (batch, height, width, channel) -> (batch, channel, height, width)
    clean_images_t = np.transpose(clean_images, (0, 3, 1, 2))
    perturb_images_t = clean_images_t + perturb_norm * perturb_basis
    # (batch, channel, height, width) -> (batch, height, width, channel)
    perturb_images = np.transpose(perturb_images_t, (0, 2, 3, 1))
  else:
    # Add random flips when adding basis vectors.
    flip = 2.0 * np.random.binomial(
        1, 0.5, size=batch_size * num_channels) - 1.0
    flat_basis = np.reshape(perturb_basis, (-1))
    perturbation = np.reshape(np.outer(flip, flat_basis),
                              (batch_size, num_channels) + perturb_basis.shape)
    # (batch, channel, height, width) -> (batch, height, width, channel)
    perturbation = np.transpose(perturbation, (0, 2, 3, 1))
    perturb_images = clean_images + perturbation

  if clip_min is not None or clip_max is not None:
    perturb_images = np.clip(perturb_images, clip_min, clip_max)

  if len(images.shape) == 3:
    return np.squeeze(perturb_images, axis=3)
  else:
    return perturb_images


class TensorFlowNeuralNetwork(object):
  """The interface between TensorFlow and the heat map generator."""

  def __init__(self,
               sess,
               image_ph,
               label_ph,
               tensor_list,
               eval_tensor):
    """Initializing TensorFlowNeuralNetwork.

    Args:
      sess: a tensorflow session.
      image_ph: a tensorflow placeholder for input images. This tensor should
        have shape [num_examples, height, width] or
        [num_examples, height, width, num_channels].
      label_ph: a tensorflow placeholder for the labels of images. Must be shape
        [num_examples].
      tensor_list: a list of tensors to evaluate. The tensors in this list can
        have arbitrary and different shapes. These tensors correspond to the
        outputs of the layers that we are interested in. To conduct frequency
        analysis, first feed clean images and evaluate the tensors, then feed
        corrupted images using Fourier basis and evaluate the tensors, and
        finally compare the difference between the two evaluations.
      eval_tensor: a tensor of shape [], i.e., a scalar, corresponding to an
        evaluation criterion on a batch. This can be the test accuracy, or any
        loss (such as cross-entropy loss).
    """
    self._sess = sess
    self._image_ph = image_ph
    self._label_ph = label_ph
    self._tensor_list = tensor_list
    self._eval_tensor = eval_tensor

  def __call__(self, images, labels):
    """Feed the images and labels to the network and run the session.

    Args:
      images: a numpy array of images. The shape of image_ph should be
        consistent with image_ph.
      labels: a numpy array for the labels of images.

    Returns:
      a list of numpy arrays with the values of tensors in tensor_list, and a
      scalar corresponds to the evaluation criterion.
    """
    data_dict = {
        self._image_ph: images,
        self._label_ph: labels
    }
    vals = self._sess.run(
        self._tensor_list + [self._eval_tensor], feed_dict=data_dict)
    return vals[:-1], vals[-1]

  def get_num_of_tensors(self):
    """Get the number of tensors in self._tensor_list.

    Returns:
      The length of self._tensor_list.
    """
    return len(self._tensor_list)


def generate_freq_heatmap(neural_network,
                          images,
                          labels,
                          custom_basis = None,
                          perturb_norm = 1.0,
                          batch_size = -1,
                          clip_min = None,
                          clip_max = None,
                          rand_flip = False,
                          seed = None,
                          relative_scale = True):
  """Generate frequency heat map.

  We conduct the frequcy analysis in the following way: 1) feed the images to
  the network and record the values of tensors in tensor_list, and 2) feed the
  images with Fourier-basis perturbations, and record the values of tensors in
  tensor_list, and 3) compare the difference between the tensors.

  Args:
    neural_network: a TensorFlowNeuralNetwork object.
    images: a numpy array of images with shape [num_examples, height, width] or
      [num_examples, height, width, num_channels]. The shape of images should
      be the same as the image placeholder in neural_network.
    labels: a numpy array for the labels of images with shape [num_examples].
    custom_basis: a numpy array of shape [height, width, height, width], with
      each slice [i, j, :, :] being the (i, j) basis vector. If None, we first
      generate the basis.
    perturb_norm: the l_2 norm of the Fourier-basis perturbations.
    batch_size: the batch size when computing the frequency heatmap. If the
      number of examples in image_np is large, we may need to compute the
      heatmap batch by batch. If batch_size is -1, we use the entire image_np.
    clip_min: lower bound for clipping operation after adding perturbation. If
      None, no lower clipping.
    clip_max: upper bound for clipping operation after adding perturbation. If
      None, no upper clipping.
    rand_flip: whether or not to randomly flip the sign of basis vectors.
    seed: numpy random seed for random flips.
    relative_scale: whether or not to return relative scale of the tensor change
      across all the frequency components. If True, the maximum change is
      normalized to be 1; otherwise, return the actual value of the model change
      under Fourier-basis perturbation, averaged across the input images.

  Returns:
    heatmap_list: a list of numpy arrays, each has shape [height, width]. The
      heatmaps in the list correspond to the tensors in the `tensor_list` in
      `neural_network`.
    eval_heatmap: a numpy array of shape [height, width], each entry in the
      array corresponds to the evaluation criterion (`eval_tensor` in
      `neural_network`) under the Fourier basis perturbation, averaged across
      the batches.
    clean_eval: a scalar corresponding to the evaluation criterion on image_np,
      averaged across the batches.
  """
  if len(images.shape) != 3 and len(images.shape) != 4:
    raise ValueError('Incorrect shape of input images.')
  if batch_size == 0 or batch_size < -1:
    raise ValueError('Invalid batch size.')

  if rand_flip:
    # Get the current random seeds.
    numpy_st0 = np.random.get_state()
    # Set new random seeds.
    if seed is not None:
      np.random.seed(seed)

  num_of_tensors = neural_network.get_num_of_tensors()
  num_examples = images.shape[0]
  height = images.shape[1]
  width = images.shape[2]

  if custom_basis is None:
    # Generate basis vectors.
    basis = _get_fourier_basis(height, width)
  else:
    # Check the shape of the custom basis.
    if custom_basis.shape != (height, width, height, width):
      err_msg = 'custom_basis must have shape [height, width, height, width].'
      raise ValueError(err_msg)
    basis = custom_basis

  heatmap_list = [
      np.zeros([height, width], dtype=np.float32) for _ in range(num_of_tensors)
  ]

  eval_hmp = np.zeros([height, width], dtype=np.float32)
  clean_eval = 0.0

  if batch_size == -1:
    batch_size_true = num_examples
  else:
    batch_size_true = batch_size
  for idx in range(0, num_examples, batch_size_true):
    # Evaluate clean inputs on this batch.
    batch_imgs = images[idx:idx + batch_size_true]
    batch_labels = labels[idx:idx + batch_size_true]
    num_examples_batch = batch_imgs.shape[0]
    tensor_clean, eval_batch = neural_network(batch_imgs, batch_labels)

    # Accumulate the evaluation criterion under clean inputs.
    clean_eval += num_examples_batch * eval_batch

    # Evaluate over all frequency basis.
    for i in range(height):
      for j in range(width):
        perturb_images = _generate_perturbed_images(batch_imgs,
                                                    basis[i, j, Ellipsis],
                                                    perturb_norm, clip_min,
                                                    clip_max, rand_flip)
        tensor_pb, eval_pb = neural_network(perturb_images, batch_labels)

        # Accumulate the evaluation criterion under perturbed inputs.
        eval_hmp[i, j] += num_examples_batch * eval_pb

        # Compute the difference between perturb_outputs and clean_outputs.
        diff = list(map(np.subtract, tensor_pb, tensor_clean))
        sum_norm_diff = _get_sum_of_norms(diff)
        for hmp, sum_norms in zip(heatmap_list, sum_norm_diff):
          hmp[i, j] += sum_norms

  # Normalize the evaluation heatmap and `heatmap_list`.
  eval_hmp /= float(num_examples)
  clean_eval /= float(num_examples)
  for hmp in heatmap_list:
    hmp /= float(num_examples)
    if relative_scale:
      # Rescale the heatmaps such that the maximum value is 1.
      hmp /= np.amax(hmp)

  if rand_flip:
    # Reset the seeds back to their original values.
    np.random.set_state(numpy_st0)

  return heatmap_list, eval_hmp, clean_eval
