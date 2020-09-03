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

# Lint as: python3
"""Utilities."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_activations(model, activation_names, numpy_x, batch_size=1024):
  """Gets intermediate activations of the model.

  Args:
    model: a tf.keras model.
    activation_names: a list of layer names of `model`.
    numpy_x: a numpy array that `model` can take as input.
    batch_size: batch size used when getting the activations.

  Returns:
    A list of numpy arrays corresponding to `activation_names`.
  """
  activation_tensors = [model.get_layer(activation_name).output
                        for activation_name in activation_names]
  # model._feed_inputs is the list of input placeholders.
  f = tf.keras.backend.function(model._feed_inputs, activation_tensors)  # pylint: disable=protected-access

  batched_acts = []
  for i in range(0, len(numpy_x), batch_size):

    batch_x = numpy_x[i:(i+batch_size)]

    # a list of len(activation_names) numpy arrays
    batch_acts = f(batch_x)
    batched_acts.append(batch_acts)

  # reassemble
  grouped_acts = zip(*batched_acts)
  activations = [np.vstack(acts) for acts in grouped_acts]

  return activations


def get_activation_shapes(model, activation_names):
  """Gets intermediate activation shapes of the model.

  Args:
    model: a tf.keras model.
    activation_names: a list of layer names of `model`.

  Returns:
    A list of shapes corresponding to `activation_names`.
  """
  # first dimension is batch size
  activation_shapes = [
      model.get_layer(activation_name).output.shape.as_list()[1:]
      for activation_name in activation_names
  ]

  return activation_shapes


def batched_predict_on_batch(unused_self,
                             x,
                             model=None,
                             batch_size=2048,
                             index=None):
  """Gets model.predict(x)[index] by calling predict_on_predict.

  Args:
    unused_self: unused.
    x: numpy array of inputs.
    model: a tf.keras model.
    batch_size: batch_size the inputs will be broken into.
    index: which result to return, if model has multiple outputs

  Returns:
    A numpy array of the same length as x.
  """
  # index: which result to return, if model has multiple outputs
  result = []
  for i in range(0, len(x), batch_size):
    batch_x = x[i:(i+batch_size)]
    batch_pred = model.predict_on_batch(batch_x)

    if index is not None:
      batch_pred = batch_pred[index]

    result.append(batch_pred.numpy())

  return np.vstack(result)


def visualize_similar(test_image_arrays, train_image_arrays_list,
                      n_cols=11, test_labels=None, train_labels=None):
  """Visualize.

  Args:
    test_image_arrays: list of numpy arrays.
    train_image_arrays_list: list of lists of numpy arrays.  Must have the same
      length as test_image_arrays.
    n_cols: number of columns in the displayed images.
    test_labels: the list of labels to be shown below the test images.
    train_labels: the list of labels to be shown below the train images.
  """
  assert len(test_image_arrays) == len(train_image_arrays_list)
  image_height = test_image_arrays[0].shape[0]
  fontsize = 10
  x_offset = 0
  y_offset = int(image_height * 1.25)

  # single test image, followed by a row of train images
  n_rows = len(test_image_arrays)
  fig = plt.figure(figsize=(n_cols, n_rows))

  for i, (test_ia, train_ias) in enumerate(
      zip(test_image_arrays, train_image_arrays_list)):
    row_index = i
    ax = fig.add_subplot(n_rows, n_cols, n_cols * row_index + 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # add borders
    patch_margin = int(image_height / 16)
    patch_shape = (test_ia.shape[0] + 2 * patch_margin,
                   test_ia.shape[1] + 2 * patch_margin, 3)
    patch = np.zeros(patch_shape)
    patch[:, :, 0] = 1
    patch[patch_margin:patch_margin + test_ia.shape[0],
          patch_margin:patch_margin + test_ia.shape[1]] = test_ia

    if test_labels is not None:
      ax.text(
          x_offset,
          y_offset + 2 * patch_margin,
          '{}'.format(test_labels[i]),
          fontsize=fontsize)

    ax.imshow(patch)

    for j, train_ia in enumerate(train_ias[:n_cols-1]):
      ax = fig.add_subplot(n_rows, n_cols, n_cols * row_index + j + 2)
      ax.set_xticks([])
      ax.set_yticks([])

      if train_labels is not None:
        ax.text(x_offset, y_offset - patch_margin,
                '{}'.format(train_labels[i][j]), fontsize=fontsize)

      ax.imshow(train_ia)

  plt.tight_layout(h_pad=1.0)
  plt.show()


def visualize_concepts(train_image_arrays_list, n_cols=10, caption=True):
  """Visualizes concepts.

  Args:
    train_image_arrays_list: list of lists of numpy arrays.
    n_cols: number of columns of the displayed images.
    caption: whether or not to include captions under each row of images.
  """
  # single test image, followed by a row of train images
  n_rows = len(train_image_arrays_list)
  fig = plt.figure(figsize=(n_cols, n_rows))

  image_height = train_image_arrays_list[0][0].shape[0]

  for i, train_ias in enumerate(train_image_arrays_list):
    for j, train_ia in enumerate(train_ias[:n_cols]):
      ax = fig.add_subplot(n_rows, n_cols, n_cols * i + j + 1)
      ax.set_xticks([])
      ax.set_yticks([])

      if caption:
        if j == 0:
          ax.text(
              0,
              int(image_height * 1.2),
              'Top {} training images of concept {}'.format(n_cols, i),
              fontsize=8)
      ax.imshow(train_ia)

  plt.tight_layout(h_pad=1.0)
  plt.show()



