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

"""Methods to generate explanations for batches of samples and predictions."""

from concurrent import futures
import multiprocessing
import os

from absl import logging

import matplotlib.pyplot as plt
import numpy as np
import saliency
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile

from invariant_explanations import config


def get_model_explanations_for_instances(model, samples, y_preds,
                                         explanation_method):
  """Extract explanations of a certain type for samples on model.

  Args:
    model: the keras model for which explanations are to be generated.
    samples: samples used to train each base model; instance of np.ndarray.
    y_preds: the predicted target of samples on each base-model;
             instance of np.ndarray.
    explanation_method: string name of explanation method used for generation.

  Returns:
    explans: the model explanations for all samples; instance of np.ndarray.
  """

  # While explanation methods can work on inputs sampled from arbitrary domains,
  # they work best when the image is sampled from the same domain upon which the
  # model was trained.
  if not ((-1 <= np.min(samples)) and (np.max(samples) <= +1)):
    raise ValueError('Sample images must be in [-1,1] domain for explaination.')
  #   samples = (samples + 1) / 2 * 255 # Omit; original domain was [-1,1].

  model = tf.keras.models.Model(
      [model.inputs],
      [
          model.get_layer('conv2d_2').output,
          model.output,
      ],
  )

  # Determine the shape of gradients by calling the model with a single input
  # and extracting the shape from the gradient output.
  model(np.expand_dims(samples[0], 0))

  # Source: github.com/PAIR-code/saliency/blob/master/Examples_core.ipynb
  class_idx_str = 'class_idx_str'
  def _call_model_function(images, call_model_args=None, expected_keys=None):
    target_class_idx = call_model_args[class_idx_str]
    images = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
      if expected_keys == [saliency.core.base.INPUT_OUTPUT_GRADIENTS]:
        tape.watch(images)
        _, output_layer = model(images)
        output_layer = output_layer[:, target_class_idx]
        gradients = np.array(tape.gradient(output_layer, images))
        return {saliency.core.base.INPUT_OUTPUT_GRADIENTS: gradients}
      else:
        conv_layer, output_layer = model(images)
        gradients = np.array(tape.gradient(output_layer, conv_layer))
        return {
            saliency.core.base.CONVOLUTION_LAYER_VALUES: conv_layer,
            saliency.core.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients
        }

  def _get_explanation(inputs):
    """Method to call the corresponding explanation method from saliency module.

    Args:
      inputs: tuple of (sample, y_pred) for which an explanation is sought.

    Returns:
      explan: the model explanation for the input tuple; instance of np.ndarray.
    """

    sample, y_pred = inputs[0], inputs[1]
    sample = sample.squeeze()
    if sample.ndim != 2:
      raise ValueError('Expected grayscale samples.')

    baseline = np.zeros(sample.shape)  # Baseline is a black image.

    # Construct a shared list of function arguments to be called below.
    # The kwargs here are shared among all methods below; method specific kwargs
    # are additionally passed in when invocation each specific method.
    call_model_kwargs = {
        'x_value': sample,
        'call_model_function': _call_model_function,
        'call_model_args': {class_idx_str: y_pred},
    }

    # Select among types of explanation methods.
    if explanation_method == 'grad':
      explan = saliency.core.GradientSaliency().GetMask(
          **call_model_kwargs,
      )
    elif explanation_method == 'smooth_grad':
      explan = saliency.core.GradientSaliency().GetSmoothedMask(
          **call_model_kwargs,
      )
    elif explanation_method == 'gradcam':
      explan = saliency.core.GradCam().GetMask(
          **call_model_kwargs,
          three_dims=False,
      )
    elif explanation_method == 'smooth_gradcam':
      explan = saliency.core.GradCam().GetSmoothedMask(
          **call_model_kwargs,
          three_dims=False,
      )
    elif explanation_method == 'ig':
      explan = saliency.core.IntegratedGradients().GetMask(
          **call_model_kwargs,
          x_steps=25,
          x_baseline=baseline,
          batch_size=20,
      )
    elif explanation_method == 'smooth_ig':
      explan = saliency.core.IntegratedGradients().GetSmoothedMask(
          **call_model_kwargs,
          x_steps=25,
          x_baseline=baseline,
          batch_size=20,
      )
    elif explanation_method == 'guided_ig':
      explan = saliency.core.GuidedIG().GetMask(
          **call_model_kwargs,
          x_steps=25,
          x_baseline=baseline,
          max_dist=1.0,
          fraction=0.5,
      )

    else:
      raise NotImplementedError

    assert explan.ndim == 2, 'Expected grayscale explans.'

    # Perform percentile clipping to remove outliers, then return explan.
    # To not repeat code, you may use the saliency library as below, but
    # first construct a 3D copy of the explan by tiling across all 3 channels.
    explan = np.expand_dims(explan, axis=2)
    explan = np.tile(explan, [1, 1, 3])
    return saliency.core.VisualizeImageGrayscale(explan)

  # Process explans in parallel. Code below inspired by:
  # https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor
  # Important: confirmed that order is preserved when calling map() in parallel.
  max_workers = multiprocessing.cpu_count() * 5
  with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    explans = np.array([
        *executor.map(
            _get_explanation,
            zip(
                samples,
                np.argmax(y_preds, axis=1),
            ),
        )
    ])

  return explans


def plot_and_save_various_explanations(model, samples, y_preds, save_file_name):
  """Method to iterate over and plot out all explantions methods for a batch.

  Args:
    model: the keras model for which explanations are to be generated.
    samples: samples used to train each base model; instance of np.ndarray.
    y_preds: the predicted target of samples on each base-model;
             instance of np.ndarray.
    save_file_name: file name to be used to save plots of explanations.
  """

  num_rows = len(config.ALLOWABLE_EXPLANATION_METHODS) + 1  # +1 for samples
  num_cols = len(samples)
  plt.rc('font', size=6)
  fig, axes = plt.subplots(
      num_rows,
      num_cols,
      figsize=(num_cols * 1.5, num_rows),
  )
  axes = axes.flatten()

  for col_idx in range(num_cols):
    axes[col_idx + 0 * num_cols].imshow(samples[col_idx])
    axes[col_idx + 0 * num_cols].axis('off')
    axes[col_idx + 0 * num_cols].set_title('sample')

  for explan_idx, row_idx in enumerate(range(1, num_rows)):
    explanation_method = config.ALLOWABLE_EXPLANATION_METHODS[explan_idx]
    logging.info('Processing %s explanations.', explanation_method)
    explans = get_model_explanations_for_instances(
        model,
        samples,
        y_preds,
        explanation_method,
    )
    for col_idx in range(num_cols):
      axes[col_idx + row_idx * num_cols].imshow(explans[col_idx])
      axes[col_idx + row_idx * num_cols].axis('off')
      axes[col_idx + row_idx * num_cols].set_title(explanation_method)

  fig.savefig(
      gfile.GFile(
          os.path.join(
              config.PLOTS_DIR_PATH,
              save_file_name,
          ),
          'wb',
      ),
      dpi=400,
  )
