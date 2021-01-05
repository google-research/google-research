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

"""Calculate influence of test points (in and OOD) on trained model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import logging
import numpy as np
import scipy
import tensorflow.compat.v1 as tf
from extrapolation.classifier import classifier
from extrapolation.influence import calculate_influence
from extrapolation.utils import dataset_utils
from extrapolation.utils import tensor_utils
from extrapolation.utils import utils


EPS = 1e-10


def conjugate_gradient_optimize(
    objective, x_init, objective_gradient, hessian_vector_product,
    callback, maxiter=None, tol=1e-5):
  conjugate_gradient_result, _, _, _, _, warning_flag = scipy.optimize.fmin_ncg(
      f=objective, x0=x_init, fprime=objective_gradient,
      fhess_p=hessian_vector_product, callback=callback, maxiter=maxiter,
      avextol=tol, full_output=True)
  return conjugate_gradient_result, warning_flag


def get_ihvp_conjugate_gradient(vec, itr, loss_function, gradient_function,
                                map_gradient_function, approx_params):
  """Calculate the inverse HVP of vec with the Hessian of loss_function.

  Note: HVP stands for Hessian-vector product.
  Let n be the number of examples in this batch.
  Let p be the number of parameters in the model which defines loss_function.
  Let the model's parameter output (model.weights) have shapes
    [s0, s1, ... s_k].
  Then vec is a list of tensors with shapes [(n, s0), (n, s1) ... (n, s_k)].

  Uses the Scipy implementation of conjugate gradient descent.

  Args:
    vec (list of tensors): the vector in our HVP, shape described above.
    itr (Iterator): an iterator of data we will use to estimate the Hessian.
    loss_function (function): a function which returns a gradient of losses.
    gradient_function (function): a function which takes the gradient of a
                                  scalar loss.
    map_gradient_function (function): a function which takes the gradient of
                                      each element of a vector of losses.
    approx_params (dict): parameters for conjugate gradient optimization.
  Returns:
    conjugate_gradient_result (tensor): a (n x p) tensor containing the desired
                                        IHVP.
  """
  # Reshape/concatenate our input so concat_vec has shape (n, p).
  concat_vec = tensor_utils.flat_concat(vec)
  def get_hvp(v):
    """Get the Hessian-vector product of v and Hessian in loss_function.

    Args:
      v (vector): a (n * p,)-shaped vector we want to multiply with H.
    Returns:
      hvp (vector): a (n, p)-shaped matrix representing Hv.
    """
    v = tf.reshape(v, concat_vec.shape)
    v = tensor_utils.reshape_vector_as([el[0] for el in vec], v)
    v_hvp = calculate_influence.hvp(v, itr,
                                    loss_function, gradient_function,
                                    map_gradient_function,
                                    n_samples=approx_params['hvp_samples'])
    if approx_params['squared']:
      v_hvp = calculate_influence.hvp(v_hvp, itr,
                                      loss_function, gradient_function,
                                      map_gradient_function,
                                      n_samples=approx_params['hvp_samples'])
    return tensor_utils.flat_concat(v_hvp)

  def objective(v):
    flat_v_hvp = get_hvp(v)
    v = tf.reshape(v, concat_vec.shape)
    objective_value = (0.5 * tf.reduce_sum(tf.multiply(v, flat_v_hvp), axis=1)
                       - tf.reduce_sum(tf.multiply(concat_vec, v), axis=1))
    logging.info('Evaluating objective: obj = {:.3f}'.
                 format(tf.reduce_mean(objective_value).numpy()))
    return tf.reduce_mean(objective_value)

  def objective_gradient(v):
    flat_v_hvp = get_hvp(v)
    grads = flat_v_hvp - concat_vec
    logging.info('Evaluating gradients: norm(grads) = {:.3f}'.
                 format(tf.linalg.norm(grads).numpy()))
    return tf.reshape(grads, [-1])

  def hessian_vector_product(_, v):
    s = time.time()
    hvp = get_hvp(v)
    t = time.time()
    logging.info('Evaluating Hessian: norm(hvp) = {:.3f} ({:.3f} seconds)'.
                 format(tf.linalg.norm(hvp).numpy(), t - s))
    return tf.reshape(hvp, [-1])

  def callback(v):
    hvp = get_hvp(v)
    err = tf.reduce_mean(tf.math.reduce_euclidean_norm(hvp - concat_vec,
                                                       axis=1))
    logging.info('Current error is: {:.4f}; Obj = {:.4f}, vnorm = {:.4f}'
                 .format(err, objective(v),
                         tf.math.reduce_euclidean_norm(v).numpy()))

  x_init = tf.reshape(concat_vec, [-1])
  conjugate_gradient_result, warning_flag = conjugate_gradient_optimize(
      objective, x_init, objective_gradient, hessian_vector_product,
      callback, maxiter=approx_params['maxiter'],
      tol=approx_params['tol'])
  conjugate_gradient_result = tf.reshape(conjugate_gradient_result,
                                         concat_vec.shape)
  return conjugate_gradient_result, warning_flag


def get_parameter_influence(model, x, y, itr,
                            approx_params=None, damping=None):
  """Estimate the influence of test examples (x, y) on the parameters of model.

  Args:
    model (Classifier): a classification model whose parameters we are
      interested in.
    x (tensor): the input data whose influence we are interested in.
    y (tensor): the target data whose influence we are interested in.
    itr (Iterator): an iterator of data we will use to estimate the Hessian.
    approx_params (dict, optional): parameters for running LiSSA.
    damping (float, optional): the amount of L2-regularization to add
      to the parameters of model (only used for conjugate gradient).
  Returns:
    ihvp_result (tensor): the HVP of the inverse hessian of model (possibly with
      some L2-regularization) with the gradient of (x, y) w.r.t the
      parameters of model.
    concat_grads (tensor): the gradients of (x, y) w.r.t the model parameters.
    warning_flag (int): a flag representing if this optimization terminated
                        successfully, returned by Scipy.
  """
  loss_function = calculate_influence.make_loss_fn(model, damping)
  gradient_function = calculate_influence.make_grad_fn(model)
  map_gradient_function = calculate_influence.make_map_grad_fn(model)
  grads = calculate_influence.get_loss_grads(x, y, loss_function,
                                             map_gradient_function)
  concat_grads = tensor_utils.flat_concat(grads)
  ihvp_result, warning_flag = get_ihvp_conjugate_gradient(
      grads, itr, loss_function, gradient_function, map_gradient_function,
      approx_params)
  return ihvp_result, concat_grads, warning_flag


def calculate_influence_ood(params):
  """Calculates influence functions for pre-trained model with OOD classes.

  Args:
    params (dict): contains a number of params - as loaded from flags.
    Should contain:
      seed (int) - random seed for Tensorflow and Numpy initialization.
      training_results_dir (str) - parent directory of the pre-trained model.
      clf_name (str) - the name of the pre-trained model's directory.
      n_test_infl (int) - number of examples to run influence functions for.
      start_ix_test_infl (int) - index to start loading examples from.
      cg_maxiter (int) - max number of iterations for conjugate gradient.
      squared (bool) - whether to calculate squared Hessian directly.
      tol (float) - tolerance for conjugate gradient.
      lam (float) - L2 regularization amount for Hessian.
      hvp_samples (int) - number of samples to take in HVP estimation.
      output_dir (str) - where results should be written - defaults to
        training_results_dir/clf_name/influence_results.
      tname (str) - extra string to add to saved tensor names; can be ''.
      preloaded_model (model or None) - if None, we should load the model
        ourselves. Otherwise, preloaded_model is the model we are interested in.
      preloaded_itr (Iterator or None) - if None, load the data iterator
        ourselves; otherwise, use preloaded_itr as the data iterator.
  """

  tf.set_random_seed(params['seed'])
  np.random.seed(params['seed'])

  # Load a trained classifier.
  modeldir = os.path.join(params['training_results_dir'], params['clf_name'])
  param_file = os.path.join(modeldir, 'params.json')
  model_params = utils.load_json(param_file)

  if params['preloaded_model'] is None:
    ckpt_path = os.path.join(modeldir, 'ckpts/bestmodel-1')
    cnn_args = {'conv_dims':
                    [int(x) for x in model_params['conv_dims'].split(',')],
                'conv_sizes':
                    [int(x) for x in model_params['conv_sizes'].split(',')],
                'dense_sizes':
                    [int(x) for x in model_params['dense_sizes'].split(',')],
                'n_classes': model_params['n_classes'], 'onehot': True}
    model = utils.load_model(ckpt_path, classifier.CNN, cnn_args)
  else:
    model = params['preloaded_model']

  # Load train/validation/test examples
  tensordir = os.path.join(modeldir, 'tensors')
  validation_x = utils.load_tensor(os.path.join(tensordir, 'valid_x_infl.npy'))
  test_x = utils.load_tensor(os.path.join(tensordir, 'test_x_infl.npy'))
  ood_x = utils.load_tensor(os.path.join(tensordir, 'ood_x_infl.npy'))

  # Get in- and out-of-distribution classes.
  n_labels = model_params['n_classes']
  all_classes = range(n_labels)
  ood_classes = ([int(x) for x in model_params['ood_classes'].split(',')]
                 if 'ood_classes' in model_params else [])
  ind_classes = [x for x in all_classes if x not in ood_classes]

  # Load an iterator of training data.
  label_noise = (model_params['label_noise']
                 if 'label_noise' in model_params else 0.)

  # We only look at a portion of the test set for computational reasons.
  ninfl = params['n_test_infl']
  start_ix = params['start_ix_test_infl']
  end_ix = start_ix + ninfl
  xinfl_validation = validation_x[start_ix: end_ix]
  xinfl_test = test_x[start_ix: end_ix]
  xinfl_ood = ood_x[start_ix: end_ix]

  # We want to rotate through all the label options.
  y_all = tf.concat([tf.one_hot(tf.fill((ninfl,), lab), depth=n_labels)
                     for lab in ind_classes], axis=0)
  y_all = tf.concat([y_all, y_all, y_all], axis=0)

  xinfl_validation_all = tf.concat([xinfl_validation for _ in ind_classes],
                                   axis=0)
  xinfl_test_all = tf.concat([xinfl_test for _ in ind_classes], axis=0)
  xinfl_ood_all = tf.concat([xinfl_ood for _ in ind_classes], axis=0)
  x_all = tf.concat([xinfl_validation_all, xinfl_test_all, xinfl_ood_all],
                    axis=0)

  cg_approx_params = {'maxiter': params['cg_maxiter'],
                      'squared': params['squared'],
                      'tol': params['tol'],
                      'hvp_samples': params['hvp_samples']}

  # Here we run conjugate gradient one example at a time, collecting
  # the following outputs.

  # H^{-1}g
  infl_value = []
  # gH^{-1}g
  infl_laplace = []
  # H^{-2}g
  infl_deriv = []
  # g
  grads = []
  # When calculating H^{-1}g with conjugate gradient, Scipy returns a flag
  # denoting the optimization's success.
  warning_flags = []
  # When calculating H^{-2}g with conjugate gradient, Scipy returns a flag
  # denoting the optimization's success.
  warning_flags_deriv = []

  for i in range(x_all.shape[0]):
    logging.info('Example {:d}'.format(i))
    s = time.time()
    xi = tf.expand_dims(x_all[i], 0)
    yi = tf.expand_dims(y_all[i], 0)
    if params['preloaded_itr'] is None:
      itr_train, _, _, _ = dataset_utils.load_dataset_ood_supervised_onehot(
          ind_classes, ood_classes, label_noise=label_noise)
    else:
      itr_train = params['preloaded_itr']
    infl_value_i, grads_i, warning_flag_i = get_parameter_influence(
        model, xi, yi, itr_train,
        approx_params=cg_approx_params,
        damping=params['lam'])
    t = time.time()
    logging.info('IHVP calculation took {:.3f} seconds'.format(t - s))
    infl_laplace_i = tf.multiply(infl_value_i, grads_i)

    infl_value_wtshape = tensor_utils.reshape_vector_as(model.weights,
                                                        infl_value_i)
    loss_function = calculate_influence.make_loss_fn(model, params['lam'])
    gradient_function = calculate_influence.make_grad_fn(model)
    map_gradient_function = calculate_influence.make_map_grad_fn(model)
    s = time.time()
    infl_deriv_i, warning_flag_deriv_i = get_ihvp_conjugate_gradient(
        infl_value_wtshape, itr_train,
        loss_function, gradient_function, map_gradient_function,
        approx_params=cg_approx_params)
    t = time.time()
    logging.info('Second IHVP calculation took {:.3f} seconds'.format(t - s))
    infl_value.append(infl_value_i)
    infl_laplace.append(infl_laplace_i)
    infl_deriv.append(infl_deriv_i)
    grads.append(grads_i)
    warning_flags.append(tf.expand_dims(warning_flag_i, 0))
    warning_flags_deriv.append(tf.expand_dims(warning_flag_deriv_i, 0))

  infl_value = tf.concat(infl_value, axis=0)
  infl_laplace = tf.concat(infl_laplace, axis=0)
  infl_deriv = tf.concat(infl_deriv, axis=0)
  grads = tf.concat(grads, axis=0)
  warning_flags = tf.concat(warning_flags, axis=0)
  warning_flags_deriv = tf.concat(warning_flags_deriv, axis=0)

  res = {}
  for infl_res, nm in [(infl_value, 'infl'),
                       (infl_deriv, 'deriv'),
                       (infl_laplace, 'laplace'),
                       (grads, 'grads'),
                       (warning_flags, 'warnflags'),
                       (warning_flags_deriv, 'warnflags_deriv')]:
    res['valid_{}'.format(nm)] = infl_res[:ninfl * len(ind_classes)]
    res['test_{}'.format(nm)] = infl_res[
        ninfl * len(ind_classes): 2 * ninfl * len(ind_classes)]
    res['ood_{}'.format(nm)] = infl_res[2 * ninfl * len(ind_classes):]

  # Save the results of these calculations.
  if params['output_dir']:
    resdir = utils.make_subdir(params['output_dir'], 'influence_results')
  else:
    resdir = utils.make_subdir(modeldir, 'influence_results')
  tensor_name_template = '{}{}-inv_hvp-cg-ix{:d}-ninfl{:d}'+ (
      '_squared' if params['squared'] else '')
  infl_tensors = [
      (tensor_name_template.format(params['tname'], label, start_ix, ninfl),
       res[label]) for label in res.keys()]
  utils.save_tensors(infl_tensors, resdir)
