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

"""Calculating influence functions for a pre-trained classifier.

"Understanding Black-Box Predictions via Influence Functions", Koh & Liang
https://arxiv.org/abs/1703.04730

Some code adapted from https://github.com/kohpangwei/influence-release/blob/
master/influence/genericNeuralNet.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from extrapolation.classifier import classifier
from extrapolation.utils import dataset_utils
from extrapolation.utils import tensor_utils
from extrapolation.utils import utils



def get_loss_grads(x, y, loss_fn, map_grad_fn):
  with tf.GradientTape(persistent=True) as tape:
    loss = loss_fn(x, y)
    grads = map_grad_fn(loss, tape)
  return grads


def hvp(v, iterator, loss_fn, grad_fn, map_grad_fn, n_samples=1):
  """Multiply the Hessian of clf at inputs (x, y) by vector v.

  Args:
    v (tensor): the vector in the HVP.
    iterator (Iterator): iterator for samples for HVP estimation.
    loss_fn (function): a function which returns a gradient of losses.
    grad_fn (function): a function which takes the gradient of a scalar loss.
    map_grad_fn (function): a function which takes the gradient of each element
                            of a vector of losses.
    n_samples (int, optional): number of minibatches to sample
                               when estimating Hessian
  Returns:
    hessian_vector_val (tensor): the HVP of clf's Hessian with v.
  """

  # tf.GradientTape tracks the operations you take while inside it, in order to
  # later auto-differentiate through those operations to get gradients.
  with tf.GradientTape(persistent=True) as tape2:

    # We need two gradient tapes to calculate second derivatives
    with tf.GradientTape() as tape:
      loss = 0.
      for _ in range(n_samples):
        x_sample, y_sample = iterator.next()
        loss += tf.reduce_mean(loss_fn(x_sample, y_sample))

    # Outside the tape, we can get the aggregated loss gradient across the
    # batch. This is the standard usage of GradientTape.
    grads = grad_fn(loss, tape)

    # For each weight matrix, we now get the product of the vector v with
    # the gradient, and the sum over the weights to get a total gradient
    # per element in x.
    vlist = []
    for g, u in zip(grads, v):
      g = tf.expand_dims(g, 0)
      prod = tf.multiply(g, u)
      vec = tf.reduce_sum(prod, axis=range(1, prod.shape.rank))
      vlist.append(vec)
    vgrads = tf.add_n(vlist)

    # We now take the gradient of the gradient-vector product. This gives us
    # the Hessian-vector product. Note that we take this gradient inside
    # the tape - this allows us to get the HVP value for each element of x.
    hessian_vector_val = map_grad_fn(vgrads, tape2)
  return hessian_vector_val


def get_inverse_hvp(v, itr_train, loss_fn, grad_fn, map_grad_fn,
                    scale=10, damping=0.0, num_samples=1,
                    recursion_depth=10000, print_iter=1):
  """Calculates an HVP, getting the inverse Hessian using the LISSA method.

  Args:
    v (Tensor): the vector in the HVP - the gradient on the test example.
    itr_train (Iterator): iterator for getting batches to estimate H.
    loss_fn (function): a function which returns a gradient of losses.
    grad_fn (function): a function which takes the gradient of a scalar loss.
    map_grad_fn (function): a function which takes the gradient of each element
                            of a vector of losses.
    scale (number): scales eigenvalues of Hessian to be <= 1.
    damping (number): in [0, 1), LISSA parameter - higher is more stable.
    num_samples (int): how many times to estimate the HVP.
    recursion_depth (int): how many steps in LISSA optimization.
    print_iter (int): how frequently to print LISSA updates.
  Returns:
    inverse_hvp (Tensor): the estimated product of the inverse Hessian
                          of clf and v.
  """

  value_constraints = [(scale, 'scale', lambda x: x > 0, 'greater than 0'),
                       (damping, 'damping', lambda x: 0 <= x <= 1,
                        'between 0 and 1 inclusive'),
                       (num_samples, 'num_samples',
                        lambda x: x > 0, 'greater than 0'),
                       (recursion_depth, 'recursion_depth',
                        lambda x: x > 0, 'greater than 0')]
  for var, vname, constraint, msg in value_constraints:
    if not constraint(var):
      raise ValueError('{} should be {}'.format(vname, msg))

  inverse_hvp = [tf.zeros_like(b) for b in v]
  for _ in range(num_samples):
    cur_estimate = v
    logging.info('cur estimate: %s', str([c.shape for c in cur_estimate]))

    for j in range(recursion_depth):
      old_estimate = cur_estimate
      hessian_vector_val = hvp(cur_estimate, itr_train,
                               loss_fn, grad_fn, map_grad_fn)

      cur_estimate = [a + (1-damping) * b - c / scale for (a, b, c)
                      in zip(v, cur_estimate, hessian_vector_val)]

      if (j % print_iter == 0) or (j == recursion_depth - 1):
        logging.info('Recursion at depth %d: norm is %.8lf, diff is %.8lf',
                     j,
                     np.linalg.norm(
                         tensor_utils.flat_concat(cur_estimate).numpy()),
                     np.linalg.norm(
                         tensor_utils.flat_concat(cur_estimate).numpy() -
                         tensor_utils.flat_concat(old_estimate).numpy()))
      old_estimate = cur_estimate
    inverse_hvp = [a + b / scale
                   for (a, b) in zip(inverse_hvp, cur_estimate)]

  inverse_hvp = [a / num_samples for a in inverse_hvp]
  return inverse_hvp


def make_loss_fn(clf, lam):
  """Return a function which returns a vector of per-examples losses.

  Args:
    clf (Classifier): the classifier whose loss we are interested in.
    lam (float): optional L2 regularization parameter.
  Returns:
    f (function): a function which runs clf on input x and output y and returns
                  a vector of losses, one for each element in x.
  """
  if lam is None:
    def f(x, y):
      train_loss, _ = clf.get_loss(x, y)
      return train_loss
  else:
    def f(x, y):
      train_loss = clf.get_loss_dampened(x, y, lam=lam)
      return train_loss
  return f


def make_grad_fn(clf):
  """Return a function which takes the gradient of a loss.

  Args:
    clf (Classifier): the classifier whose gradient we are interested in.
  Returns:
    f (function): a function which takes a scalar loss and GradientTape and
                  returns the gradient of loss w.r.t clf.weights.
  """
  def f(loss, tape):
    return tape.gradient(loss, clf.weights)
  return f


def make_map_grad_fn(clf):
  """Return a function which takes the gradient of each element of loss vector.

  Args:
    clf (Classifier): the classifier whose gradient we are interested in.
  Returns:
    f (function): a function which takes a vector v and a GradientTape and
                  takes the gradient on the tape for each element of v.
  """
  def f(v, tape):
    return tf.map_fn(lambda l: tape.gradient(l, clf.weights), v,
                     dtype=tf.nest.map_structure(
                         lambda x: x.dtype, clf.weights))
  return f


def get_influence_on_test_loss(test_batch, train_batch, itr_train, clf,
                               approx_params=None, lam=None):
  """Calculate influence of examples in train_batch on examples in test_batch.

  Args:
    test_batch (tensor): examples to calculate influence on.
    train_batch (tensor): examples to calculate influence of.
    itr_train (Iterator): where to sample data for estimating Hessian from.
    clf (Classifier): the classifier we are interested in influence for.
    approx_params (dict): optional parameters for LISSA optimization.
    lam (float): optional L2-regularization for Hessian estimation.
  Returns:
    predicted_loss_diffs (tensors): n_test x n_train tensor, with entry
      (i, j) the influence of train example j on test example i.
  """

  loss_fn = make_loss_fn(clf, None)
  reg_loss_fn = make_loss_fn(clf, lam)
  grad_fn = make_grad_fn(clf)
  map_grad_fn = make_map_grad_fn(clf)

  x, y = test_batch
  test_grad_loss_no_reg_val = get_loss_grads(x, y, loss_fn, map_grad_fn)

  if approx_params is None:
    approx_params = {}
  inverse_hvp = get_inverse_hvp(
      test_grad_loss_no_reg_val, itr_train, reg_loss_fn, grad_fn, map_grad_fn,
      **approx_params)

  xtr, ytr = train_batch
  train_grad_loss_val = get_loss_grads(xtr, ytr, loss_fn, map_grad_fn)
  predicted_loss_diffs = tf.matmul(
      tensor_utils.flat_concat(inverse_hvp),
      tensor_utils.flat_concat(train_grad_loss_val), transpose_b=True)
  return predicted_loss_diffs


def convert_index(i, d):
  ncols = d.shape[0]
  xval = i // ncols
  yval = i % ncols
  return xval, yval


def run(params):
  """Calculates influence functions for a pre-loaded model.

  params should contain:
    seed (int): random seed for Tensorflow and Numpy initialization.
    training_results_dir (str): the parent directory of the pre-trained model.
    clf_name (str): the name of the pre-trained model's directory.
    n_test_infl (int): number of test examples to run influence functions for.
    n_train_infl (int): number of train examples to run influence functions for.
    lissa_recursion_depth (int): how long to run LiSSA for.
    output_dir (str): where results should be written - defaults to
      training_results_dir/clf_name/influence_results.
  Args:
    params (dict): contains a number of params - as loaded from flags.
  """
  tf.set_random_seed(params['seed'])
  np.random.seed(params['seed'])

  # Load a trained classifier.
  modeldir = os.path.join(params['training_results_dir'], params['clf_name'])
  ckpt_path = os.path.join(modeldir, 'ckpts/bestmodel-1')
  param_file = os.path.join(modeldir, 'params.json')
  clf_params = utils.load_json(param_file)

  cnn_args = {'conv_dims':
                  [int(x) for x in clf_params['conv_dims'].split(',')],
              'conv_sizes':
                  [int(x) for x in clf_params['conv_sizes'].split(',')],
              'dense_sizes':
                  [int(x) for x in clf_params['dense_sizes'].split(',')],
              'n_classes': 10, 'onehot': True}
  clf = utils.load_model(ckpt_path, classifier.CNN, cnn_args)

  # Load train/valid/test examples
  tensordir = os.path.join(modeldir, 'tensors')
  train_x = utils.load_tensor(os.path.join(tensordir, 'train_x_infl.npy'))
  train_y = utils.load_tensor(os.path.join(tensordir, 'train_y_infl.npy'))
  test_x = utils.load_tensor(os.path.join(tensordir, 'test_x_infl.npy'))
  test_y = utils.load_tensor(os.path.join(tensordir, 'test_y_infl.npy'))

  # Calculate influence functions for this model.
  if params['output_dir']:
    resdir = utils.make_subdir(params['output_dir'], 'influence_results')
  else:
    resdir = utils.make_subdir(modeldir, 'influence_results')

  itr_train, _, _ = dataset_utils.load_dataset_supervised_onehot()
  bx, by = test_x[:params['n_test_infl']], test_y[:params['n_test_infl']]
  train_loss, _ = clf.get_loss(bx, by)
  logging.info('Current loss: {:3f}'.format(tf.reduce_mean(train_loss)))
  bx_tr, by_tr = (train_x[:params['n_train_infl']],
                  train_y[:params['n_train_infl']])
  predicted_loss_diffs = get_influence_on_test_loss(
      (bx, by), (bx_tr, by_tr), itr_train, clf,
      approx_params={'scale': 100, 'damping': 0.1, 'num_samples': 1,
                     'recursion_depth': params['lissa_recursion_depth'],
                     'print_iter': 10}, lam=0.01)
  utils.save_tensors([('predicted_loss_diffs', predicted_loss_diffs)], resdir)

  d = predicted_loss_diffs.numpy()
  df = d.flatten()
  dfsort = sorted(zip(df, range(len(df))), key=lambda x: abs(x[0]),
                  reverse=True)

  # Ordering note: df[:10] == d[0,:10]
  # Loop over the highest influence pairs.
  imgs_to_save = []
  titles = []
  n_img_pairs = 10
  for inf_val, idx in dfsort[:n_img_pairs]:
    i, tr_i = convert_index(idx, d)
    imgs_to_save += [bx[i, :, :, 0], bx_tr[tr_i, :, :, 0]]
    titles += ['Test Image: inf={:.8f}'.format(inf_val),
               'Train Image: inf={:.8f}'.format(inf_val)]
  utils.save_images(os.path.join(resdir, 'most_influential.pdf'),
                    imgs_to_save, n_img_pairs, titles=titles)

  # For each test image, find the training image with the highest influence
  imgs_to_save = []
  titles = []
  n_img_pairs = 10  # Do this many pairs from the top of the test tensor
  for i in range(n_img_pairs):
    infl_i = d[i, :]
    max_train_ind = np.argmax(infl_i)
    max_infl_val = max(infl_i)
    imgs_to_save += [bx[i, :, :, 0], bx_tr[max_train_ind, :, :, 0]]
    titles += ['Test Image {:d}: inf={:.8f}'.format(i, max_infl_val),
               'Train Image {:d}: inf={:.8f}'.format(
                   max_train_ind, max_infl_val)]
  utils.save_images(os.path.join(resdir, 'most_influential_by_test_img.pdf'),
                    imgs_to_save, n_img_pairs, titles=titles)
