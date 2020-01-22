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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from extrapolation.classifier import classifier
from extrapolation.influence import calculate_influence as ci
from extrapolation.influence import eigenvalues
from extrapolation.utils import dataset_utils
from extrapolation.utils import tensor_utils



class EigenvaluesTest(tf.test.TestCase):

  def test_deterministic_largest_eigenvalue_estimation_correct(self):
    tf.compat.v1.random.set_random_seed(0)
    d_ev = 10
    mat = tf.random.normal((d_ev, d_ev), stddev=1)
    mat = tf.abs((mat + tf.transpose(mat)) / 2.)
    e, v = tf.linalg.eigh(mat)

    def matmul_fn_largest(v):
      return tf.transpose(tf.matmul(mat, v, transpose_b=True))

    largest_ev, largest_evec = eigenvalues.iterated_ev(
        matmul_fn_largest, 200, d_ev, print_freq=50)
    self.assertAllClose(largest_ev, e[-1], atol=0.01)
    self.assertAllClose(tf.abs(largest_evec),
                        tf.abs(tf.reshape(v[:, -1], [1, -1])), atol=0.01)

    evec_product = matmul_fn_largest(largest_evec)
    self.assertAllClose(
        tensor_utils.cosine_similarity(evec_product, largest_evec), 1.,
        atol=0.1)
    self.assertAllClose(evec_product, largest_ev * largest_evec, atol=0.01)

  def test_stochastic_largest_eigenvalue_estimation_correct(self):
    tf.compat.v1.random.set_random_seed(0)
    d_ev = 10
    mat = tf.random.normal((d_ev, d_ev), stddev=1)
    mat = tf.abs((mat + tf.transpose(mat)) / 2.)
    e, v = tf.linalg.eigh(mat)

    def matmul_fn_largest(v):
      mv_prod = tf.transpose(tf.matmul(mat, v, transpose_b=True))
      return mv_prod + tf.random.normal(mv_prod.shape, stddev=0.1)

    largest_ev, largest_evec = eigenvalues.iterated_ev_mean(
        matmul_fn_largest, 2000, d_ev, burnin=1000, print_freq=100)
    self.assertAllClose(largest_ev, e[-1], atol=0.2)
    self.assertAllClose(tf.abs(largest_evec),
                        tf.abs(tf.reshape(v[:, -1], [1, -1])), atol=0.5)

    evec_product = matmul_fn_largest(largest_evec)
    self.assertAllClose(
        tensor_utils.cosine_similarity(evec_product, largest_evec), 1.,
        atol=0.1)
    self.assertAllClose(evec_product, largest_ev * largest_evec, atol=0.5)

  def test_estimate_scaling_correct(self):
    tf.compat.v1.random.set_random_seed(0)
    d_ev = 10
    mat = tf.random.normal((d_ev, d_ev), stddev=1)
    mat = tf.abs((mat + tf.transpose(mat)) / 2.)
    e, v = tf.linalg.eigh(mat)

    def matmul_fn(v):
      return tf.transpose(tf.matmul(mat, v, transpose_b=True))

    def matmul_fn_stochastic(v):
      mv_prod = matmul_fn(v)
      return mv_prod + tf.random.normal(mv_prod.shape, stddev=0.1)

    n_scaling = 1000
    for i in [0, d_ev / 2, -1]:
      eigvec = tf.reshape(v[:, int(i)], [1, -1])
      eigval = e[int(i)]

      est_eigval = eigenvalues.estimate_scaling(eigvec,
                                                matmul_fn_stochastic, n_scaling)
      self.assertAllClose(eigval, est_eigval, 0.1)

      mv_prod = matmul_fn(eigvec)
      self.assertAllClose(mv_prod / est_eigval, eigvec, 0.1)

  def test_largest_and_smallest_eigenvalue_estimation_correct(self):
    tf.compat.v1.random.set_random_seed(0)
    x_shape = (10, 5)
    y_shape = (10, 1)
    conv_dims = []
    conv_sizes = []
    dense_sizes = [5]
    n_classes = 3
    model = classifier.CNN(conv_dims, conv_sizes, dense_sizes, n_classes)
    itr = dataset_utils.get_supervised_batch_noise_iterator(x_shape, y_shape)
    loss_fn = ci.make_loss_fn(model, 1.)
    grad_fn = ci.make_grad_fn(model)
    map_grad_fn = ci.make_map_grad_fn(model)
    x, y = itr.next()
    _, _ = model.get_loss(x, y)

    loss_fn = ci.make_loss_fn(model, None)
    grad_fn = ci.make_grad_fn(model)
    map_grad_fn = ci.make_map_grad_fn(model)

    with tf.GradientTape(persistent=True) as tape:
      # First estimate the Hessian using training data from itr.
      with tf.GradientTape() as tape_inner:
        loss = tf.reduce_mean(loss_fn(x, y))
      grads = grad_fn(loss, tape_inner)
      concat_grads = tf.concat([tf.reshape(w, [-1, 1]) for w in grads], 0)
      hessian_mapped = map_grad_fn(concat_grads, tape)
      # hessian_mapped is a list of n_params x model-shaped tensors
      # should just be able to flat_concat it
      hessian = tensor_utils.flat_concat(hessian_mapped)
    eigs, _ = tf.linalg.eigh(hessian)
    largest_ev, smallest_ev = eigs[-1], eigs[0]

    # We don't know what these eigenvalues should be, but just test that
    # the functions don't crash.
    est_largest_ev = eigenvalues.estimate_largest_ev(
        model, 1000, itr, loss_fn, grad_fn, map_grad_fn, burnin=100)
    est_smallest_ev = eigenvalues.estimate_smallest_ev(
        largest_ev, model, 1000, itr, loss_fn, grad_fn, map_grad_fn, burnin=100)
    self.assertAllClose(largest_ev, est_largest_ev, 0.5)
    self.assertAllClose(smallest_ev, est_smallest_ev, 0.5)

if __name__ == '__main__':
  tf.test.main()
