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

"""Tests for dual_ibp_method."""

import numpy as np
import tensorflow.compat.v1 as tf

from caql import dual_ibp_method

tf.disable_v2_behavior()


class DualIBPMethodTest(tf.test.TestCase):

  def setUp(self):
    super(DualIBPMethodTest, self).setUp()
    self.sess = tf.Session()

  def testCreate_Dual_IBP_Approx(self):
    num_layers = 3
    batch_size = 2
    action_max = 1.0
    action_tensor_center = tf.tile(
        tf.convert_to_tensor(
            np.array([1.0, 2.0, 3.0, 4.0]).reshape([1, 4]).astype(np.float32)),
        [2, 1])
    W_T_list = [
        tf.convert_to_tensor(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
                      [10.0, 11.0, 12.0]]).astype(np.float32)),
        tf.convert_to_tensor(
            np.array([[-1.0], [-3.0], [-5.0]]).astype(np.float32))
    ]
    b_T_list = [
        tf.tile(
            tf.convert_to_tensor(
                np.array([1.0, -0.5, 0.1]).reshape([1, 3]).astype(np.float32)),
            [2, 1]),
        tf.tile(
            tf.convert_to_tensor(
                np.array([2.0]).reshape([1, 1]).astype(np.float32)), [2, 1])
    ]
    (neg_J_tilde, l_list, u_list, D_list, Nu_list, lv_sum, gamma_list, psi,
     Nu_hat_1) = dual_ibp_method.create_dual_ibp_approx(
         num_layers,
         batch_size,
         action_max,
         W_T_list,
         b_T_list,
         action_tensor_center,
         return_full_info=True)

    self.assertIsInstance(neg_J_tilde, tf.Tensor)
    self.assertEqual((2, 1), neg_J_tilde.shape)

    self.assertIsInstance(l_list, list)
    self.assertEqual(num_layers - 1, len(l_list))
    for itr, ele in enumerate(l_list):
      self.assertIsInstance(ele, tf.Tensor)
      if itr == 0:
        self.assertEqual((2, 4), ele.shape)
      elif itr == 1:
        self.assertEqual((2, 3), ele.shape)

    self.assertIsInstance(u_list, list)
    self.assertEqual(num_layers - 1, len(u_list))
    for itr, ele in enumerate(u_list):
      self.assertIsInstance(ele, tf.Tensor)
      if itr == 0:
        self.assertEqual((2, 4), ele.shape)
      elif itr == 1:
        self.assertEqual((2, 3), ele.shape)

    self.assertIsInstance(D_list, list)
    self.assertEqual(num_layers - 1, len(D_list))
    for itr, ele in enumerate(D_list):
      self.assertIsInstance(ele, tf.Tensor)
      if itr == 0:
        self.assertEqual((2, 4), ele.shape)
      elif itr == 1:
        self.assertEqual((2, 3), ele.shape)

    self.assertIsInstance(Nu_list, list)
    self.assertEqual(num_layers - 1, len(Nu_list))
    for itr, ele in enumerate(Nu_list):
      self.assertIsInstance(ele, tf.Tensor)
      if itr == 0:
        self.assertEqual((2, 3, 1), ele.shape)
      elif itr == 1:
        self.assertEqual((2, 3, 1), ele.shape)

    self.assertIsInstance(lv_sum, tf.Tensor)
    self.assertEqual((2, 1), lv_sum.shape)

    self.assertIsInstance(gamma_list, list)
    self.assertEqual(num_layers - 1, len(gamma_list))
    for ele in gamma_list:
      self.assertIsInstance(ele, tf.Tensor)
      self.assertEqual((2, 1), ele.shape)

    self.assertIsInstance(psi, tf.Tensor)
    self.assertEqual((2, 1), psi.shape)
    self.assertIsInstance(Nu_hat_1, tf.Tensor)
    self.assertEqual((2, 4, 1), Nu_hat_1.shape)

    neg_J_tilde_np = self.sess.run(neg_J_tilde)
    l_list_np = self.sess.run(l_list)
    u_list_np = self.sess.run(u_list)
    D_list_np = self.sess.run(D_list)
    Nu_list_np = self.sess.run(Nu_list)
    lv_sum_np = self.sess.run(lv_sum)
    gamma_list_np = self.sess.run(gamma_list)
    psi_np = self.sess.run(psi)
    Nu_hat_1_np = self.sess.run(Nu_hat_1)

    self.assertArrayNear(
        np.array([[-508.], [-508.]]).flatten(),
        neg_J_tilde_np.flatten(),
        err=1e-4)
    for itr, ele in enumerate(l_list_np):
      if itr == 0:
        print(ele)
        self.assertArrayNear(
            np.array([[0., 1., 2., 3.], [0., 1., 2., 3.]]).flatten(),
            ele.flatten(),
            err=1e-4)
      elif itr == 1:
        self.assertArrayNear(
            np.array([[49., 53.5, 60.1], [49., 53.5, 60.1]]).flatten(),
            ele.flatten(),
            err=1e-4)
    for itr, ele in enumerate(u_list_np):
      if itr == 0:
        self.assertArrayNear(
            np.array([[2., 3., 4., 5.], [2., 3., 4., 5.]]).flatten(),
            ele.flatten(),
            err=1e-4)
      elif itr == 1:
        self.assertArrayNear(
            np.array([[93., 105.5, 120.1], [93., 105.5, 120.1]]).flatten(),
            ele.flatten(),
            err=1e-4)
    for itr, ele in enumerate(D_list_np):
      if itr == 0:
        self.assertArrayNear(
            np.array([[0., 0., 0., 0.], [0., 0., 0., 0.]]).flatten(),
            ele.flatten(),
            err=1e-4)
      elif itr == 1:
        self.assertArrayNear(
            np.array([[1., 1., 1.], [1., 1., 1.]]).flatten(),
            ele.flatten(),
            err=1e-4)
    for itr, ele in enumerate(Nu_list_np):
      if itr == 0:
        self.assertArrayNear(
            np.array([[[0.], [0.], [0.]], [[0.], [0.], [0.]]]).flatten(),
            ele.flatten(),
            err=1e-4)
      elif itr == 1:
        self.assertArrayNear(
            np.array([[[-1.], [-3.], [-5.]], [[-1.], [-3.], [-5.]]]).flatten(),
            ele.flatten(),
            err=1e-4)
    self.assertArrayNear(
        np.array([[0.], [0.]]).flatten(), lv_sum_np.flatten(), err=1e-4)
    for itr, ele in enumerate(gamma_list_np):
      if itr == 0:
        self.assertArrayNear(
            np.array([[0.], [0.]]).flatten(), ele.flatten(), err=1e-4)
      elif itr == 1:
        self.assertArrayNear(
            np.array([[2.], [2.]]).flatten(), ele.flatten(), err=1e-4)

    self.assertArrayNear(
        np.array([[-758.], [-758.]]).flatten(), psi_np.flatten(), err=1e-4)
    self.assertArrayNear(
        np.array([[[-22.], [-49.], [-76.], [-103.]],
                  [[-22.], [-49.], [-76.], [-103.]]]).flatten(),
        Nu_hat_1_np.flatten(),
        err=1e-4)


if __name__ == '__main__':
  tf.test.main()
