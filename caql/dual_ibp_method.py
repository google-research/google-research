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

"""Dual IBP functions."""
import tensorflow.compat.v1 as tf

from caql import dual_method


def create_dual_ibp_approx(num_layers, batch_size, action_max, W_T_list,
                           b_T_list, action_tensor_center,
                           return_full_info=False):

  #layers_n: number of hidden units each layer
  #W_T_list, b_T_list: multiplicatie and bias weights for each layer
  #X: raw input, y: one-hot encoding of labels

  # List of bounds (l_i,u_i) for i = 1,...,K-1
  l_list = [
      action_tensor_center - action_max * tf.ones_like(action_tensor_center)
  ]
  u_list = [
      action_tensor_center + action_max * tf.ones_like(action_tensor_center)
  ]

  # List of transition matrices D_i for i = 1,...,K-1
  D_list = [tf.zeros_like(action_tensor_center)]

  # Indicators of spanning ReLu neurons for i = 1,...,K-1
  I_list = [tf.zeros_like(action_tensor_center)]

  # Indicators of active ReLu neurons for i = 1,...,K-1
  Ip_list = [tf.zeros_like(action_tensor_center)]

  # Final list of duals nu_i for i = 1,...,K-1
  Nu_list = [
      tf.zeros([batch_size, W_T_list[0].get_shape().as_list()[1], 1])
      for i in range(num_layers - 1)
  ]

  # Initialize Nu_K
  Nu_K = -tf.expand_dims(-tf.eye(1), axis=-1)

  # Final list of b_i'*nu_{i+1} for i = 1,...,K-1
  gamma_list = [b_T_list[i] for i in range(num_layers - 1)]

  ################## get bounds for layers i = 2,...,K-1
  for i in range(2, num_layers):
    pre_l_i = l_list[-1]
    pre_u_i = u_list[-1]

    mu_i = 0.5 * (pre_l_i + pre_u_i)
    r_i = 0.5 * (pre_u_i - pre_l_i)

    l_i = tf.matmul(mu_i, W_T_list[i - 2]) - tf.matmul(
        r_i, tf.abs(W_T_list[i - 2])) + b_T_list[i - 2]

    u_i = tf.matmul(mu_i, W_T_list[i - 2]) + tf.matmul(
        r_i, tf.abs(W_T_list[i - 2])) + b_T_list[i - 2]

    l_list.append(l_i)
    u_list.append(u_i)

    # form Ip, I
    Ip_i, I_i = dual_method.get_I(l_list[-1], u_list[-1])
    I_list.append(I_i)
    Ip_list.append(Ip_i)

    # form D
    D_i = dual_method.get_D(l_list[-1], u_list[-1], Ip_i, I_i)
    D_list.append(D_i)

  ############## Go backward and form Nu_i

  # initialize Nu_{K-1} & gamma_{K-1}
  Nu_list[-1] = tf.einsum('ij,jk->ijk', D_list[-1], W_T_list[-1])
  Nu_K = tf.tile(Nu_K, [Nu_list[-1].get_shape().as_list()[0], 1, 1])
  Nu_list[-1] = tf.einsum('ijk,ikm->ijm', Nu_list[-1], Nu_K)

  gamma_list[-1] = tf.einsum('ij,ijm->im', gamma_list[-1], Nu_K)

  # initialize lv_sum
  lv_sum = tf.einsum('ij,ijm->im', l_list[-1] * I_list[-1],
                     tf.nn.relu(Nu_list[-1]))

  # update Nu_j for layers j = K-2,...,2
  # and gamma_j for layers j = K-2,...,2
  for j in range(num_layers - 2, 1, -1):
    Nu_hat_j = tf.einsum('jk,ikm->ijm', W_T_list[j - 1], Nu_list[j])

    gamma_list[j - 1] = tf.einsum('ij,ijm->im', b_T_list[j - 1], Nu_list[j])

    Nu_list[j - 1] = tf.einsum('ij,ijk->ijk', D_list[j - 1], Nu_hat_j)

    lv_sum = tf.add(
        lv_sum,
        tf.einsum('ij,ijm->im', l_list[j - 1] * I_list[j - 1],
                  tf.nn.relu(Nu_list[j - 1])))

  # update nu_hat_1 and gamma_1
  Nu_hat_1 = tf.einsum('jk,ikm->ijm', W_T_list[0], Nu_list[1])

  gamma_list[0] = tf.einsum('ij,ijm->im', b_T_list[0], Nu_list[1])

  # Compute J_tilde
  psi = tf.einsum('ij,ijm->im', action_tensor_center,
                  Nu_hat_1) + tf.add_n(gamma_list)

  Nu_hat_1_norm = tf.norm(Nu_hat_1, 1, axis=1, keepdims=False)

  J_tilde = -psi - action_max * Nu_hat_1_norm + lv_sum

  if return_full_info:
    return (-J_tilde, l_list, u_list, D_list, Nu_list, lv_sum, gamma_list, psi,
            Nu_hat_1)
  else:
    return -J_tilde
