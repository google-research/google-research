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

"""Dual functions."""
import tensorflow.compat.v1 as tf


def get_I(l, u):
  # l,u are None, n_layer tensors
  # Ip: active relu units
  # I: unstable relu units

  Ip = tf.where(
      tf.logical_and(tf.greater_equal(l, 0.), tf.greater(u, 0.)),
      tf.ones_like(u), tf.zeros_like(u))
  I = tf.where(
      tf.logical_and(tf.greater(u, 0.), tf.less(l, 0.)), tf.ones_like(u),
      tf.zeros_like(u))

  return Ip, I


def get_D(l, u, Ip, I):

  # D matrix for each layer
  D = Ip + tf.where(tf.greater(I, 0.5), tf.divide(u, u - l), tf.zeros_like(I))

  return D


def create_dual_approx(num_layers, batch_size, action_max, W_T_list, b_T_list,
                       action_tensor_center, return_full_info=False):

  #layers_n: number of hidden units each layer
  #W_T_list, b_T_list: multiplicatie and bias weights for each layer
  #action_tensor_center: raw input, y: one-hot encoding of labels

  # List of bounds (l_i,u_i) for i = 2,...,K-1
  l_list = [tf.zeros_like(action_tensor_center)]
  u_list = [tf.zeros_like(action_tensor_center)]

  # List of transition matrices D_i for i = 2,...,K-1
  D_list = [tf.zeros_like(action_tensor_center)]

  # Indicators of spanning ReLu neurons for i = 2,...,K-1
  I_list = [tf.zeros_like(action_tensor_center)]

  # Indicators of active ReLu neurons for i = 2,...,K-1
  Ip_list = [tf.zeros_like(action_tensor_center)]

  # Final list of duals nu_i for i = 2,...,K-1
  Nu_list = [tf.zeros([batch_size, W_T_list[0].get_shape().as_list()[1], 1])]

  # Initialize Nu_K
  Nu_K = -tf.expand_dims(-tf.eye(1), axis=-1)

  # Final list of b_i'*nu_{i+1} for i = 1,...,K-1
  gamma_list = [b_T_list[0]]

  # Pre-compute bounds for layer 2
  # Initialize Nu_hat_1
  Nu_hat_1 = tf.tile(tf.expand_dims(W_T_list[0], axis=0), [batch_size, 1, 1])

  # Initialize bounds
  l_2 = tf.matmul(action_tensor_center,
                  W_T_list[0]) + gamma_list[0] - action_max * tf.norm(
                      Nu_hat_1, 1, axis=1, keepdims=False)
  u_2 = tf.matmul(action_tensor_center,
                  W_T_list[0]) + gamma_list[0] + action_max * tf.norm(
                      Nu_hat_1, 1, axis=1, keepdims=False)

  # Add to list (store in vector format)
  l_list.append(l_2)
  u_list.append(u_2)

  # Recursion

  for i in range(2, num_layers):
    # form Ip, I
    Ip_i, I_i = get_I(l_list[i - 1], u_list[i - 1])
    I_list.append(I_i)
    Ip_list.append(Ip_i)

    # form D
    D_i = get_D(l_list[i - 1], u_list[i - 1], Ip_i, I_i)
    D_list.append(D_i)

    # initialize nu_i
    Nu_list.append(tf.einsum('ij,jk->ijk', D_i, W_T_list[i - 1]))

    # initialize gamma_i
    gamma_list.append(b_T_list[i - 1])

    # if final iteration, update with Nu_K
    if i == num_layers - 1:
      Nu_K = tf.tile(Nu_K, [Nu_list[i - 1].get_shape().as_list()[0], 1, 1])
      Nu_list[i - 1] = tf.einsum('ijk,ikm->ijm', Nu_list[i - 1], Nu_K)
      gamma_list[i - 1] = tf.einsum('ij,ijm->im', gamma_list[i - 1], Nu_K)

    # initialize next layer bounds
    l_ip1 = tf.einsum('ij,ijm->im', l_list[i - 1] * I_list[i - 1],
                      tf.nn.relu(-Nu_list[i - 1]))
    u_ip1 = -tf.einsum('ij,ijm->im', l_list[i - 1] * I_list[i - 1],
                       tf.nn.relu(Nu_list[i - 1]))

    # update nu for layers i-1,...,2
    for j in range(i - 1, 1, -1):
      Nu_hat_j = tf.einsum('jk,ikm->ijm', W_T_list[j - 1], Nu_list[j])

      Nu_list[j - 1] = tf.einsum('ij,ijk->ijk', D_list[j - 1], Nu_hat_j)

      l_ip1 = tf.add(
          l_ip1,
          tf.einsum('ij,ijm->im', l_list[j - 1] * I_list[j - 1],
                    tf.nn.relu(-Nu_list[j - 1])))
      u_ip1 = tf.subtract(
          u_ip1,
          tf.einsum('ij,ijm->im', l_list[j - 1] * I_list[j - 1],
                    tf.nn.relu(Nu_list[j - 1])))

    # update nu_hat_1
    Nu_hat_1 = tf.einsum('jk,ikm->ijm', W_T_list[0], Nu_list[1])

    # start sum
    psi = tf.einsum('ij,ijm->im', action_tensor_center,
                    Nu_hat_1) + gamma_list[i - 1]

    # update gamma for layers 1,...,i-1
    for j in range(1, i):
      gamma_list[j - 1] = tf.einsum('ij,ijm->im', b_T_list[j - 1], Nu_list[j])

      psi = tf.add(psi, gamma_list[j - 1])

    Nu_hat_1_norm = tf.norm(Nu_hat_1, 1, axis=1, keepdims=False)

    if i < num_layers - 1:
      # finalize bounds
      l_ip1 = tf.add(l_ip1, psi - action_max * Nu_hat_1_norm)
      u_ip1 = tf.add(u_ip1, psi + action_max * Nu_hat_1_norm)

      # add to list
      l_list.append(l_ip1)
      u_list.append(u_ip1)

    else:
      # compute J_tilde
      J_tilde = -psi - action_max * Nu_hat_1_norm - u_ip1

  if return_full_info:
    return (-J_tilde, l_list, u_list, D_list, Nu_list, gamma_list, psi, l_ip1,
            u_ip1, Nu_hat_1)
  else:
    return -J_tilde
