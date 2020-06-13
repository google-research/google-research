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

"""Implementation of a VariablePool.

VariablePool interface allows centrlized variable creation and maintance.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc

import numpy as np
import six
import tensorflow.compat.v1 as tf
from typing import Any, Text, List

from structured_multihashing.smh import virtual_variable

SUPPORTED_DTYPES = [tf.bfloat16, tf.float16, tf.float32, tf.float64]

VARIABLE_POOL_TYPE = tf.float32


@six.add_metaclass(abc.ABCMeta)
class VariablePool(object):
  """A class that handles a pool of virtual-variables.

  Allocates slices of the pool in an non-overlapping way.
  """

  @property
  def core_variables(self):
    """A list of variables created by the VariablePool."""
    return self._core_variables

  @property
  def scope_name(self):
    """The name of the scope core variables are created in."""
    return self._scope_name

  @property
  def core_size(self):
    """The number of variables used by the pool."""
    return sum(var.shape.num_elements() for var in self.core_variables)

  @abc.abstractproperty
  def status(self):
    """Total number indices allocated."""
    pass

  @abc.abstractproperty
  def pool_size(self):
    """Total number of elements supported by the VariablePool."""
    pass

  @abc.abstractmethod
  def get_slice(self, shape):
    """Allocates variables from the variable pool with given shape.

    Args:
      shape: A list of integers representing the shape of the slice. Same as
        tf.get_variable().

    Returns:
      A tf.Tensor with appropriate shape.
    """
    pass


class ProductVariablePool(VariablePool):
  """A VariablePool were the pool is a matrix products."""

  def __init__(self,
               trainable,
               pool_size,
               fraction,
               stddev = 1.0,
               initializer = tf.random_normal_initializer,
               use_kronecker_product = False,
               index_store_type = virtual_variable.IndexStoreType.basic):
    """Creates an instance of `ProductVariablePool`.

    Args:
      trainable: boolean, indicate whether the created variables are trainable
        or not.
      pool_size: int, total number of virtual variables requried. The acutal
        number of virtual variables created can be larger than the number
        specified by this argument.
      fraction: float, the fraction of `pool_size` of variables to create.
      stddev: float, standard deviation for the variable pool. Default value is
        1.0.
      initializer: A tf.initializer e.g. 'truncated_normal_initializer' or
        'random_normal_initializer'. Default value is
        tf.random_normal_initializer.
      use_kronecker_product: Indicate product should be a kronecker product or a
        matrix prodcut.
      index_store_type: IndexStoreType, key of SUPPORTED_INDEX_STORES.
    """
    if fraction <= 0 or fraction > 1.0:
      raise ValueError('fraction %f must be >0 and <=1.0' % fraction)

    self._scope_name = 'ProductVariablePool'
    if use_kronecker_product:
      variable_generator = _create_kronecker_variables
    else:
      variable_generator = _create_matmul_variables

    with tf.variable_scope(self._scope_name):
      variables, size, pool = variable_generator(pool_size, fraction,
                                                 initializer, stddev, trainable)

    self._core_variables = variables
    self._virtual_variables = tf.reshape(pool, [size], name='weight_pool')

    index_store_cls = virtual_variable.get_index_store(index_store_type)
    self._index_store = index_store_cls(size)

  @property
  def status(self):
    return self._index_store.current()

  @property
  def pool_size(self):
    return self._index_store.size

  def get_slice(self, shape):
    """Allocates variables from the variable pool with given shape."""
    tensor_num_elements = int(np.prod(shape))
    sliced_virtual_variables = self._index_store.allocate_variables_from_pool(
        self._virtual_variables, tensor_num_elements)
    weight_tensor = tf.reshape(sliced_virtual_variables, shape)

    return tf.identity(weight_tensor, 'product_slice%d' % self.status)


HASH_POOL_SEED = 412013


class HashVariablePool(VariablePool):
  """A VariablePool were mapping into pool is hashed."""

  def __init__(self,
               trainable,
               stddev,
               pool_size,
               fraction,
               initializer,
               seed = HASH_POOL_SEED,
               index_store_type = virtual_variable.IndexStoreType.basic):
    """Creates an instance of `HashVariablePool`.

    Args:
      trainable: boolean, indicate whether the created variables are trainable
        or not.
      stddev: float, standard deviation for the variable pool.
      pool_size: int, total number of virtual variables requried. The acutal
        number of virtual variables created can be larger than the number
        specified by this argument.
      fraction: float, the fraction of `pool_size` of variables to create.
      initializer: A tf.initializer e.g. 'truncated_normal_initializer' or
        'random_normal'.
      seed: Integer, seed for the random hashing.
      index_store_type: String, key of SUPPORTED_INDEX_STORES. 'padding' is not
        supported by HashVariablePool yet.
    """
    del seed  # unused
    if fraction <= 0 or fraction > 1.0:
      raise ValueError('fraction %f must be >0 and <=1.0' % fraction)
    self._scope_name = 'HashVariablePool'
    self._hash_indices = None

    hash_size = int(np.floor(fraction * pool_size))
    if not hash_size:
      raise ValueError(
          'fraction %f too low, results in 0 size hash for pool size %d.' %
          (fraction, pool_size))

    index_store_cls = virtual_variable.get_index_store(index_store_type)
    self._index_store = index_store_cls(pool_size)
    if self._index_store.type == virtual_variable.IndexStoreType.padding:
      raise ValueError('HashVariablePool does not support PaddingIndexStore '
                       'yet.')
    replicas = int(np.ceil(float(pool_size + 1) / hash_size))

    # The following is for python2/3 compatibility. As range(k) does not return
    # a list in python 3.
    base_index_list = range(hash_size)
    if not isinstance(base_index_list, list):
      base_index_list = list(base_index_list)

    indices = np.array(base_index_list * replicas)

    # len(indices) = hash_size * replicas
    #             >= hash_size * (pool_size + 1) / hash_size
    #             ~ pool_size
    assert len(indices) >= pool_size
    indices = indices[:pool_size]

    # Preserving the state is done in order to not mess up with other elements
    # that might depend on numpy seed for some reason.
    # debuggin:
    # np_state = np.random.get_state()
    # np.random.seed(seed=seed)
    # random_indices = np.random.permutation(len(indices))
    # tf.logging.info('First 4 indices = %d %d', random_indices[:4], seed)
    # self._set_hash_indices(random_indices)
    # np.random.set_state(np_state)
    self._set_hash_indices(indices)

    with tf.variable_scope(self._scope_name):
      self._hash = tf.get_variable(
          'hash', [int(hash_size)],
          trainable=trainable,
          initializer=initializer(stddev=stddev))
      self._core_variables = [self._hash]

  @property
  def status(self):
    return self._index_store.current()

  @property
  def pool_size(self):
    return self._index_store.size

  def _set_hash_indices(self, indices):
    if self._hash_indices is not None and len(
        self._hash_indices) != len(indices):
      raise ValueError('Trying to set wrong length of indices %d' %
                       len(indices))
    self._hash_indices = indices

  def get_slice(self, shape):
    """Allocates variables from self._virtual_variables."""
    tensor_num_elements = int(np.prod(shape))
    sliced_hash_indices = self._index_store.allocate_variables_from_pool(
        tf.convert_to_tensor(self._hash_indices), tensor_num_elements)
    sliced_virtual_variables = tf.gather(
        self._hash, indices=sliced_hash_indices)
    weight_tensor = tf.reshape(sliced_virtual_variables, shape)
    return tf.identity(weight_tensor, 'hash_slice_%d' % self.status)


VARIABLE_POOLS_NAMES = {
    'PRODUCT_POOL': ProductVariablePool,
    'HASH_POOL': HashVariablePool
}


def has_seed_arg(class_name):
  return 'HASH_POOL' == class_name


class MetaVariablePool(VariablePool):
  """MetaVariablePool returns a specified of other variable pools."""

  def __init__(self,
               trainable,
               stddev,
               pool_size,
               fraction,
               initializer,
               elements,
               reduce_by = 'SUM',
               index_store_type = virtual_variable.IndexStoreType.basic):
    """Creates an instance of `MetaVariablePool`.

    Args:
      trainable: boolean, indicate whether the created variables are trainable
        or not.
      stddev: float, standard deviation for the variable pool.
      pool_size: int, total number of virtual variables requried. The acutal
        number of virtual variables created can be larger than the number
        specified by this argument.
      fraction: float, the fraction of `pool_size` of variables to create.
      initializer: A tf.initializer e.g. 'truncated_normal_initializer' or
        'random_normal'.
      elements: Names of VariablePools to build, must be keys of
        VARIABLE_POOLS_NAMES.
      reduce_by: SUM, PRODUCT.
      index_store_type: String, key of SUPPORTED_INDEX_STORES.
    """
    if fraction <= 0 or fraction > 1.0:
      raise ValueError('fraction %f must be >0 and <=1.0' % fraction)
    self._scope_name = 'MetaVariablePool'
    self._sub_pools = []
    self._core_variables = []
    if reduce_by == 'SUM':
      self._reduce = tf.add_n
    elif reduce_by == 'PROD':
      self._reduce = _prod_n
    else:
      raise ValueError('unsupported reduce_by %s' % reduce_by)
    num_elements = len(elements)
    element_stddev = stddev / num_elements
    element_fraction = fraction / num_elements

    with tf.variable_scope(self._scope_name):
      for i, class_name in enumerate(elements):
        if class_name not in VARIABLE_POOLS_NAMES:
          raise ValueError(
              'Unrecognized element %s, supported elements are %s' %
              (class_name, VARIABLE_POOLS_NAMES.keys()))
        kwargs = {
            'trainable': trainable,
            'stddev': element_stddev,
            'pool_size': pool_size,
            'fraction': element_fraction,
            'initializer': initializer,
            'index_store_type': index_store_type
        }
        with tf.variable_scope('subpool_%d' % i):
          if has_seed_arg(class_name):
            kwargs['seed'] = HASH_POOL_SEED + i
          pool = VARIABLE_POOLS_NAMES[class_name](**kwargs)  # type: ignore
        self._sub_pools.append(pool)
        self._core_variables.extend(pool.core_variables)

  @property
  def status(self):
    return self._sub_pools[0].status

  @property
  def pool_size(self):
    return self._sub_pools[0].pool_size

  def get_slice(self, shape):
    """Allocates variables from the variable pool with given shape."""
    weight_tensor = self._reduce(
        [pool.get_slice(shape) for pool in self._sub_pools])
    return tf.identity(weight_tensor, 'meta_slice%d' % self.status)


def _compute_correlated_stddev(target_std, n_terms):
  """Computes corrected std for Kronecker Products initialization.

  Computes std of a_i,b_i such that std(sum a_i*b_i) = target_std; assuming
  a_i, b_i ~ i.i.d with mean=0 and var = corrected_stddev^2.

  Details of computation:
   target_std^2 = var(sum(a_i*b_i)) = sum(var(a_i*b_i)) =
     = num_elements*var(a_i)*var(b_i) = num_elements*corrected_stddev^4 ===>
   corrected_stddev = sqrt(target_std/sqrt(num_elements))

  Args:
    target_std: Target std we are looking to achieve
    n_terms: Number of terms (a_i*b_i) in summation.

  Returns:
    Corrected std value.
  """
  corrected_stddev = np.sqrt(target_std / np.sqrt(n_terms))
  return np.float32(corrected_stddev)


def _prod_n(tensors):
  """Returns the product of the elements in the `tensors`.

  Like tf.add_n, but with a product.

  Args:
    tensors: A list of tf.Tensor to multiply.

  Returns:
    The product of elements of tensors
  """
  if not tensors:
    raise ValueError('Empty list of tensors')
  res = tensors[0]
  for tensor in tensors[1:]:
    res *= tensor
  return res


def _create_kronecker_variables(pool_size, fraction, initializer, stddev,
                                trainable):
  """Creates kronecker product variable pool from parameters."""
  # When using a Kronecker product we create a pair of matrices of size:
  # (num_replicas, matrix_dim, matrix_dim) a reduce sum of their product is
  # of size (maxtrix_dim^2, maxtrix_dim^2) with maxtrix_dim^4 elements.
  matrix_dim = int(np.ceil(pool_size**0.25))
  num_elements = matrix_dim * matrix_dim
  num_replicas = int(np.floor(0.5 * fraction * pool_size / num_elements))
  if not num_replicas:
    raise ValueError(
        'fraction %f too low, results in 0 replicas for pool size %d dim %d.' %
        (fraction, pool_size, matrix_dim))

  size = matrix_dim**4
  correlated_stddev = _compute_correlated_stddev(stddev, num_replicas)

  left_matrix = tf.get_variable(
      'variable_left', [num_replicas, matrix_dim, matrix_dim],
      trainable=trainable,
      dtype=VARIABLE_POOL_TYPE,
      initializer=initializer(stddev=correlated_stddev))
  right_matrix = tf.get_variable(
      'variable_right', [num_replicas, matrix_dim, matrix_dim],
      trainable=trainable,
      dtype=VARIABLE_POOL_TYPE,
      initializer=initializer(stddev=correlated_stddev))
  left = tf.linalg.LinearOperatorFullMatrix(left_matrix, is_square=True)
  right = tf.linalg.LinearOperatorFullMatrix(right_matrix, is_square=True)
  pool = tf.reduce_sum(
      tf.linalg.LinearOperatorKronecker([left, right],
                                        is_square=True).to_dense(), 0)
  return [left_matrix, right_matrix], size, pool


def _create_matmul_variables(pool_size, fraction, initializer, stddev,
                             trainable):
  """Creates matrix-multiply variable pool from parameters."""
  # When using a matrix multiply we create a pair of matrices of size:
  # (matrix_dim, num_replicas) and (num_replicas, matrix_dim) their products
  # is of size (maxtrix_dim, maxtrix_dim) with maxtrix_dim^2 elements.
  matrix_dim = int(np.ceil(pool_size**0.5))
  num_replicas = int(np.floor(0.5 * fraction * pool_size / matrix_dim))
  if not num_replicas:
    raise ValueError(
        'fraction %f too low, results in 0 replicas for pool size %d dim %d.' %
        (fraction, pool_size, matrix_dim))

  size = int(matrix_dim**2)
  correlated_stddev = _compute_correlated_stddev(stddev, num_replicas)

  left_matrix = tf.get_variable(
      'variable_left', [matrix_dim, num_replicas],
      trainable=trainable,
      dtype=VARIABLE_POOL_TYPE,
      initializer=initializer(stddev=correlated_stddev))
  right_matrix = tf.get_variable(
      'variable_right', [num_replicas, matrix_dim],
      trainable=trainable,
      dtype=VARIABLE_POOL_TYPE,
      initializer=initializer(stddev=correlated_stddev))
  pool = tf.matmul(left_matrix, right_matrix)
  return [left_matrix, right_matrix], size, pool
