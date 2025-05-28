# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Semiring."""
import abc
from typing import Generic, TypeVar

from lingvo import compat as tf
import tensorflow_probability as tfp

import utils

T = TypeVar('T')
TensorTuple = utils.TensorTuple
LogTensor = tuple[tf.Tensor]
DualTensor = tuple[tf.Tensor, tf.Tensor]
LogReverseKLTensor = tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]


class Semiring(abc.ABC, Generic[T]):
  """Abstract base class for a semiring.

  A monoid is a set equipped with a binary associative operation and an identity
  element.

  A semiring is a set equipped with addition and multiplication such that:
  1) Addition (+) is a commutative monoid with identity (0).
  2) Multiplication (*) is a monoid with identity (1).
  3) Multiplication distributes over addition from both sides.
  4) The additive identity (0) is an annihilator for multiplication (*), i.e.
  multiplying any element with (0) results in (0).

  Concrete subclasses of Semiring need to implement seven different methods:
  1) additive_identity
  2) add
  3) add_list
  4) multiplicative_identity
  5) multiply
  6) multiply_list
  7) convert_logits

  add and multiply are binary operations as are in the definition of a semiring.
  However, add_list and multiply_list are needed as well because there are often
  more efficient ways to implement addition and multiplication than to do it
  iteratively. convert_logits converts the given logits into the input values
  expected by the semiring.
  """

  @abc.abstractmethod
  def additive_identity(self,
                        shape = (1,),
                        dtype = tf.float32):
    """Returns additive identity of the specified shape and datatype."""

  @abc.abstractmethod
  def add(self, elem_1, elem_2):
    """Adds two elements."""

  @abc.abstractmethod
  def add_list(self, elems_list):
    """Adds a list of elements."""

  @abc.abstractmethod
  def multiplicative_identity(self,
                              shape = (1,),
                              dtype = tf.float32):
    """Returns multiplicative identity of the specified shape and datatype."""

  @abc.abstractmethod
  def multiply(self, elem_1, elem_2):
    """Multiplies two elements."""

  @abc.abstractmethod
  def multiply_list(self, elems_list):
    """Multiplies a list of elements."""

  @abc.abstractmethod
  def convert_logits(self, elem):
    """Converts logits into semiring inputs."""


class LogSemiring(Semiring[LogTensor]):
  """Log semiring.

  Each element is of the form log(p) where p is a real number from [0, 1].

  Additive identity:
  (0) = neg inf.

  Addition:
  a (+) b = LogSumExp(a, b).

  Multiplicative identity:
  (1) = 0.

  Multiplication:
  a (*) b = a + b.

  Convert logits:
  log(p) -> log(p).

  Note that multiplication is implemented in a numerically stable manner.
  """
  _LOGP = 0  # First argument in LogTensor

  def additive_identity(self,
                        shape = (1,),
                        dtype = tf.float32):
    del self
    return (utils.logzero(shape=shape, dtype=dtype),)

  def add(self, elem_1, elem_2):
    del self
    return tuple(map(utils.logsumexp_list, zip(elem_1, elem_2)))

  def add_list(self, elems_list):
    del self
    return tuple(map(utils.logsumexp_list, zip(*elems_list)))

  def multiplicative_identity(self,
                              shape = (1,),
                              dtype = tf.float32):
    del self
    return (tf.zeros(shape=shape, dtype=dtype),)

  def multiply(self, elem_1, elem_2):
    return (utils.safe_result(elem_1[self._LOGP] + elem_2[self._LOGP]),)

  def multiply_list(self, elems_list):
    elems_list = [e[self._LOGP] for e in elems_list]
    return (utils.safe_result(tf.add_n(elems_list)),)

  def convert_logits(self, elem):
    del self
    return elem


class LogEntropySemiring(Semiring[DualTensor]):
  """Log Entropy semiring.

  Each element is of the form <log(p), log(-plog(q))> where p and q are real
  numbers from [0, 1]. Addition and multiplication follow the dual number system
  but with a log morphism applied on both arguments.
  https://en.wikipedia.org/wiki/Dual_number

  Additive identity:
  (0) = <neg inf, neg inf>.

  Addition:
  <a, b> (+) <c, d> = <LogSumExp(a, c), LogSumExp(b, d)>.

  Multiplicative identity:
  (1) = <0, neg inf>.

  Multiplication:
  <a, b> (*) <c, d> = <a + c, LogSumExp(a + d, b + c)>.

  Convert logits:
  <log(p), log(q)> -> <log(p), log(-plog(q))>.

  Note that multiplication is implemented in a numerically stable manner.
  """
  _LOGP = 0  # First argument in DualTensor
  _LOGMINUSPLOGQ = 1  # Second argument in DualTensor

  def additive_identity(self,
                        shape = (1,),
                        dtype = tf.float32):
    del self
    neg_inf = utils.logzero(shape=shape, dtype=dtype)
    return (neg_inf, neg_inf)

  def add(self, elem_1, elem_2):
    del self
    return tuple(map(utils.logsumexp_list, zip(elem_1, elem_2)))

  def add_list(self, elems_list):
    del self
    return tuple(map(utils.logsumexp_list, zip(*elems_list)))

  def multiplicative_identity(
      self,
      shape = (1,),
      dtype = tf.float32):
    del self
    zero = tf.zeros(shape=shape, dtype=dtype)
    neg_inf = utils.logzero(shape=shape, dtype=dtype)
    return (zero, neg_inf)

  def multiply(self, elem_1, elem_2):
    logp = utils.safe_result(elem_1[self._LOGP] + elem_2[self._LOGP])
    logminusplogq = utils.logcrossmultiply(elem_1[self._LOGP],
                                           elem_1[self._LOGMINUSPLOGQ],
                                           elem_2[self._LOGP],
                                           elem_2[self._LOGMINUSPLOGQ])
    return (logp, logminusplogq)

  def multiply_list(self, elems_list):
    # Compute the result iteratively.
    elems = tuple(map(tf.stack, zip(*elems_list)))
    elems = tfp.math.scan_associative(self.multiply, elems)
    return tuple(e[-1] for e in elems)

  def convert_logits(self, elem):
    del self
    logp, logq = elem
    logminusplogq = utils.logminus(logp, logq)
    return (logp, logminusplogq)


class LogReverseKLSemiring(Semiring[LogReverseKLTensor]):
  """Log Reverse-KL semiring.

  Each element is of the form <log(p), log(q), log(-qlog(q)), log(-qlog(p))>
  where p and q are real numbers from [0, 1].

  Additive identity:
  (0) = <neg inf, neg inf, neg inf, neg inf>.

  Addition:
  <a, b, c, d> (+) <e, f, g, h> = <LogSumExp(a, e), LogSumExp(b, f),
                                   LogSumExp(c, g), LogSumExp(d, h)>.

  Multiplicative identity:
  (1) = <0, 0, neg inf, neg inf>.

  Multiplication:
  <a, b, c, d> (*) <e, f, g, h> = <a + e, b + f, LogSumExp(b + g, c + f),
                                   LogSumExp(b + h, d + f)>.

  Convert logits:
  <log(p), log(q)> -> <log(p), log(q), log(-qlog(q)), log(-qlog(p))>.

  Note that multiplication is implemented in a numerically stable manner.
  """
  _LOGP = 0  # First argument in LogReverseKLTensor
  _LOGQ = 1  # Second argument in LogReverseKLTensor
  _LOGMINUSQLOGQ = 2  # Third argument in LogReverseKLTensor
  _LOGMINUSQLOGP = 3  # Fourth argument in LogReverseKLTensor

  def additive_identity(
      self,
      shape = (1,),
      dtype = tf.float32):
    del self
    neg_inf = utils.logzero(shape=shape, dtype=dtype)
    return (neg_inf, neg_inf, neg_inf, neg_inf)

  def add(self, elem_1,
          elem_2):
    del self
    return tuple(map(utils.logsumexp_list, zip(elem_1, elem_2)))

  def add_list(self,
               elems_list):
    del self
    return tuple(map(utils.logsumexp_list, zip(*elems_list)))

  def multiplicative_identity(
      self,
      shape = (1,),
      dtype = tf.float32):
    del self
    zero = tf.zeros(shape=shape, dtype=dtype)
    neg_inf = utils.logzero(shape=shape, dtype=dtype)
    return (zero, zero, neg_inf, neg_inf)

  def multiply(self, elem_1,
               elem_2):
    logp = utils.safe_result(elem_1[self._LOGP] + elem_2[self._LOGP])
    logq = utils.safe_result(elem_1[self._LOGQ] + elem_2[self._LOGQ])
    logminusqlogq = utils.logcrossmultiply(elem_1[self._LOGQ],
                                           elem_1[self._LOGMINUSQLOGQ],
                                           elem_2[self._LOGQ],
                                           elem_2[self._LOGMINUSQLOGQ])
    logminusqlogp = utils.logcrossmultiply(elem_1[self._LOGQ],
                                           elem_1[self._LOGMINUSQLOGP],
                                           elem_2[self._LOGQ],
                                           elem_2[self._LOGMINUSQLOGP])
    return (logp, logq, logminusqlogq, logminusqlogp)

  def multiply_list(self,
                    elems_list):
    # Compute the result iteratively.
    elems = tuple(map(tf.stack, zip(*elems_list)))
    elems = tfp.math.scan_associative(self.multiply, elems)
    return tuple(e[-1] for e in elems)

  def convert_logits(self, elem):
    del self
    logp, logq = elem
    logminusqlogq = utils.logminus(logq, logq)
    logminusqlogp = utils.logminus(logq, logp)
    return (logp, logq, logminusqlogq, logminusqlogp)
