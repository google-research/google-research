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

# Lint as: python2, python3
"""Interleaving Transition Kernel."""
# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import warnings

from six.moves import range
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.mcmc import TransitionKernel


__all__ = [
    'Interleaved',
]


# Cause all warnings to always be triggered.
# Not having this means subsequent calls wont trigger the warning.
warnings.simplefilter('always')

InterleavedKernelResults = collections.namedtuple(
    'InterleavedKernelResults',
    [
        'accepted_results',
        'is_accepted',
        'log_accept_ratios',
        'extras'
    ])


def noop(state):
  return state


def make_name(super_name, default_super_name, sub_name):
  """Helper which makes a `str` name; useful for tf.name_scope."""
  name = super_name if super_name is not None else default_super_name
  if sub_name is not None:
    name += '_' + sub_name
  return name


class Interleaved(TransitionKernel):

  def __init__(self, inner_kernel_cp, inner_kernel_ncp,
               to_cp=noop, to_ncp=noop, seed=None, name=None):

    self._seed_stream = tfp.util.SeedStream(
        seed, 'interleaved_one_step')

    if (inner_kernel_cp.seed == inner_kernel_ncp.seed and
        inner_kernel_cp.seed is not None):
      raise Exception(
          'The two interleaved kernels cannot have the same random seed.')

    self._parameters = dict(
        inner_kernels={'cp': inner_kernel_cp, 'ncp': inner_kernel_ncp},
        to_cp=to_cp,
        to_ncp=to_ncp,
        seed=seed,
        name=name)

  @property
  def to_cp(self):
    return self._parameters['to_cp']

  @property
  def to_ncp(self):
    return self._parameters['to_ncp']

  @property
  def inner_kernel(self):
    return self._parameters['inner_kernels']

  @property
  def seed(self):
    return self._parameters['seed']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return True

  def one_step(self, current_state, previous_kernel_results):

    with tf.name_scope(
        name=make_name(self.name, 'iterleaved', 'one_step'),
        values=[current_state, previous_kernel_results]):

      # Take a step in the CP space
      [
          next_cp_state,
          kernel_res_cp
      ] = self.inner_kernel['cp'].one_step(
          current_state,
          previous_kernel_results.accepted_results)

      current_ncp_state = self.to_ncp(next_cp_state)

      for i in range(len(current_ncp_state)):
        current_ncp_state[i] = tf.identity(current_ncp_state[i])

      previous_kernel_results_ncp = self.bootstrap_results(
          current_ncp_state,
          inner_name='ncp',
          is_accepted=kernel_res_cp.is_accepted,
          log_accept_ratios={
              'cp': kernel_res_cp.log_accept_ratio,
              'ncp': previous_kernel_results.log_accept_ratios['ncp']},
          extras={'cp': kernel_res_cp.extra,
                  'ncp': previous_kernel_results.extras['ncp']})

      # Take a step in the NCP space
      [
          next_ncp_state,
          kernel_res_ncp,
      ] = self.inner_kernel['ncp'].one_step(
          current_ncp_state,
          previous_kernel_results_ncp.accepted_results)

      next_state = self.to_cp(next_ncp_state)

      for i in range(len(next_state)):
        next_state[i] = tf.identity(next_state[i])

      kernel_results = self.bootstrap_results(
          next_state,
          inner_name='cp',
          is_accepted=kernel_res_ncp.is_accepted,
          log_accept_ratios={'cp': kernel_res_cp.log_accept_ratio,
                             'ncp': kernel_res_ncp.log_accept_ratio},
          extras={'cp': kernel_res_cp.extra, 'ncp': kernel_res_ncp.extra})
      return next_state, kernel_results

  def bootstrap_results(self,
                        init_state,
                        inner_name='INIT',
                        is_accepted=None,
                        log_accept_ratios=None,
                        extras=None):
    """Returns an object with the same type as returned by `one_step`."""
    with tf.name_scope(
        name=make_name(self.name, 'interleaved', 'bootstrap_results'),
        values=[init_state]):

      if inner_name == 'INIT':
        names = ['cp', 'ncp']
      else:
        names = [inner_name]

      for kernel_name in names:
        extras = extras if extras is not None else {'cp': [], 'ncp': []}

        pkr = self.inner_kernel[kernel_name].bootstrap_results(init_state)
        if extras is not None and extras[kernel_name]:
          pkr = pkr._replace(extra=extras[kernel_name])
        if log_accept_ratios is not None and kernel_name in log_accept_ratios:
          pkr = pkr._replace(log_accept_ratio=log_accept_ratios[kernel_name])
        extras[kernel_name] = pkr.extra

      return InterleavedKernelResults(
          accepted_results=pkr,
          is_accepted=True if is_accepted is None else is_accepted,
          log_accept_ratios=({
              'cp': tf.zeros_like(len(pkr.proposed_state), dtype=tf.float32),
              'ncp': tf.zeros_like(len(pkr.proposed_state), dtype=tf.float32)}
                             if log_accept_ratios is None
                             else log_accept_ratios),
          extras=extras
      )
