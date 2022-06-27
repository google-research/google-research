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

"""Hyperparameter utilities, containing functions to convert from the

unconstrained to the constrained parameterizations and vice versa, as well
as specifying the valid ranges of each hyperparameter.
"""
import pdb
from functools import partial

import jax
import jax.numpy as jnp

meta_opt_init_range = {
    'lr': (1e-4, 1e-2),
    'mom': (0.01, 0.9),
    'b1': (0.1, 0.9),
    'b2': (0.01, 0.999),
    'eps': (1e-9, 10),
    'wd': (1e-9, 10),
    'mask': (0.2, 0.8),
}

default_value = {
    'lr': 0.1,
    'mom': 0.9,
    'b1': 0.9,
    'b2': 0.99,
    'eps': 1e-8,
    'wd': 1e-9,
    'mask': 0.5
}

cons_range = {
    'lr': (1e-8, 10),
    'mom': (0.01, 0.999),
    'b1': (0.01, 0.99),
    'b2': (0.01, 0.9999),
    'eps': (1e-9, 10),
    'wd': (1e-9, 10),
    'mask': (0.01, 0.999),
}

uncons_funcs = {
    'lr': jnp.log,
    'mom': jax.scipy.special.logit,
    'b1': jax.scipy.special.logit,
    'b2': jax.scipy.special.logit,
    'eps': jnp.log,
    'wd': jnp.log,
    'mask': jax.scipy.special.logit,
}

cons_funcs = {
    'lr':
        jnp.exp,
    'mom':
        jax.nn.sigmoid,
    'b1':
        jax.nn.sigmoid,
    'b2':
        jax.nn.sigmoid,
    'eps':
        lambda x: jnp.clip(
            jnp.exp(x), a_min=cons_range['eps'][0], a_max=cons_range['eps'][1]),
    'wd':
        lambda x: jnp.clip(
            jnp.exp(x), a_min=cons_range['wd'][0], a_max=cons_range['wd'][1]),
    'mask':
        jax.nn.sigmoid,
}

hparam_range_dict = {}
for hparam_name in cons_range:
  hparam_range_dict[hparam_name] = (uncons_funcs[hparam_name](
      cons_range[hparam_name][0]), uncons_funcs[hparam_name](
          cons_range[hparam_name][1]))

meta_opt_init_range_dict = {}
for hparam_name in meta_opt_init_range:
  meta_opt_init_range_dict[hparam_name] = (uncons_funcs[hparam_name](
      meta_opt_init_range[hparam_name][0]), uncons_funcs[hparam_name](
          meta_opt_init_range[hparam_name][1]))

if __name__ == '__main__':
  for hparam_name in cons_range:
    print(hparam_name)
    print('\tCons range: ({:6.4e}, {:6.4e})'.format(cons_range[hparam_name][0],
                                                    cons_range[hparam_name][1]))
    print('\tUncons range: ({:6.4e}, {:6.4e})'.format(
        hparam_range_dict[hparam_name][0], hparam_range_dict[hparam_name][1]))
    print('\tCons of hparam range: ({:6.4e}, {:6.4e})'.format(
        cons_funcs[hparam_name](hparam_range_dict[hparam_name][0]),
        cons_funcs[hparam_name](hparam_range_dict[hparam_name][1])))

abbreviation_dict = {
    'm': 'mom',
    'lr': 'lr',
    'b1': 'b1',
    'b2': 'b2',
    'eps': 'eps',
    'wd': 'wd',
    'f-pl': 'fixed-pl',
    'f': 'fixed',
    'l': 'linear',
    'l-pl': 'linear-pl',
    'itd': 'inverse-time-decay',
    'itd-pl': 'inverse-time-decay-pl',
    'mom': 'mom',
    'momentum': 'mom',
    'weight_decay': 'wd',
    'fixed-pl': 'fixed-pl',
    'fixed': 'fixed',
    'linear': 'linear',
    'linear-pl': 'linear-pl',
    'inverse-time-decay': 'inverse-time-decay',
    'inverse-time-decay-pl': 'inverse-time-decay-pl',
    'mask': 'mask',
}
