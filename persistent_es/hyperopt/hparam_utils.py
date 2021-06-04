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

import ipdb
from functools import partial

import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnums=3)
def scaled_sigmoid(x, range_min, range_max, do_floor=False):
  scaled_output = jax.nn.sigmoid(x) * (range_max - range_min) + range_min
  if do_floor:
    scaled_output = jnp.floor(scaled_output).astype(jnp.int32)
  return scaled_output

@jax.jit
def scaled_sigmoid_inverse(x, range_min, range_max):
  normalized_value = (x - range_min) / (range_max - range_min)
  return jax.scipy.special.logit(normalized_value)

@jax.jit
def softplus_inverse(x):
  return jnp.log(jnp.exp(x) - 1.)

cons_range = { 'lr': (1e-6, 1.0),
               'b1': (1e-2, 0.999),
               'b2': (1e-2, 0.999),
               'eps': (1e-9, 0.1),
               'momentum': (0.001, 0.99),

               'l1': (1e-7, 1),
               'l2': (1e-7, 1),
               'weight_decay': (1e-9, 1e-1),

               'dropout_prob': (0.01, 0.9),
               'dropouts': (0.01, 0.85),
               'hflip_prob': (0.01, 0.99),
               'vflip_prob': (0.01, 0.99),
               'cutoutsize': (0.1, 20.0),
               'crop_border': (0.1, 18.0),
               'hue_jitter': (0.01, 0.99),
               'sat_jitter': (0.01, 0.99),
             }

uncons_func_dict = { 'lr': jnp.log,
                     'b1': lambda x: scaled_sigmoid_inverse(x, 1e-4, 1.0 - 1e-4),
                     'b2': lambda x: scaled_sigmoid_inverse(x, 1e-4, 1.0 - 1e-4),
                     'eps': jnp.log,
                     'momentum': jax.scipy.special.logit,
                     'weight_decay': jnp.log,
                     'l1': jnp.log,
                     'l2': jnp.log,
                     'cutoutsize': lambda x: scaled_sigmoid_inverse(x, 0.0, 26.0),
                     'dropout_prob': lambda x: scaled_sigmoid_inverse(x, 1e-3, 0.95),
                     'dropouts': lambda x: scaled_sigmoid_inverse(x, 1e-3, 0.95),
                     'hflip_prob': jax.scipy.special.logit,
                     'vflip_prob': jax.scipy.special.logit,
                     'crop_border': lambda x: scaled_sigmoid_inverse(x, 0.0, 20.0),
                     'hue_jitter': jax.scipy.special.logit,
                     'sat_jitter': jax.scipy.special.logit,
                   }

cons_func_dict = { 'lr': lambda x: jnp.exp(x),
                   'b1': lambda x: scaled_sigmoid(x, 1e-4, 1.0 - 1e-4, False),
                   'b2': lambda x: scaled_sigmoid(x, 1e-4, 1.0 - 1e-4, False),
                   'momentum': jax.nn.sigmoid,
                   'eps': lambda x: jnp.clip(jnp.exp(x), a_min=cons_range['eps'][0], a_max=cons_range['eps'][1]),
                   'weight_decay': lambda x: jnp.clip(jnp.exp(x), a_min=cons_range['weight_decay'][0], a_max=cons_range['weight_decay'][1]),
                   'l1': lambda x: jnp.exp(x),
                   'l2': lambda x: jnp.exp(x),
                   'cutoutsize': lambda x: scaled_sigmoid(x, 0.0, 22.0, True),
                   'dropout_prob': lambda x: scaled_sigmoid(x, 1e-2, 0.95, False),
                   'dropouts': lambda x: jnp.clip(scaled_sigmoid(x, 1e-2, 0.95, False), a_min=cons_range['dropouts'][0], a_max=cons_range['dropouts'][1]),
                   'hflip_prob': jax.nn.sigmoid,
                   'vflip_prob': jax.nn.sigmoid,
                   'crop_border': lambda x: scaled_sigmoid(x, 0.0, 20.0, True),
                   'hue_jitter': jax.nn.sigmoid,
                   'sat_jitter': jax.nn.sigmoid,
                 }

meta_opt_init_range = { 'lr': (1e-5, 1e-3),
                        'dropouts': (0.01, 0.85),
                        'l2': (1e-7, 1),
                      }

hparam_range_dict = {}
hparam_range_dict['loc'] = (0,1)
for hparam_name in cons_range:
    hparam_range_dict[hparam_name] = (uncons_func_dict[hparam_name](cons_range[hparam_name][0]),
                                      uncons_func_dict[hparam_name](cons_range[hparam_name][1]))

meta_opt_init_range_dict = {}
for hparam_name in meta_opt_init_range:
    meta_opt_init_range_dict[hparam_name] = (uncons_func_dict[hparam_name](meta_opt_init_range[hparam_name][0]),
                                             uncons_func_dict[hparam_name](meta_opt_init_range[hparam_name][1]))


if __name__ == '__main__':
    for hparam_name in cons_range:
        print(hparam_name)
        print('\tCons range: ({:6.4e}, {:6.4e})'.format(cons_range[hparam_name][0], cons_range[hparam_name][1]))
        print('\tUncons range: ({:6.4e}, {:6.4e})'.format(hparam_range_dict[hparam_name][0], hparam_range_dict[hparam_name][1]))
        print('\tCons of hparam range: ({:6.4e}, {:6.4e})'.format(cons_func_dict[hparam_name](hparam_range_dict[hparam_name][0]),
                                                                  cons_func_dict[hparam_name](hparam_range_dict[hparam_name][1])))


abbreviation_dict = { 'm': 'momentum',
                      'lr': 'lr',
                      'b1': 'b1',
                      'b2': 'b2',
                      'eps': 'eps',
                      'cs': 'cutoutsize',
                      'hfp': 'hflip_prob',
                      'vfp': 'vflip_prob',
                      'dp': 'dropout_prob',
                      'dps': 'dropouts',
                      'wd': 'weight_decay',
                      'hj': 'hue_jitter',
                      'sj': 'sat_jitter',
                      'cb': 'crop_border',
                      'f-pl': 'fixed-pl',
                      'f': 'fixed',
                      'l': 'linear',
                      'l-pl': 'linear-pl',
                      'p': 'piecewise',
                      'p-pl': 'piecewise-pl',
                      'itd': 'inverse-time-decay',

                      'momentum': 'momentum',
                      'cutoutsize': 'cutoutsize',
                      'hflip_prob': 'hflip_prob',
                      'vflip_prob': 'vflip_prob',
                      'dropout_prob': 'dropout_prob',
                      'dropouts': 'dropouts',
                      'weight_decay': 'weight_decay',
                      'l1': 'l1',
                      'l2': 'l2',
                      'hue_jitter': 'hue_jitter',
                      'sat_jitter': 'sat_jitter',
                      'crop_border': 'crop_border',
                      'fixed-pl': 'fixed-pl',
                      'fixed': 'fixed',
                      'linear': 'linear',
                      'linear-pl': 'linear-pl',
                      'piecewise': 'piecewise',
                      'piecewise-pl': 'piecewise-pl',
                      'inverse-time-decay': 'inverse-time-decay',
                    }
