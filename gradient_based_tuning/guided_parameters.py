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
"""Guided Parameters config functions for JAX."""
import copy
import functools
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf
# pylint: disable=g-import-not-at-top
try:
  import guided_parameters_utils
except ImportError:
  from gradient_based_tuning import guided_parameters_utils
# pylint: enable=g-import-not-at-top

_HYPER_PARAMETER_TYPES = [
    'beta1',
    'decay_rate',
    'eps',
    'label_smoothing',
    'learning_rate',
    'weight_decay',
]


def get_hyper_parameter_types():
  return _HYPER_PARAMETER_TYPES


def get_opt_key_from_var_type(var_type):
  """Converts guided parameter var_type into optimizer_dict key."""
  if var_type in get_hyper_parameter_types():
    return 'hp-%s' % var_type
  raise ValueError('Unrecognized: %s' % var_type)


def get_var_type_from_opt_key(opt_key):
  """Converts optimizer_dict key into guided parameter var_type."""
  split_key = opt_key.split('p-')
  if len(split_key) != 2:
    raise ValueError('Invalid opt_key: %s' % opt_key)
  return split_key[1]


def get_raw_vars_and_act_fns(
    optimizer_dict,
    guided_vars_dict,
):
  """Builds dicts of raw vars and activation fns for access in guidance step.

  Args:
    optimizer_dict: a dict of in-use optimizers
    guided_vars_dict: a dict of configurations for the guided parameters

  Returns:
    A dict mapping all guided parameter optimizer keys to associated raw vars,
      if present in optimizer_dict, or None (unused keys must be present to be
      checked in JAX concretization-safe manner within guidance step)
    A dict mapping all present optimizer keys to their associated activation fns
  """
  raw_vars_dict = {}
  act_fn_dict = {}
  # loop through and put the raw_vars / act_fns in dicts for easy access
  for guided_opt_key in optimizer_dict:
    if guided_opt_key == 'model':
      continue  # skip the model
    dp_vars_subdict_key = get_var_type_from_opt_key(guided_opt_key)
    cur_dp_vars_subdict = guided_vars_dict[dp_vars_subdict_key]
    cur_dp_optimizer = optimizer_dict[guided_opt_key]
    cur_raw_vars = cur_dp_optimizer.target
    raw_vars_dict[guided_opt_key] = cur_raw_vars
    cur_act_fn = get_act_fn_from_vars_subdict(cur_dp_vars_subdict)
    act_fn_dict[guided_opt_key] = cur_act_fn
  return raw_vars_dict, act_fn_dict


def get_activated_hparams(optimizer_dict, guided_vars_dict):
  """Gets dict of activated hparams from optimizer_dict and guided_vars_dict."""
  all_hparams = get_hyper_parameter_types()
  activated_hparams = {'hp-%s' % x: None for x in all_hparams}
  for hp in all_hparams:
    if 'hp-%s' % hp in optimizer_dict:
      cur_dp_vars_subdict = guided_vars_dict[hp]
      act_fn = get_act_fn_from_vars_subdict(cur_dp_vars_subdict)
      activated_vars = act_fn(optimizer_dict['hp-%s' % hp].target)
      # jnp.clip does not allow both a_min and a_max to be None, so check first.
      if cur_dp_vars_subdict['clip_min'] is not None or cur_dp_vars_subdict[
          'clip_max'] is not None:
        activated_vars = jnp.clip(
            jnp.array(activated_vars), cur_dp_vars_subdict['clip_min'],
            cur_dp_vars_subdict['clip_max'])
      activated_hparams['hp-%s' % hp] = activated_vars
  return activated_hparams


def get_noninit_guided_config(
    guided_hparam_types,
    model_opt_type,
    lr_activation_fn='exp',
    activation_floor=0,
    guided_opt_type='adam',
    init_dict=None,
    learning_rate_scalar_override=None,
):
  """Return non-initialized guided configs."""
  default_config = {
      'coordinate_size': 1,
      'coordinate_shape': [1],
      'reset_per_epoch': False,
      'reset_per_batch': False,
      'init': 'const',
      'init_scalar': 1,
      'optimizer_type': guided_opt_type,
      'trainable': True,
      'learning_rate_scalar': 1,
      'activation_fn': 'sig',
      'activation_ceiling': 1,
      'activation_floor': 0,
      'use_window': True,
      'grad_window': None,
      'override_guided_lr': 0.0,
  }
  default_values = {
      'beta1': 0.9,
      'beta2': 0.999,
      'decay_rate': 0.8,
      'epsilon': 1e-6,
      'label_smoothing': 1e-5,
      'learning_rate': 1,
      'weight_decay': 1e-5,
  }
  learning_rate_scalars = {
      'beta1': 1,
      'beta2': 1,
      'decay_rate': 1,
      'epsilon': 1,
      'label_smoothing': 1,
      'learning_rate': 1,
      'weight_decay': 1,
  }
  if learning_rate_scalar_override is not None:
    learning_rate_scalars.update(learning_rate_scalar_override)
  if init_dict is not None:
    default_values.update(init_dict)
  allopt_lr = dict(
      default_config, **{
          'init_scalar': default_values['learning_rate'],
          'activation_fn': lr_activation_fn,
          'activation_floor': activation_floor,
          'learning_rate_scalar': learning_rate_scalars['learning_rate'],
      })
  lamb_adam_beta1 = dict(
      default_config, **{
          'init_scalar': default_values['beta1'],
          'activation_fn': 'sig',
          'activation_ceiling': 1 - 1e-5,
          'learning_rate_scalar': learning_rate_scalars['beta1'],
      })
  adagrad_beta1 = dict(
      default_config, **{
          'init_scalar': default_values['beta1'],
          'activation_fn': 'sig',
          'activation_ceiling': 1 - 1e-5,
          'learning_rate_scalar': learning_rate_scalars['beta1'],
      })
  lamb_adam_beta2 = dict(
      default_config, **{
          'init_scalar': default_values['beta2'],
          'activation_fn': 'sig',
          'activation_ceiling': 1 - 1e-5,
          'learning_rate_scalar': learning_rate_scalars['beta2'],
      })
  lamb_adam_eps = dict(
      default_config, **{
          'init_scalar': default_values['epsilon'],
          'activation_fn': 'sig',
          'activation_floor': 1e-10,
          'learning_rate_scalar': learning_rate_scalars['epsilon'],
      })
  lamb_adam_weight_decay = dict(
      default_config, **{
          'init_scalar': default_values['weight_decay'],
          'activation_fn': 'sig',
          'activation_ceiling': 1,
          'learning_rate_scalar': learning_rate_scalars['weight_decay'],
      })
  adagrad_eps = dict(
      default_config, **{
          'init_scalar': default_values['epsilon'],
          'activation_fn': 'sig',
          'learning_rate_scalar': learning_rate_scalars['epsilon'],
      })
  momentum_beta = dict(
      default_config, **{
          'init_scalar': default_values['beta1'],
          'activation_fn': 'sig',
          'learning_rate_scalar': learning_rate_scalars['beta1'],
      })
  momentum_weight_decay = dict(
      default_config, **{
          'init_scalar': default_values['weight_decay'],
          'activation_fn': 'sig',
          'activation_ceiling': 1,
          'learning_rate_scalar': learning_rate_scalars['weight_decay'],
      })
  adafactor_beta1 = dict(
      default_config,
      **{
          # 0 by default (momentum off), but must be non-zero to learn
          'init_scalar': default_values['beta1'],
          'activation_fn': 'sig',
          'learning_rate_scalar': learning_rate_scalars['beta1'],
      })
  adafactor_decay_rate = dict(
      default_config, **{
          'init_scalar': default_values['decay_rate'],
          'activation_fn': 'sig',
          'learning_rate_scalar': learning_rate_scalars['decay_rate'],
      })
  label_smoothing = dict(
      default_config, **{
          'coordinate_shape': 1,
          'init_scalar': default_values['label_smoothing'],
          'learning_rate_scalar': learning_rate_scalars['label_smoothing'],
          'activation_fn': 'exp',
          'activation_ceiling': 1,
          'activation_floor': 0,
      })
  ret = {}
  if 'label_smoothing' in guided_hparam_types:
    ret.update({'label_smoothing': label_smoothing})
  guide_all_available_hparams = 'all' in guided_hparam_types

  if 'lrsig' in guided_hparam_types:
    allopt_lr.update({
        'activation_fn': 'sig',
        'activation_ceiling': 20,
        'activation_floor': 0.05,
    })
  if 'learning_rate' in guided_hparam_types or guide_all_available_hparams:
    ret.update({'learning_rate': allopt_lr})

  if model_opt_type == 'lamb':
    if 'beta1' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'beta1': lamb_adam_beta1})
    if 'beta2' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'beta2': lamb_adam_beta2})
    if 'eps' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'eps': lamb_adam_eps})
    if 'weight_decay' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'weight_decay': lamb_adam_weight_decay})
  elif model_opt_type == 'adam':
    if 'beta1' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'beta1': lamb_adam_beta1})
    if 'beta2' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'beta2': lamb_adam_beta2})
    if 'eps' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'eps': lamb_adam_eps})
    if 'weight_decay' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'weight_decay': lamb_adam_weight_decay})
  elif model_opt_type == 'adagrad':
    if 'beta1' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'beta1': adagrad_beta1})
    if 'eps' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'eps': adagrad_eps})
  elif model_opt_type == 'mom':
    if 'beta' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'beta': momentum_beta})
    if 'weight_decay' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'weight_decay': momentum_weight_decay})
  elif model_opt_type == 'adafactor':
    if 'beta1' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'beta1': adafactor_beta1})
    if 'decay_rate' in guided_hparam_types or guide_all_available_hparams:
      ret.update({'decay_rate': adafactor_decay_rate})
  elif model_opt_type in ['sgd', 'gd', 'gradient_descent']:
    pass  # Has no hyper-params other than
  else:
    raise ValueError('Unrecognized FLAGS.model_opt_type: %s' % model_opt_type)

  if not ret:
    raise ValueError('Failed to add any guided parameters: %s' % model_opt_type)

  return ret


def get_act_fn_from_vars_subdict(dp_vars_type_subdict):
  """Returns the specific activation function of the dp_var_type_subdict."""
  return functools.partial(
      guided_parameters_utils.get_activation_fn_by_name(
          dp_vars_type_subdict['activation_fn']),
      steepness=dp_vars_type_subdict['learning_rate_scalar'],
      ceiling=dp_vars_type_subdict['activation_ceiling'],
      floor=dp_vars_type_subdict['activation_floor'])


def create_raw_vars_array(init_dict):
  """Initialize the raw vars array."""
  init_fn = guided_parameters_utils.get_init_by_name(init_dict['init'])
  if init_dict['activation_fn'] == 'sigcent':
    if init_dict['init_scalar'] == 1:
      inv_array = jnp.zeros(1, dtype=jnp.float32)
    else:
      raise ValueError(
          'Must init to 1 when using sigcent activation fn. Failed with init_scalar=%s'
          % init_dict['init_scalar'])
  else:
    init_array = init_fn(1, init_dict['init_scalar'])

    inv_fn = guided_parameters_utils.get_activation_inverter_by_name(
        init_dict['activation_fn'])
    inv_array = inv_fn(
        init_array,
        steepness=init_dict['learning_rate_scalar'],
        ceiling=init_dict['activation_ceiling'],
        floor=init_dict['activation_floor'])
  return inv_array


def init_guided_vars(guided_vars_dict):
  """Adds 'raw_guided_vars' field to all guided params in guided_vars_dict.

  For each dict in the guided_vars_dict, produce the initialized ndarray as
  determined by the init, coordinate_shape, learning_rate_scalar, and
  activation_fn fields of the single_var_dict, and assigned it to the
  'raw_guided_vars' key of the subdict in question.

  Args:
    guided_vars_dict: dict without raw vars initialized

  Returns:
    an initialized guided_vars_dict
  """
  for var_type in guided_vars_dict:
    guided_vars_dict[var_type]['raw_guided_vars'] = create_raw_vars_array(
        guided_vars_dict[var_type])
  return guided_vars_dict


def reset_subdict_raw_vars(guided_vars_dict, var_type):
  """Reset var_type of the guided_vars_dict, overriding current state."""
  old_subdict = guided_vars_dict[var_type]
  guided_vars_dict[var_type]['raw_guided_vars'] = create_raw_vars_array(
      old_subdict)
  return guided_vars_dict


def get_guided_vars_dict(
    guided_hparam_types,
    model_opt_type,
    guided_opt_type='adam',
    lr_activation_fn='exp',
    activation_floor=0,
    init_dict=None,
    learning_rate_scalar_override=None,
):
  """Returns the initialized guided_vars_dict specified by guided_var_set.

  Args:
    guided_hparam_types: which hparams to guide
    model_opt_type: model optimizer type
    guided_opt_type: meta-optimizer type
    lr_activation_fn: activation fn for learning rate, if LR is guided
    activation_floor: min value for guided activation fns
    init_dict: initialization values (post-activation) for guided hparams
    learning_rate_scalar_override: dict overriding learning rate scalars, keyed
      by guided_hparam_types

  Returns:
    an initialized guided_vars_dict

  The guided_vars_dict is a dict of individual guided variable configurations
  each individual dict represents a single dimension for guided-parameters
  ie 'ex_index', 'dppl_bucket', 'bert_tgt_bucket'.
  """

  guided_vars_dict_noninit = get_noninit_guided_config(
      guided_hparam_types=guided_hparam_types,
      model_opt_type=model_opt_type,
      lr_activation_fn=lr_activation_fn,
      activation_floor=activation_floor,
      guided_opt_type=guided_opt_type,
      init_dict=init_dict,
      learning_rate_scalar_override=learning_rate_scalar_override,
  )
  return init_guided_vars(guided_vars_dict_noninit)


def make_dict_json_safe(vars_dict):
  """Iterate through a pytree, turn DeviceArrays into pickle-able ndarrays."""
  for k in vars_dict:
    if isinstance(vars_dict[k], dict):
      vars_dict[k] = make_dict_json_safe(vars_dict[k])
    elif isinstance(vars_dict[k], jax.Array):
      vars_dict[k] = jnp.asarray(vars_dict[k])
    elif isinstance(vars_dict[k], np.ndarray):
      vars_dict[k] = vars_dict[k].tolist()
  return vars_dict


def save_guided_vars_dict(guided_vars_dict, model_dir):
  """Save the guided vars dict as a pickled dict."""
  guided_vars_dict_local = copy.deepcopy(guided_vars_dict)
  guided_vars_dict_local = make_dict_json_safe(guided_vars_dict_local)
  json_file = os.path.join(model_dir, 'guided_vars_dict.pkl')
  with tf.io.gfile.GFile(json_file, 'wb') as f:
    pickle.dump(guided_vars_dict_local, f)
  txt_file = os.path.join(model_dir, 'guided_vars_dict.txt')
  with tf.io.gfile.GFile(txt_file, 'wb') as f:
    f.write(str(guided_vars_dict_local))


def load_guided_vars_dict(model_dir):
  """Load guided_vars_dict from a pickled dict saved."""
  json_file = os.path.join(model_dir, 'guided_vars_dict.pkl')
  with tf.io.gfile.GFile(json_file, 'rb') as f:
    guided_vars_dict = pickle.load(f)
  return guided_vars_dict


def save_epoch(epoch, model_dir):
  """Saves the current epoch in a text file."""
  epoch_file = os.path.join(model_dir, 'cur_epoch.txt')
  with tf.io.gfile.GFile(epoch_file, 'w') as f:
    f.write(str(epoch))


def load_epoch(model_dir):
  """Reads the current epoch from a text file, returns 0 if not found."""
  epoch_file = os.path.join(model_dir, 'cur_epoch.txt')
  try:
    with tf.io.gfile.GFile(epoch_file, 'r') as f:
      cur_epoch = int(f.read())
    return cur_epoch
  except (tf.errors.NotFoundError, ValueError):
    # If there is no cur_epoch saved, it is the start of training, so return 0.
    return 0


def save_guide_loop_metrics(
    model_dir,
    current_step,
    metrics_guide_loop_list,
    learned_vars_list,
    track_ex_guide_loop,
):
  """Save guide loop metrics and learned vars in a pickled dict."""
  output_dir = os.path.join(model_dir, 'guide_loop_analysis')
  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)

  # aggregate metric data
  metrics_loop_dict = {}
  for metrics_guide_single_step_dict in metrics_guide_loop_list:
    for key, val in metrics_guide_single_step_dict.items():
      if 'guide_loop' not in key:
        continue
      if key not in metrics_loop_dict:
        metrics_loop_dict[key] = []
      metrics_loop_dict[key].append(jnp.asarray(val).tolist())

  # aggregate learned vars data
  learned_vars_loop_dict = {}
  for learned_vars_single_step_dict in learned_vars_list:
    for key, val in learned_vars_single_step_dict.items():
      if key not in learned_vars_loop_dict:
        learned_vars_loop_dict[key] = []
      learned_vars_loop_dict[key].append(val)
  learned_vars_loop_dict.update(metrics_loop_dict)
  file_name = 'guide_loop_step_%s%s.pkl' % (current_step, '_tracked'
                                            if track_ex_guide_loop else '')
  pkl_file = os.path.join(output_dir, file_name)
  with tf.io.gfile.GFile(pkl_file, 'wb') as f:
    pickle.dump(learned_vars_loop_dict, f)
