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
"""Tests for guided_parameters."""
import math

from absl.testing import absltest
from flax import optim
import numpy as np

from gradient_based_tuning import guided_parameters


class GetOptKeyTest(absltest.TestCase):

  def test_get_opt_key_valid_hp(self):
    out = guided_parameters.get_opt_key_from_var_type('beta1')
    self.assertEqual(out, 'hp-beta1')

  def test_var_type_invalid(self):
    with self.assertRaisesRegex(ValueError, '(?i)Unrecognized.*unk'):
      _ = guided_parameters.get_opt_key_from_var_type('unk')


class GetVarTypeTest(absltest.TestCase):

  def test_get_var_type_valid_hp(self):
    out = guided_parameters.get_var_type_from_opt_key('hp-beta1')
    self.assertEqual(out, 'beta1')

  def test_opt_key_invalid(self):
    with self.assertRaisesRegex(ValueError, '(?i)Invalid.*unk'):
      _ = guided_parameters.get_var_type_from_opt_key('unk')


class GetRawVarsAndActFnsTest(absltest.TestCase):

  def test_empty(self):
    opt_dict = {}
    gv_dict = {}
    raw_vars_dict, act_fn_dict = guided_parameters.get_raw_vars_and_act_fns(
        opt_dict, gv_dict)
    self.assertEqual(raw_vars_dict, {})
    self.assertEqual(act_fn_dict, {})

  def test_ignores_model(self):
    opt_dict = {
        'model':
            optim.GradientDescent(learning_rate=1
                                 ).create(np.array([0.0, 0.1, 0.2, 0.9])),
    }
    gv_dict = {}
    raw_vars_dict, act_fn_dict = guided_parameters.get_raw_vars_and_act_fns(
        opt_dict, gv_dict)
    self.assertEqual(raw_vars_dict, {})
    self.assertEqual(act_fn_dict, {})

  def test_single_guided_parameter(self):
    opt_dict = {
        'hp-beta1':
            optim.GradientDescent(learning_rate=1
                                 ).create(np.array([0.0, 0.1, 0.2, 0.9])),
    }
    gv_dict = {
        'beta1': {
            'learning_rate_scalar': 1,
            'activation_fn': 'linear',
            'activation_ceiling': None,
            'activation_floor': None,
            'clip_min': None,
            'clip_max': None,
        }
    }
    raw_vars_dict, act_fn_dict = guided_parameters.get_raw_vars_and_act_fns(
        opt_dict, gv_dict)
    self.assertListEqual(list(raw_vars_dict['hp-beta1']), [0.0, 0.1, 0.2, 0.9])
    self.assertCountEqual(list(act_fn_dict), ['hp-beta1'])

  def test_multi_guided_parameter(self):
    opt_dict = {
        'hp-beta1':
            optim.GradientDescent(learning_rate=1
                                 ).create(np.array([0.0, 0.1, 0.2, 0.9])),
        'dp-ex_index':
            optim.GradientDescent(learning_rate=1
                                 ).create(np.array([1, 2, 3, 4, 5])),
    }
    gv_dict = {
        'beta1': {
            'learning_rate_scalar': 1,
            'activation_fn': 'linear',
            'activation_ceiling': None,
            'activation_floor': None,
        },
        'ex_index': {
            'learning_rate_scalar': 1,
            'activation_fn': 'linear',
            'activation_ceiling': None,
            'activation_floor': None,
        }
    }
    raw_vars_dict, act_fn_dict = guided_parameters.get_raw_vars_and_act_fns(
        opt_dict, gv_dict)
    self.assertListEqual(list(raw_vars_dict['hp-beta1']), [0.0, 0.1, 0.2, 0.9])
    self.assertListEqual(list(raw_vars_dict['dp-ex_index']), [1, 2, 3, 4, 5])
    self.assertCountEqual(list(act_fn_dict), ['hp-beta1', 'dp-ex_index'])


class GetActivatedHparamsTest(absltest.TestCase):

  def test_empty(self):
    opt_dict = {}
    gv_dict = {}
    out = guided_parameters.get_activated_hparams(opt_dict, gv_dict)
    self.assertEqual(
        out, {
            'hp-beta1': None,
            'hp-decay_rate': None,
            'hp-eps': None,
            'hp-learning_rate': None,
            'hp-weight_decay': None,
            'hp-label_smoothing': None,
        })

  def test_ignores_model(self):
    opt_dict = {
        'model':
            optim.GradientDescent(learning_rate=1
                                 ).create(np.array([0.0, 0.1, 0.2, 0.9])),
    }
    gv_dict = {}
    out = guided_parameters.get_activated_hparams(opt_dict, gv_dict)
    self.assertEqual(
        out, {
            'hp-beta1': None,
            'hp-decay_rate': None,
            'hp-eps': None,
            'hp-learning_rate': None,
            'hp-weight_decay': None,
            'hp-label_smoothing': None,
        })

  def test_single_hparam(self):
    opt_dict = {
        'hp-beta1':
            optim.GradientDescent(learning_rate=1
                                 ).create(np.array([0.0, 0.1, 0.2, 0.9])),
    }
    gv_dict = {
        'beta1': {
            'learning_rate_scalar': 1,
            'activation_fn': 'linear',
            'activation_ceiling': None,
            'activation_floor': None,
            'clip_min': None,
            'clip_max': None,
        }
    }
    out = guided_parameters.get_activated_hparams(opt_dict, gv_dict)
    self.assertListEqual(list(out['hp-beta1']), [0.0, 0.1, 0.2, 0.9])

  def test_single_hparam_learning_rate_scalar(self):
    opt_dict = {
        'hp-beta1':
            optim.GradientDescent(learning_rate=1
                                 ).create(np.array([0.0, 0.1, 0.2, 0.9])),
    }
    gv_dict = {
        'beta1': {
            'learning_rate_scalar': 10,
            'activation_fn': 'linear',
            'activation_ceiling': None,
            'activation_floor': None,
            'clip_min': None,
            'clip_max': None,
        }
    }
    out = guided_parameters.get_activated_hparams(opt_dict, gv_dict)
    self.assertListEqual(list(out['hp-beta1']), [0., 1., 2., 9.])

  def test_clip_low(self):
    opt_dict = {
        'hp-beta1':
            optim.GradientDescent(learning_rate=1
                                 ).create(np.array([0.0, 0.1, 0.2, 0.9])),
    }
    gv_dict = {
        'beta1': {
            'learning_rate_scalar': 1,
            'activation_fn': 'linear',
            'activation_ceiling': None,
            'activation_floor': None,
            'clip_min': 0.1,
            'clip_max': 1,
        }
    }
    out = guided_parameters.get_activated_hparams(opt_dict, gv_dict)
    self.assertSequenceAlmostEqual(list(out['hp-beta1']), [0.1, 0.1, 0.2, 0.9])

  def test_clip_high(self):
    opt_dict = {
        'hp-beta1':
            optim.GradientDescent(learning_rate=1
                                 ).create(np.array([0.0, 0.1, 0.2, 0.9])),
    }
    gv_dict = {
        'beta1': {
            'learning_rate_scalar': 1,
            'activation_fn': 'linear',
            'activation_ceiling': None,
            'activation_floor': None,
            'clip_min': 0,
            'clip_max': 0.5,
        }
    }
    out = guided_parameters.get_activated_hparams(opt_dict, gv_dict)
    self.assertSequenceAlmostEqual(list(out['hp-beta1']), [0.0, 0.1, 0.2, 0.5])

  def test_clip_both(self):
    opt_dict = {
        'hp-beta1':
            optim.GradientDescent(learning_rate=1
                                 ).create(np.array([0.0, 0.1, 0.2, 0.9])),
    }
    gv_dict = {
        'beta1': {
            'learning_rate_scalar': 1,
            'activation_fn': 'linear',
            'activation_ceiling': None,
            'activation_floor': None,
            'clip_min': 0.05,
            'clip_max': 0.5,
        }
    }
    out = guided_parameters.get_activated_hparams(opt_dict, gv_dict)
    self.assertSequenceAlmostEqual(list(out['hp-beta1']), [0.05, 0.1, 0.2, 0.5])

  def test_multiple_hparams(self):
    opt_dict = {
        'hp-beta1':
            optim.GradientDescent(learning_rate=1
                                 ).create(np.array([0.0, 0.1, 0.2, 0.9])),
        'hp-learning_rate':
            optim.GradientDescent(learning_rate=1
                                 ).create(np.array([0.0, 0.1, 0.2, 0.9])),
    }
    gv_dict = {
        'beta1': {
            'learning_rate_scalar': 1,
            'activation_fn': 'linear',
            'activation_ceiling': None,
            'activation_floor': None,
            'clip_min': None,
            'clip_max': None,
        },
        'learning_rate': {
            'learning_rate_scalar': 1,
            'activation_fn': 'linear',
            'activation_ceiling': None,
            'activation_floor': None,
            'clip_min': None,
            'clip_max': None,
        }
    }
    out = guided_parameters.get_activated_hparams(opt_dict, gv_dict)
    self.assertListEqual(list(out['hp-beta1']), [0.0, 0.1, 0.2, 0.9])
    self.assertListEqual(list(out['hp-learning_rate']), [0.0, 0.1, 0.2, 0.9])


class GetActFnFromVarsSubdictTest(absltest.TestCase):

  def test_error_on_empty(self):
    with self.assertRaisesRegex(KeyError, 'activation_fn'):
      _ = guided_parameters.get_act_fn_from_vars_subdict({})

  def test_exp(self):
    dp_vars_type_subdict = {
        'learning_rate_scalar': 1,
        'activation_ceiling': 1,
        'activation_floor': 0,
        'activation_fn': 'exp'
    }
    out = guided_parameters.get_act_fn_from_vars_subdict(dp_vars_type_subdict)
    self.assertEqual(out.keywords, {'steepness': 1, 'ceiling': 1, 'floor': 0})  # pytype: disable=attribute-error
    self.assertAlmostEqual(out(0.5), 1.648721, delta=1e-5)
    self.assertEqual(out(1), math.e)
    self.assertAlmostEqual(out(2), 7.389056, delta=1e-5)

  def test_exp_with_scalar(self):
    dp_vars_type_subdict = {
        'learning_rate_scalar': 2,
        'activation_ceiling': 1,
        'activation_floor': 0,
        'activation_fn': 'exp'
    }
    out = guided_parameters.get_act_fn_from_vars_subdict(dp_vars_type_subdict)
    self.assertEqual(out.keywords, {'steepness': 2, 'ceiling': 1, 'floor': 0})  # pytype: disable=attribute-error
    self.assertAlmostEqual(out(0.25), 1.648721, delta=1e-5)
    self.assertEqual(out(0.5), math.e)
    self.assertAlmostEqual(out(1), 7.389056, delta=1e-5)
    self.assertAlmostEqual(out(2), 54.59815, delta=1e-5)

  def test_exp_with_floor(self):
    dp_vars_type_subdict = {
        'learning_rate_scalar': 1,
        'activation_ceiling': 1,
        'activation_floor': 1,
        'activation_fn': 'exp'
    }
    out = guided_parameters.get_act_fn_from_vars_subdict(dp_vars_type_subdict)
    self.assertEqual(out.keywords, {'steepness': 1, 'ceiling': 1, 'floor': 1})  # pytype: disable=attribute-error
    self.assertAlmostEqual(out(0.5), 2.648721, delta=1e-5)
    self.assertEqual(out(1), 1 + math.e)
    self.assertAlmostEqual(out(2), 8.38906, delta=1e-5)

  def test_sig(self):
    dp_vars_type_subdict = {
        'learning_rate_scalar': 1,
        'activation_ceiling': 2,
        'activation_floor': 0,
        'activation_fn': 'sig'
    }
    out = guided_parameters.get_act_fn_from_vars_subdict(dp_vars_type_subdict)
    self.assertEqual(out.keywords, {'steepness': 1, 'ceiling': 2, 'floor': 0})  # pytype: disable=attribute-error
    self.assertAlmostEqual(out(-50), 0, delta=1e-5)
    self.assertEqual(out(0), 1)
    self.assertAlmostEqual(out(50), 2, delta=1e-5)

  def test_exp_with_ceiling_and_floor(self):
    dp_vars_type_subdict = {
        'learning_rate_scalar': 1,
        'activation_ceiling': 7,
        'activation_floor': 2,
        'activation_fn': 'sig'
    }
    out = guided_parameters.get_act_fn_from_vars_subdict(dp_vars_type_subdict)
    self.assertEqual(out.keywords, {'steepness': 1, 'ceiling': 7, 'floor': 2})  # pytype: disable=attribute-error
    self.assertAlmostEqual(out(-50), 2, delta=1e-5)
    self.assertEqual(out(0), 4.5)
    self.assertAlmostEqual(out(50), 7, delta=1e-5)


if __name__ == '__main__':
  absltest.main()
