# coding=utf-8
# Copyright 2019 The Google Research Authors.
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
"""Tests for edward2_autoreparam.models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow_probability import edward2 as ed
from edward2_autoreparam import models


class ModelsTest(tf.test.TestCase):

  def _sanity_check_conversion(self, model, model_args, observed, to_cp, to_ncp,
                               make_to_cp):

    with ed.tape() as model_tape:
      model(*model_args)

    model_tape_ = self.evaluate(model_tape)
    example_params = list(model_tape_.values())[:-1]

    # Test that `make_to_cp`, when given the centered parameterization as the
    # source, generates the identity fn.
    param_names = [p for v in model_tape_.keys() for p in (v + '_a', v + '_b')]
    centered_parameterization = {p: 1. for p in param_names}
    identity_cp = make_to_cp(**centered_parameterization)
    example_params_copy = identity_cp(example_params)
    c1_ = self.evaluate(example_params_copy)
    c2_ = self.evaluate(example_params_copy)
    self.assertAllClose(c1_, c2_)
    self.assertAllClose(c1_, example_params)

    # Test that `to_ncp` and `to_cp` are deterministic and consistent
    ncp_params = to_ncp(example_params)
    cp_params = to_cp(ncp_params)

    ncp_params_, cp_params_ = self.evaluate((ncp_params, cp_params))
    ncp_params2_, cp_params2_ = self.evaluate((ncp_params, cp_params))
    # Test determinism
    self.assertAllClose(ncp_params_, ncp_params2_)
    self.assertAllClose(cp_params_, cp_params2_)

    # Test round-trip consistency:
    self.assertAllClose(cp_params_, example_params)

  def test_german_credit_lognormal(self):

    (model, model_args, observed,
     to_cp, to_ncp, make_to_cp) = models.get_german_credit_lognormalcentered()

    self._sanity_check_conversion(model, model_args, observed, to_cp, to_ncp,
                                  make_to_cp)

  def test_radon_stddvs(self):
    (model, model_args, observed,
     to_cp, to_ncp, make_to_cp) = models.get_radon_model_stddvs()
    self._sanity_check_conversion(model, model_args, observed, to_cp, to_ncp,
                                  make_to_cp)

  def test_eight_schools(self):
    (model, model_args, observed,
     to_cp, to_ncp, make_to_cp) = models.get_eight_schools()
    self._sanity_check_conversion(model, model_args, observed, to_cp, to_ncp,
                                  make_to_cp)


if __name__ == '__main__':
  tf.test.main()
