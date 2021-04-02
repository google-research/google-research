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

"""Script for running experiments and saving the resulting plots."""

import warnings

from dp_multiq import experiment

# To suppress RuntimeWarnings about log(0).  This quantity occurs frequently in
# our code, and doesn't mean that anything is going wrong; numpy.log(0) will
# produce -numpy.inf, which our code handles appropriately.
warnings.simplefilter("ignore", category=RuntimeWarning)

experiment.experiment(methods=[
    experiment.QuantilesEstimationMethod.JOINT_EXP, experiment
    .QuantilesEstimationMethod.IND_EXP, experiment.QuantilesEstimationMethod
    .APP_IND_EXP, experiment.QuantilesEstimationMethod.SMOOTH,
    experiment.QuantilesEstimationMethod.CSMOOTH,
    experiment.QuantilesEstimationMethod.LAP_TREE
])
