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

"""Mapping of all defined pipelines in the project.

 pipeline name --> pipeline class.
"""

from gift.pipelines import end2end
from gift.pipelines import gradual_adaptation_with_mixup
from gift.pipelines import manifold_mixup
from gift.pipelines import multi_env_domain_adversarial
from gift.pipelines import multi_env_end2end
from gift.pipelines import multi_env_manifold_mixup
from gift.pipelines import self_adaptive_trainer
from gift.pipelines import student_trainer

SINGLE_ENV_TRAINERS = {
    'end2end': end2end.End2end,
    'manifold_mixup': manifold_mixup.ManifoldMixup
}

SINGLE_ENV_STUDENT_TRAINERS = {
    'student_trainer': student_trainer.StudentEnd2EndTrainer,
    'student_multi_env_trainer': student_trainer.StudentMultiEnvEnd2EndTrainer
}

MULTI_ENV_TRAINERS = {
    'multi_env_end2end':
        multi_env_end2end.MultiEnvEnd2End,
    'self_adaptive_gradual_trainer':
        self_adaptive_trainer.SelfAdaptiveGradualTrainer
}

MULTI_ENV_WITH_REPS_TRAINERS = {
    'multi_env_reps2reps':
        multi_env_end2end.MultiEnvReps2Reps,
    'multi_env_reps2reps_with_hungarian_matching':
        multi_env_end2end.MultiEnvReps2RepsWithHungarianMatching,
    'multi_env_manifold_mixup':
        multi_env_manifold_mixup.MultiEnvManifoldMixup,
    'self_adaptive_gradual_mixup':
        gradual_adaptation_with_mixup.GradualDomainAdaptationWithMixup,
}

MULTI_ENV_SEMI_SUPERVISED_TRAINERS = {
    'multi_env_domain_adversarial':
        multi_env_domain_adversarial.MultiEnvDomainAdverserial
}

ALL_TRAINERS = {}
ALL_TRAINERS.update(SINGLE_ENV_TRAINERS)
ALL_TRAINERS.update(MULTI_ENV_TRAINERS)
ALL_TRAINERS.update(MULTI_ENV_WITH_REPS_TRAINERS)
ALL_TRAINERS.update(SINGLE_ENV_STUDENT_TRAINERS)
ALL_TRAINERS.update(MULTI_ENV_SEMI_SUPERVISED_TRAINERS)


def get_trainer_class(train_mode):
  """Maps trainer name to a tainer class.

  Args:
    train_mode: string; Determines the training strategy (e.g. e2e).

  Returns:
    A dataset builder.
  """
  if train_mode not in ALL_TRAINERS.keys():
    raise ValueError('Unrecognized trainer: {}'.format(train_mode))
  return ALL_TRAINERS[train_mode]
