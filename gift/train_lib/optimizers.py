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

"""Defines different optimizers."""

from flax import optim as optimizers


def get_optimizer(hparams):
  """Constructs  the optimizer from the given HParams.

  Args:
    hparams: Hyper parameters.

  Returns:
    A flax optimizer.
  """
  if hparams.optimizer == 'sgd':
    return optimizers.GradientDescent(
        learning_rate=hparams.lr_hparams['initial_learning_rate'])
  if hparams.optimizer == 'nesterov':
    return optimizers.Momentum(
        learning_rate=hparams.lr_hparams['initial_learning_rate'],
        beta=hparams.opt_hparams.get('momentum', 0.9),
        weight_decay=hparams.opt_hparams.get('weight_decay', 0.0),
        nesterov=True)
  if hparams.optimizer == 'momentum':
    return optimizers.Momentum(
        learning_rate=hparams.lr_hparams['initial_learning_rate'],
        beta=hparams.opt_hparams.get('momentum', 0.9),
        weight_decay=hparams.opt_hparams.get('weight_decay', 0.0),
        nesterov=False)
  if hparams.optimizer == 'adam':
    return optimizers.Adam(
        learning_rate=hparams.lr_hparams['initial_learning_rate'],
        beta1=hparams.opt_hparams.get('beta1', 0.9),
        beta2=hparams.opt_hparams.get('beta2', 0.999),
        eps=hparams.opt_hparams.get('epsilon', 1e-8),
        weight_decay=hparams.opt_hparams.get('weight_decay', 0.0),
    )
  if hparams.optimizer == 'rmsprop':
    return optimizers.RMSProp(
        learning_rate=hparams.lr_hparams.get('initial_learning_rate'),
        beta2=hparams.opt_hparams.get('beta2', 0.9),
        eps=hparams.opt_hparams.get('epsilon', 1e-8))
  else:
    raise NotImplementedError('Optimizer {} not implemented'.format(
        hparams.optimizer))
