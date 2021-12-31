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

"""Utils for Stochastic Polyak solver. Including projections for pytress."""


import jaxopt
import optax
from stochastic_polyak import spsdam_solver
from stochastic_polyak import spsdiag_solver
from stochastic_polyak import spsL1_solver
from stochastic_polyak import spssqrt_solver
from stochastic_polyak import ssps_solver


def get_solver(flags, config, loss_fun, losses):
  """Gets the solver used for training based on FLAGS and config.

  Args:
    flags: Flags passed
    config: Hyperparameter configuration for training and evaluation.
    loss_fun: A loss function that return a real output for each batch
    losses: Loss function that return a b-dimensional output, one for each
      element in the batch

  Returns:
    solver: The optimizaiton solver
    solver_param_name: Name of the solver with parameter choices
  """
  momentum = flags.momentum
  if flags.momentum == 0:
    # Try to get default momentum from config
    momentum = config.get("momentum", flags.momentum)
  delta = flags.slack_delta
  if flags.slack_delta == -1:
    # Try to get default delta from config
    delta = config.get("delta", delta)
  lmbda = flags.slack_lmbda
  if flags.slack_lmbda == -1:
    # Try to get default lmbda from config
    lmbda = config.get("lmbda", lmbda)

  # Initialize solver and parameters.
  if config.solver == "SGD":
    opt = optax.sgd(config.learning_rate, momentum)
    solver = jaxopt.OptaxSolver(opt=opt, fun=loss_fun, has_aux=True)
  elif config.solver == "SPS":
    solver = jaxopt.PolyakSGD(
        fun=loss_fun,
        maxiter=flags.max_steps_per_epoch,
        momentum=momentum,
        delta=delta,
        max_stepsize=config.max_step_size,
        has_aux=True)
    solver_param_name = "m-"+str(momentum)+"-d-"+str(delta)
  elif config.solver == "SPSDam":
    solver = spsdam_solver.SPSDam(fun=loss_fun,
                                  lmbda=lmbda,
                                  lmbda_schedule=config.lmbda_schedule,
                                  momentum=momentum,
                                  has_aux=True)
    solver_param_name = "m-"+str(momentum)+"-lmbda-"+str(lmbda)
  elif config.solver == "SPSL1":
    solver = spsL1_solver.SPSL1(
        fun=loss_fun, lmbda=lmbda, delta=delta, momentum=momentum, has_aux=True)
    solver_param_name = "m-"+str(momentum)+"-lmbda-"+str(lmbda)+"-d-"+str(delta)
  elif config.solver == "SPSsqrt":
    solver = spssqrt_solver.SPSsqrt(
        fun=loss_fun, lmbda=lmbda, momentum=momentum, has_aux=True)
    solver_param_name = "m-"+str(momentum)+"-lmbda-"+str(lmbda)
  elif config.solver == "SSPS":
    solver = ssps_solver.SystemStochasticPolyak(
        fun=losses,
        delta=delta,
        learning_rate=config.learning_rate,
        choose_update=flags.choose_update,
        has_aux=True)
  elif config.solver == "SPSDiag":
    solver = spsdiag_solver.DiagonalStochasticPolyak(
        fun=losses,
        learning_rate=config.learning_rate,
        delta=delta,
        momentum=momentum,
        has_aux=True)
    solver_param_name = "m-"+str(momentum)+"-delta-"+str(delta)
  else:
    raise ValueError("Unknown solver: %s" % config.solver)

  return solver, solver_param_name
