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

# python3
"""Fixed tasks containing toy 2d optimization problems."""
from task_set import registry

from task_set.tasks import losg_tasks
from task_set.tasks.losg_problems import problem_generator as pg
from task_set.tasks.losg_problems import problem_spec


def problem_fn_to_problem_definition(problem_fn):
  return lambda: (problem_spec.Spec(problem_fn, (), {}), None, None)


@registry.task_registry.register_fixed("TwoD_Bowl1")
def _():
  problem_fn = lambda: pg.Bowl(condition_number=1., random_seed=1)
  return losg_tasks.LOSGProblemTask(
      problem_fn_to_problem_definition(problem_fn), seed=123)


@registry.task_registry.register_fixed("TwoD_Bowl10")
def _():
  problem_fn = lambda: pg.Bowl(condition_number=10., random_seed=1)
  return losg_tasks.LOSGProblemTask(
      problem_fn_to_problem_definition(problem_fn), seed=123)


@registry.task_registry.register_fixed("TwoD_Bowl100")
def _():
  problem_fn = lambda: pg.Bowl(condition_number=100., random_seed=2)
  return losg_tasks.LOSGProblemTask(
      problem_fn_to_problem_definition(problem_fn), seed=123)


@registry.task_registry.register_fixed("TwoD_Bowl1000")
def _():
  problem_fn = lambda: pg.Bowl(condition_number=1000., random_seed=2)
  return losg_tasks.LOSGProblemTask(
      problem_fn_to_problem_definition(problem_fn), seed=123)


@registry.task_registry.register_fixed("TwoD_Rosenbrock")
def _():
  problem_fn = pg.Rosenbrock
  return losg_tasks.LOSGProblemTask(
      problem_fn_to_problem_definition(problem_fn), seed=123)


@registry.task_registry.register_fixed("TwoD_Ackley")
def _():
  problem_fn = pg.Ackley
  return losg_tasks.LOSGProblemTask(
      problem_fn_to_problem_definition(problem_fn), seed=123)


@registry.task_registry.register_fixed("TwoD_Beale")
def _():
  problem_fn = pg.Beale
  return losg_tasks.LOSGProblemTask(
      problem_fn_to_problem_definition(problem_fn), seed=123)


@registry.task_registry.register_fixed("TwoD_StyblinskiTang")
def _():
  problem_fn = pg.StyblinskiTang
  return losg_tasks.LOSGProblemTask(
      problem_fn_to_problem_definition(problem_fn), seed=123)
