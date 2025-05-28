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

"""Exponses functions and classes to the module level."""

from typing import Sequence
from sparse_deferred.implicit import matrix

ComputeEngine = matrix.ComputeEngine
Matrix = matrix.Matrix
SparseMatrix = matrix.SparseMatrix
Tensor = matrix.Tensor
DType = matrix.DType
Shape = matrix.Shape


def sum(matrices):  # pylint: disable=redefined-builtin
  return matrix.Sum(*matrices)


def prod(matrices):
  return matrix.Product(*matrices)


def transpose(mat):
  return mat.T
