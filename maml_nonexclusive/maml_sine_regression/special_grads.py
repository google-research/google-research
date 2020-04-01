# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: disable=line-too-long
# pylint: disable=bad-indentation
# pylint: disable=bad-continuation
# pylint: disable=protected-access
"""Code for second derivatives not implemented in TensorFlow library."""
from tensorflow.compat.v1.python.framework import ops
from tensorflow.compat.v1.python.ops import array_ops
from tensorflow.compat.v1.python.ops import gen_nn_ops


@ops.RegisterGradient("MaxPoolGrad")
def _MaxPoolGradGrad(op, grad):
    # pylint: disable=protected-access
    gradient = gen_nn_ops._max_pool_grad(op.inputs[0], op.outputs[0],
            grad, op.get_attr("ksize"), op.get_attr("strides"),
            padding=op.get_attr("padding"), data_format=op.get_attr("data_format"))
    gradgrad1 = array_ops.zeros(shape=array_ops.shape(op.inputs[1]), dtype=gradient.dtype)
    gradgrad2 = array_ops.zeros(shape=array_ops.shape(op.inputs[2]), dtype=gradient.dtype)
    return (gradient, gradgrad1, gradgrad2)
