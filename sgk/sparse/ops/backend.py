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

"""Exposes C++ operations in Python."""
import tensorflow.compat.v1 as tf

try:
  from sgk.sparse.ops.cc import gen_sparse_ops  # pylint: disable=g-import-not-at-top
  kernels = gen_sparse_ops
except ImportError:
  # TODO(tgale): Avoid harcoding the library path.
  kernels = tf.load_op_library("/usr/local/sgk/lib/libsgk.so")
