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

"""Use contrack custom ops in python."""

import os

from rules_python.python.runfiles import runfiles
from tensorflow.python.framework import load_library

manifest = os.environ.get("RUNFILES_MANIFEST_FILE")
directory = os.environ.get("RUNFILES_DIR")

r = runfiles.Create()
CUSTOM_OPS_LIB = r.Rlocation("contrack/_custom_ops.so")
custom_ops = load_library.load_op_library(CUSTOM_OPS_LIB)

sequence_concat = custom_ops.SequenceConcat
new_id = custom_ops.NewId
