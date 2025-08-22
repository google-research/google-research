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

"""Flag definition for the setup run-time options.

Flag support needs to be defined outside of the parameter file as the parameter
file get imported in the trainer main and flag definitions have to be in files
imported prior to the start of main to be effective as command-line flags.
"""

from lingvo import compat as tf

tf.flags.DEFINE_string("feature_neighborhood_train_path", None,
                       "Required glob of training files.")
tf.flags.DEFINE_string("feature_neighborhood_dev_path", None,
                       "Required glob of dev files.")
tf.flags.DEFINE_string("feature_neighborhood_test_path", None,
                       "Required glob of test files.")
tf.flags.DEFINE_string("input_symbols", None, "Required path to input_symbols.")
tf.flags.DEFINE_string("output_symbols", None,
                       "Required path to output_symbols.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_boolean("append_eos", True, "Append </s> symbol.")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size.")
tf.flags.DEFINE_integer("max_neighbors", 30, "Maximum number of neighbors.")
tf.flags.DEFINE_integer("max_pronunciation_len", 40,
                        "Maximum padded length of pronunciations.")
tf.flags.DEFINE_integer("max_spelling_len", 20,
                        "Maximum padded length of spellings.")
tf.flags.DEFINE_boolean("neigh_use_tpu", False,
                        "Is this model training on TPU?")
tf.flags.DEFINE_boolean("split_output_on_space", False,
                        "Do we split output tokens on space or by character?")
