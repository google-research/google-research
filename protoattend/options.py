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

"""Input flags."""

import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# Input parameters
flags.DEFINE_integer("img_size", 28, "Image size in pixels.")
flags.DEFINE_integer("num_classes", 10, "Number of classes.")

# Model parameters
flags.DEFINE_integer("attention_dim", 16, "Attention dimension.")
flags.DEFINE_integer("val_dim", 64, "Value dimension.")
flags.DEFINE_integer("final_units", 256, "Final units.")
flags.DEFINE_string("normalization", "sparsemax", "normalization.")
flags.DEFINE_float("epsilon_sparsity", 0.000001, "Epsilon.")
flags.DEFINE_float("sparsity_weight", 0.0001, "Sparsity weight.")
flags.DEFINE_float("alpha_intermediate", 0.5,
                   "Coefficient for intermediate loss term.")

# Training parameters
flags.DEFINE_integer("random_seed", 1, "Random seed.")
flags.DEFINE_integer("num_steps", 400000, "Number of training steps.")
flags.DEFINE_integer("display_step", 1000, "Display step.")
flags.DEFINE_integer("val_step", 4000, "Validation step.")
flags.DEFINE_integer("save_step", 15000, "Save step.")
flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate.")
flags.DEFINE_integer("decay_every", 8000, "Decay interval.")
flags.DEFINE_float("decay_rate", 0.9, "Decay rate.")
flags.DEFINE_integer("gradient_thresh", 20, "Gradient clipping threshold.")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_integer("example_cand_size", 1024,
                     "Training candidate database size.")
flags.DEFINE_integer("eval_cand_size", 32768,
                     "Validation candidate database size.")
