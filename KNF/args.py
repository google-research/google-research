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

"""Configuration file."""
from absl import flags

FLAGS = flags.FLAGS
_SEED = flags.DEFINE_integer("seed", 123, "The random seed.")

_DATASET = flags.DEFINE_string("dataset", "M4",
                               "dataset classes: M4, Cryptos, Traj")

_DATA_DIR = flags.DEFINE_string(
    "data_dir", "data_prep/M4/",
    "Data directory containing train and test data.")

_NUM_FEATS = flags.DEFINE_integer("num_feats", 1, "Number of features.")

_REGULARIZE_RANK = flags.DEFINE_bool(
    "regularize_rank", False, "Whether to regularize dynamics module rank.")

_USE_REVIN = flags.DEFINE_bool("use_revin", True,
                               "Whether to use reinversible normalization.")

_USE_INSTANCENORM = flags.DEFINE_bool("use_instancenorm", True,
                                      "Whether to use instance normalization.")

_ADD_GLOBAL_OPERATOR = flags.DEFINE_bool(
    "add_global_operator", True, "Whether to use a gloabl Koopman operator.")

_ADD_CONTROL = flags.DEFINE_bool("add_control", True,
                                 "Whether to use a control module.")

_DATA_FREQ = flags.DEFINE_string("data_freq", "None",
                                 "The frequency of the time series data.")

_LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.001,
                                    "The initial learning rate.")

_DROPOUT_RATE = flags.DEFINE_float("dropout_rate", 0.0, "The dropout rate.")

_DECAY_RATE = flags.DEFINE_float("decay_rate", 0.9, "The learning decay rate.")

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 128, "The batch size.")

_LATENT_DIM = flags.DEFINE_integer("latent_dim", 64,
                                   "The dimension of latent Koopman space.")
_NUM_STEPS = flags.DEFINE_integer(
    "num_steps",
    5,
    "The number of steps of predictions in one autoregressive call.",
)

_CONTROL_HIDDEN_DIM = flags.DEFINE_integer(
    "control_hidden_dim", 64,
    "The hidden dimension of the module for learning adjustment matrix.")

_NUM_LAYERS = flags.DEFINE_integer(
    "num_layers", 5, "The number of layers in the encoder and decoder.")

_CONTROL_NUM_LAYERS = flags.DEFINE_integer(
    "control_num_layers", 3,
    "The number of layers in the module for learning adjustment matrix.")

_JUMPS = flags.DEFINE_integer(
    "jumps", 5, "The number of skipped steps when genrating sliced samples.")

_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs", 1000, "The maximum number of epochs.")

_MIN_EPOCHS = flags.DEFINE_integer(
    "min_epochs", 60, "The minimum number of epochs the model is trained with.")

_INPUT_DIM = flags.DEFINE_integer(
    "input_dim", 5,
    "The number of observations taken by the encoder at each step")

_INPUT_LENGTH = flags.DEFINE_integer(
    "input_length", 45,
    "The lookback window length for learning Koopman operator.")

_HIDDEN_DIM = flags.DEFINE_integer(
    "hidden_dim", 256, "The hidden dimension of the encoder and decoder.")

_TRAIN_OUTPUT_LENGTH = flags.DEFINE_integer(
    "train_output_length", 10,
    "The training output length for backpropogation.")

_TEST_OUTPUT_LENGTH = flags.DEFINE_integer(
    "test_output_length", 13, "The forecasting horizon on the test set.")

_NUM_HEADS = flags.DEFINE_integer("num_heads", 1, "Transformer number of heads")

_TRANSFORMER_DIM = flags.DEFINE_integer("transformer_dim", 128,
                                        "Transformer feedforward dimension.")

_TRANSFORMER_NUM_LAYERS = flags.DEFINE_integer(
    "transformer_num_layers", 3, "Number of Layers in Transformer Encoder.")

_NUM_SINS = flags.DEFINE_integer("num_sins", -1, "number of sine functions.")

_NUM_POLY = flags.DEFINE_integer("num_poly", -1, "number of sine functions.")

_NUM_EXP = flags.DEFINE_integer("num_exp", -1, "number of sine functions.")
