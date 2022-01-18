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

"""Things to be gin configurables."""

import gin
import tensorflow  as tf


configurables = {
    'tf.keras.activations': [
        tf.keras.activations.elu,
        tf.keras.activations.exponential,
        tf.keras.activations.gelu,
        tf.keras.activations.hard_sigmoid,
        tf.keras.activations.linear,
        tf.keras.activations.relu,
        tf.keras.activations.selu,
        tf.keras.activations.sigmoid,
        tf.keras.activations.softmax,
        tf.keras.activations.softplus,
        tf.keras.activations.softsign,
        tf.keras.activations.swish,
        tf.keras.activations.tanh,
    ],
    'tf.keras.initializers': [
        tf.keras.initializers.Constant,
        tf.keras.initializers.GlorotNormal,
        tf.keras.initializers.GlorotUniform,
        tf.keras.initializers.HeNormal,
        tf.keras.initializers.HeUniform,
        tf.keras.initializers.Identity,
        tf.keras.initializers.LecunNormal,
        tf.keras.initializers.LecunUniform,
        tf.keras.initializers.Ones,
        tf.keras.initializers.Orthogonal,
        tf.keras.initializers.RandomNormal,
        tf.keras.initializers.RandomUniform,
        tf.keras.initializers.TruncatedNormal,
        tf.keras.initializers.VarianceScaling,
        tf.keras.initializers.Zeros,
    ],
    'tf.keras.layers': [
        tf.keras.layers.GlobalAveragePooling1D,
        tf.keras.layers.GlobalAveragePooling2D,
        tf.keras.layers.GlobalAveragePooling3D,
        tf.keras.layers.GlobalMaxPool1D,
        tf.keras.layers.GlobalMaxPool2D,
        tf.keras.layers.GlobalMaxPool3D,
    ],
    'tf.keras.losses': [
        tf.keras.losses.BinaryCrossentropy,
        tf.keras.losses.CategoricalCrossentropy,
        tf.keras.losses.CategoricalHinge,
        tf.keras.losses.CosineSimilarity,
        tf.keras.losses.Hinge,
        tf.keras.losses.Huber,
        tf.keras.losses.KLDivergence,
        tf.keras.losses.LogCosh,
        tf.keras.losses.MeanAbsoluteError,
        tf.keras.losses.MeanAbsolutePercentageError,
        tf.keras.losses.MeanSquaredError,
        tf.keras.losses.MeanSquaredLogarithmicError,
        tf.keras.losses.Poisson,
        tf.keras.losses.SparseCategoricalCrossentropy,
        tf.keras.losses.SquaredHinge,
    ],
    'tf.keras.metrics': [
        tf.keras.metrics.AUC,
        tf.keras.metrics.Accuracy,
        tf.keras.metrics.BinaryAccuracy,
        tf.keras.metrics.BinaryCrossentropy,
        tf.keras.metrics.CategoricalAccuracy,
        tf.keras.metrics.CategoricalCrossentropy,
        tf.keras.metrics.CosineSimilarity,
        tf.keras.metrics.MeanAbsoluteError,
        tf.keras.metrics.MeanSquaredError,
        tf.keras.metrics.RootMeanSquaredError,
        tf.keras.metrics.SparseCategoricalAccuracy,
        tf.keras.metrics.SparseCategoricalCrossentropy,
        tf.keras.metrics.SparseTopKCategoricalAccuracy,
        tf.keras.metrics.TopKCategoricalAccuracy,
    ],
    'tf.keras.optimizers': [
        tf.keras.optimizers.Adadelta,
        tf.keras.optimizers.Adagrad,
        tf.keras.optimizers.Adam,
        tf.keras.optimizers.Adamax,
        tf.keras.optimizers.Ftrl,
        tf.keras.optimizers.Nadam,
        tf.keras.optimizers.RMSprop,
        tf.keras.optimizers.SGD,
    ],
}

for module, configurables in configurables.items():
  for configurable in configurables:
    gin.config.external_configurable(configurable, module=module)
