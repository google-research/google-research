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

"""Module for building the classifier network."""
import gin
import tensorflow as tf


@gin.configurable
def build_classifier(observation_spec, action_spec, input_noise=1.0):
  """Builds the classifier network."""
  sa_classifier = tf.keras.Sequential([
      tf.keras.layers.Dense(32, activation="relu"),
      tf.keras.layers.Dense(2, activation=None),
  ])
  advantage_classifier = tf.keras.Sequential([
      tf.keras.layers.Dense(32, activation="relu"),
      tf.keras.layers.Dense(2, activation=None),
  ])
  s_dim = observation_spec.shape.as_list()[0]
  a_dim = action_spec.shape.as_list()[0]
  sas_input = tf.keras.layers.Input((2 * s_dim + a_dim,))
  sa_input = sas_input[:, :-s_dim]
  if input_noise > 0:
    noisy_sas_input = tf.keras.layers.GaussianNoise(input_noise)(sas_input)
    noisy_sa_input = tf.keras.layers.GaussianNoise(input_noise)(sa_input)
  else:
    noisy_sas_input = sas_input
    noisy_sa_input = sa_input
  sa_logits = sa_classifier(noisy_sa_input)
  sa_probs = tf.nn.softmax(sa_logits, name="SA_probs")
  advantage_logits = advantage_classifier(noisy_sas_input)
  sas_logits = sa_logits + advantage_logits
  sas_probs = tf.nn.softmax(sas_logits, name="SAS_probs")
  sas_classifier = tf.keras.Model(
      inputs=sas_input, outputs=[sa_probs, sas_probs])
  return sas_classifier
