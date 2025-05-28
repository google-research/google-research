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

"""Package for dealing with different models."""

from typing import Any, Dict

from models import base
from models import lstm_seq2seq
from models import lstm_seq2seq_saf
from models import tft
from models import tft_saf
import tensorflow as tf


def get_model_type(model_type, chosen_hparams,
                   loss_form):
  """Return a forecast model based on the type."""
  if loss_form == "MAE":
    training_loss_object = tf.keras.losses.MeanAbsoluteError(
        name="training_loss")
    self_supervised_loss_object = tf.keras.losses.MeanAbsoluteError(
        name="self_supervised_loss")
  elif loss_form == "MSE":
    training_loss_object = tf.keras.losses.MeanSquaredError(
        name="training_loss")
    self_supervised_loss_object = tf.keras.losses.MeanSquaredError(
        name="self_supervised_loss")
  else:
    raise Exception("The loss type is not supported")

  if model_type == "lstm_seq2seq":
    model = lstm_seq2seq.ForecastModel(
        loss_object=training_loss_object, hparams=chosen_hparams)
  elif model_type == "lstm_seq2seq_saf":
    model = lstm_seq2seq_saf.ForecastModel(
        loss_object=training_loss_object,
        self_supervised_loss_object=self_supervised_loss_object,
        hparams=chosen_hparams)
  elif model_type == "tft":
    model = tft.ForecastModel(
        loss_object=training_loss_object, hparams=chosen_hparams)
  elif model_type == "tft_saf":
    model = tft_saf.ForecastModel(
        loss_object=training_loss_object,
        self_supervised_loss_object=self_supervised_loss_object,
        hparams=chosen_hparams)
  else:
    raise Exception("The chosen model is not supported")

  return model
