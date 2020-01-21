# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Modes the model can be in."""


class Modes(object):
  """Definition of the mode the model is functioning in."""

  # Model is in a training state. No streaming is done.
  TRAINING = 'TRAINING'

  # Below are three options for inference:

  # Model is in inference mode and has state for efficient
  # computation/streaming, where state is kept inside of the model
  STREAM_INTERNAL_STATE_INFERENCE = 'STREAM_INTERNAL_STATE_INFERENCE'

  # Model is in inference mode and has state for efficient
  # computation/streaming, where state is received from outside of the model
  STREAM_EXTERNAL_STATE_INFERENCE = 'STREAM_EXTERNAL_STATE_INFERENCE'

  # Model its in inference mode and it's topology is the same with training
  # mode (with removed droputs etc)
  NON_STREAM_INFERENCE = 'NON_STREAM_INFERENCE'
