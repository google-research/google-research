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

from huggingface_hub import PyTorchModelHubMixin


class BaseTrainer(PyTorchModelHubMixin):
    r"""
    Base class for all trainers - this base class implements the basic functions that we
    need for a trainer.

    The trainer needs to have the following functions:
        - step: takes in a batch of data and performs a step of training
        - loss: takes in a batch of data and returns the loss
        - compute_rewards: takes in a batch of data and returns the rewards
        - _build_models_and_tokenizer: builds the models and tokenizer
        - _build_dataset: builds the dataset
    Each user is expected to implement their own trainer class that inherits from this base
    if they want to use a new training algorithm.
    """

    def __init__(self, config):
        self.config = config

    def step(self, *args):
        raise NotImplementedError("Not implemented")

    def loss(self, *args):
        raise NotImplementedError("Not implemented")

    def compute_rewards(self, *args):
        raise NotImplementedError("Not implemented")

    def _save_pretrained(self, save_directory):
        raise NotImplementedError("Not implemented")
