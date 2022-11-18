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

"""Tests for inference."""

import functools
import os
from typing import Tuple

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np

 import resources
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import incremental
from scaling_transformer_inference_efficiency import inference
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency import layers_parallel


_TOY_HPARAMS = checkpoint.HParams(
    layers=3,
    embed=128,
    ff=256,
    heads=2,
    qkv=32,
    max_len=128,
    vocab=32128,
)

_LOADED_MODEL = None


# PaLM correctness test relies on internal checkpoints
