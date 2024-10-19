# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Tests for benchmark."""

from absl import app
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency.usage import benchmarks


def main(argv):
  del argv
  batch = 32
  seqlen = 2048
  num_samples = 1
  benchmarks.benchmark_generate('basic', checkpoint.HParams.PALM_8B, batch,
                                seqlen, num_samples)


if __name__ == '__main__':
  app.run(main)
