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

"""Runs incremental generation on an example set of text."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import incremental
from scaling_transformer_inference_efficiency import inference
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.layers import layers_pjit


def run(model, quantized):
  """Runs text generation in a simple scenario."""
  the_vocab = checkpoint.load_vocab()

  if quantized:
    ckpt = checkpoint.QuantizedCheckpoint
    params = weights.QuantizedWeights
    if model == '8b':
      cs = checkpoint.CheckpointSpec.PALM_8B_QUANTIZED
    elif model == '62b':
      cs = checkpoint.CheckpointSpec.PALM_62B_QUANTIZED
    elif model == '540b':
      cs = checkpoint.CheckpointSpec.PALM_540B_QUANTIZED
      raise NotImplementedError
  else:
    ckpt = checkpoint.Checkpoint
    params = weights.Weights
    if model == '8b':
      cs = checkpoint.CheckpointSpec.PALM_8B
    elif model == '62b':
      cs = checkpoint.CheckpointSpec.PALM_62B
    elif model == '540b':
      cs = checkpoint.CheckpointSpec.PALM_540B

  loaded_cs = ckpt.load_unsharded_to_host(cs)
  print(jax.tree.map(jnp.shape, loaded_cs))

  the_model = incremental.JittedModel(
      cs.hparams, the_vocab.eos_id,
      functools.partial(inference.infer, cs.hparams,
                        layers_pjit.pjit_transformer_layer),
      weights.physical_axes())
  with the_model.mesh:
    the_weights = params.from_checkpoint(cs.hparams, the_model.mesh, loaded_cs)

  # Generation, split over 3 chunks - using lazy prefix broadcasting
  prompt = incremental.Chunk.tokenize(
      the_vocab, ['[web] Which is worse out of'] * 4, is_first_chunk=True)
  prompt_result = the_model.prefill(the_weights, [], prompt)
  tasks = incremental.Chunk.tokenize(
      the_vocab,
      ['a banana and a pear? Well,'] * 4 + ['a car and a bus? Well,'] * 4,
      is_first_chunk=False)
  tasks_result = the_model.prefill(the_weights, [prompt_result], tasks)
  num_samples = 2
  sample_ids = np.arange(num_samples * 8)
  steps = 32
  temperature = 0.7
  samples, _ = the_model.generate(steps, the_weights,
                                  incremental.Sampling(temperature),
                                  [prompt_result, tasks_result], sample_ids)
  samples = samples.detokenize(the_vocab)
  for sample in samples:
    print(sample)


if __name__ == '__main__':
  import argparse  # pylint: disable = g-import-not-at-top
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, required=True)
  parser.add_argument('--quantized', type=bool, required=True)
  args = parser.parse_args()
  models = ['8b', '62b', '540b']
  if args.model not in models:
    print(f'--model must one of {models}')

  run(args.model, args.quantised)
