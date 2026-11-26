# Copyright 2026 The Google Research Authors.
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
"""Regression tests for the architecture definitions in the KIP notebook."""

import ast
import functools
import json
import math
from pathlib import Path
import types
import unittest


class _FakeStax:
  """Records network layers without importing JAX or neural_tangents."""

  def Relu(self):
    return ('relu',)

  def Conv(self, width, kernel_shape, **kwargs):
    return ('conv', width, kernel_shape, kwargs)

  def Dense(self, units, **kwargs):
    return ('dense', units, kwargs)

  def Flatten(self):
    return ('flatten',)

  def serial(self, *layers):
    return layers


def _convolutional_network():
  notebook = json.loads(
      (Path(__file__).parent / 'KIP.ipynb').read_text(encoding='utf-8'))
  for cell in notebook['cells']:
    source = ''.join(cell.get('source', []))
    if 'def FullyConvolutionalNetwork(' not in source:
      continue
    for node in ast.parse(source).body:
      if isinstance(node, ast.FunctionDef):
        if node.name == 'FullyConvolutionalNetwork':
          stub = _FakeStax()
          namespace = {
              'stax': stub,
              'np': types.SimpleNamespace(sqrt=math.sqrt),
              'functools': functools,
          }
          module = ast.Module(body=[node], type_ignores=[])
          exec(compile(module, 'KIP.ipynb', 'exec'), namespace)
          return namespace['FullyConvolutionalNetwork']
  raise AssertionError('FullyConvolutionalNetwork is missing from the notebook')


class KIPConvolutionalArchitectureTest(unittest.TestCase):

  def test_zero_depth_initializes_output_layers(self):
    network = _convolutional_network()
    layers = network(depth=0, width=16, num_classes=4)
    self.assertEqual([layer[0] for layer in layers], ['flatten', 'dense'])
    self.assertEqual(layers[-1][1], 4)

  def test_multiple_conv_layers_precede_classifier(self):
    network = _convolutional_network()
    layers = network(
        depth=3, width=32, num_classes=5, parameterization='standard')
    self.assertEqual(
        [layer[0] for layer in layers],
        ['conv', 'relu', 'conv', 'relu', 'conv', 'relu', 'flatten', 'dense'])
    for layer in layers[::2][:3]:
      self.assertEqual(layer[1:3], (32, (3, 3)))
      self.assertEqual(layer[3]['parameterization'], 'standard')
    self.assertEqual(layers[-1][1], 5)


if __name__ == '__main__':
  unittest.main()
