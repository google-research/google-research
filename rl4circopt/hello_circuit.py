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

"""Hello Circuit."""

from absl import app

from rl4circopt import circuit
from rl4circopt import rules


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  rule = rules.ExchangeCommutingOperations()
  circ = circuit.Circuit(7, [
      circuit.Operation(circuit.ControlledZGate(), [0, 1]),
      circuit.Operation(circuit.RotZGate(0.42), [0]),
      circuit.Operation(circuit.ControlledZGate(), [1, 2]),
      circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [1]),
      circuit.Operation(circuit.ControlledZGate(), [0, 1]),
      circuit.Operation(circuit.ControlledZGate(), [1, 2])
  ])

  transformations = tuple(rule.scan(circ))

  # Show 4 transformations.
  print(transformations)


if __name__ == '__main__':
  app.run(main)
