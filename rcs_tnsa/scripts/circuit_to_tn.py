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

"""Script to convert Cirq.circuit into .graph and .groups files for TN contraction optimization."""

import argparse
import collections
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm


def circuit_to_tensor_network_edges(
    circuit, graph_filename, groups_filename, open_output=False
):
  """Generate .graph and .groups files from a Cirq circuit.

  Args:
      circuit: The input Cirq circuit.
      graph_filename: The name of the file to save the graph connectivity.
      groups_filename: The name of the file to save the fSim groups.
      open_output: Output indices are open. Default is *False*.

  Raises:
      RuntimeError: if indices from fSim gates are not processed correctly.
  """

  qubit_to_node = {q: i for i, q in enumerate(sorted(circuit.all_qubits()))}
  next_node = len(qubit_to_node)
  edges = []
  edge_num = 0
  node_to_qubits_and_edges = collections.defaultdict(list)
  node_to_gate = {n: None for n in qubit_to_node.values()}

  # Iterate over gates.
  for moment in circuit:
    for g in moment:
      # Assign a unique node to each gate
      gate_node = next_node
      next_node += 1

      # Edges connecting nodes.
      for qubit in sorted(g.qubits):
        edges.append((qubit_to_node[qubit], gate_node))
        node_to_qubits_and_edges[gate_node].append((qubit, edge_num))
        node_to_qubits_and_edges[qubit_to_node[qubit]].append((qubit, edge_num))
        node_to_gate[gate_node] = g.gate
        qubit_to_node[qubit] = gate_node
        edge_num += 1

  # Open output edges
  if open_output:
    for qubit in sorted(circuit.all_qubits()):
      edges.append((qubit_to_node[qubit], -1))
      # Output indices are not sliceable, hence not in groups.
      edge_num += 1

  # Generate groups or sets of indices that can be sliced together.
  groups = []
  for node, qubit_edge in node_to_qubits_and_edges.items():
    gate = node_to_gate[node]
    if len(qubit_edge) == 4 and isinstance(gate, cirq.FSimGate):
      qubit0, edge0 = qubit_edge[0]
      qubit1, edge1 = qubit_edge[1]
      qubit2, edge2 = qubit_edge[2]
      qubit3, edge3 = qubit_edge[3]
      if qubit0 == qubit2 and qubit1 == qubit3:
        groups += [(edge0, edge3), (edge1, edge2)]
      elif qubit0 == qubit3 and qubit1 == qubit2:
        groups += [(edge0, edge2), (edge1, edge3)]
      else:
        raise RuntimeError('FSim qubit configuration is inconsistent.')
  all_groups = set(range(edge_num))
  for group in groups:
    for edge in group:
      all_groups.discard(edge)
  all_groups = sorted((edge,) for edge in all_groups)
  for group in groups:
    all_groups.append(tuple(sorted(edge for edge in group)))

  # Write the edges to output .graph file
  with open(graph_filename, 'w') as f:
    for edge in edges:
      f.write(f'2 {edge[0]} {edge[1]}\n')

  # Write groups of edges to output .groups file.
  with open(groups_filename, 'w') as f:
    for group in all_groups:
      f.write(' '.join(str(edge) for edge in group) + '\n')


def circuit_from_qsim(file):
  """Reads a circuit from a .qsim file.

  Args:
      file: file to read.

  Returns:
      A cirq.Circuit object.
  """
  lines = file.readlines()

  # Get qubits.
  num_qubits = int(lines[0].strip())
  qubits = [cirq.LineQubit(i) for i in range(num_qubits)]

  # Read lines of moment numbers and gates.
  moment_nums = []
  gates = []
  for line in lines[1:]:
    words = line.strip().split()
    moment_nums.append(int(words[0]))
    gate_name = words[1]

    if gate_name == 'fs':
      qubit_nums = map(int, words[2:4])
      theta = float(words[4])
      phi = float(words[5])
      gate = cirq.FSimGate(theta=theta, phi=phi)
      gates.append(gate.on(*(qubits[qn] for qn in qubit_nums)))
    elif gate_name in ('x_1_2', 'y_1_2', 'hz_1_2', 'rz'):
      qubit_num = int(words[2])
      if gate_name == 'x_1_2':
        gate = cirq.X**0.5
      elif gate_name == 'y_1_2':
        gate = cirq.Y**0.5
      elif gate_name == 'hz_1_2':
        gate = cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5)
      elif gate_name == 'rz':
        rads = float(words[3])
        gate = cirq.rz(rads)
      else:
        raise ValueError(f'Unsupported gate: {gate_name}')
      gates.append(gate.on(qubits[qubit_num]))
    else:
      raise ValueError(f'Unsupported gate: {gate_name}')

  # Create moments.
  gates_by_moment = [[] for _ in range(max(moment_nums) + 1)]
  for moment_num, gate in zip(moment_nums, gates):
    gates_by_moment[moment_num].append(gate)
  moments = [cirq.Moment(moment) for moment in gates_by_moment]

  # Create and return a cirq.Circuit
  return cirq.Circuit(moments)


def main():
  """Convert Cirq circuit into .graph and .groups files for TN contraction optimization."""
  parser = argparse.ArgumentParser(
      description=(
          'Convert Cirq circuit into .graph and .groups files for TN'
          ' contraction optimization.'
      )
  )

  # Add arguments.
  parser.add_argument(
      'input_filename',
      type=str,
      help='Input filename with json, qasm, or qsim circuit.',
  )
  parser.add_argument(
      'graph_filename', type=str, help='Output .graph filename.'
  )
  parser.add_argument(
      'groups_filename', type=str, help='Output .groups filename.'
  )
  parser.add_argument(
      '--open_output',
      action='store_true',
      help='If set, it creates the tensor network with open ouptut indices.',
  )
  # Parse the arguments
  args = parser.parse_args()

  # Access the arguments.
  input_filename = args.input_filename
  graph_filename = args.graph_filename
  groups_filename = args.groups_filename
  open_output = args.open_output

  # Read input file format.
  if '.' not in input_filename:
    raise ValueError('Input file has no specific format.')
  input_format = input_filename.split('.')[1]
  accepted_formats = ('json', 'qasm', 'qsim')

  # Read circuit from file.
  with open(input_filename, 'r') as f:
    if input_format == 'json':
      circuit = cirq.read_json(f)
    elif input_format == 'qasm':
      circuit = circuit_from_qasm(f.read())
    elif input_format == 'qsim':
      circuit = circuit_from_qsim(f)
    else:
      raise ValueError(f'Input format has to be one of: {accepted_formats}')

  # Process circuit.
  circuit_to_tensor_network_edges(
      circuit, graph_filename, groups_filename, open_output=open_output
  )


if __name__ == '__main__':
  main()
