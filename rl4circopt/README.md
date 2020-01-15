# rl4circopt: A library for quantum circuit optimization.


## Description

This is a physics engine for quantum circuit optimization.
For a quantum circuit defined by basic representation objects in `circuit`,
the user can scan the circuit and apply transformation rules defined in
`rules` to generate a new circuit.

## Hello Circuit

A simple example to start:

```
# From google-research/
python -m rl4circopt/hello_circuit
```

## Convert to / from Cirq

[Cirq](https://github.com/quantumlib/Cirq) is a python framework for creating, editing, and invoking Noisy Intermediate Scale Quantum (NISQ) circuits.

`cirq_converter` library is a tool to convert gates, operations and circuits
between our types to Cirq's.
