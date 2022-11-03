# Gigamol

## Introduction

This package contains libraries and scripts for building the models
described in [Machine learning on DNA-encoded libraries:
A new paradigm for hit-finding](https://pubs.acs.org/doi/10.1021/acs.jmedchem.0c00452).

## Usage

**molecule graph proto**

```
from rdkit import Chem
from gigamol.molecule_graph_proto import molecule_graph

mol = Chem.MolFromSmiles('C(Cl)Cl')
mg = molecule_graph.MoleculeGraph(mol)
pb = mg.to_proto()
[p.graph_distance for p in pb.atom_pairs] == [1, 1, 2]

example = tf.train.Example()
example.features.feature['example_ids'].bytes_list.value.append('test_123'.encode())
example.features.feature['smiles'].bytes_list.value.append('C(Cl)Cl'.encode())
example.features.feature['molecule_graph'].bytes_list.value.append(mg.to_proto().SerializeToString())
example_protos = [example.SerializeToString()]
```
