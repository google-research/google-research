# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Parse output of topology_from_geom.

Process records with dupl.

topologies.

Input looks like:

Smiles,StartSmi,conformer_id,NBts,IsStart
CCC1=NNNN1,CCC1=NNNN1,6104990001,1,True
CC(N)C(N)NN,CC(N)C(N)NN,6103750002,1,True

Use the input to create a mapping between smiles and BondTopologyId
and then scan the data again, adding the right BondTopologyId to the
found smiles.
"""

from typing import List
from absl import app
from absl import flags
from absl import logging
import pandas as pd
from rdkit import Chem

from tensorflow import gfile
# from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("input", None, "Glob files created by topology_from_geom")


def ring_atoms(mol):
  """Return a number that describes the ring membership of `mol`.

  The number encodes the number of each kind of atom that is in a ring.
  The algorithm below works because there will never be >9 atoms in
  a molecule.
  Args:
    mol: Molecule

  Returns:
    a whole number
  """
  result = 0
  for atom in mol.GetAtoms():
    if atom.IsInRing():
      z = atom.GetAtomicNum()
      if z == 6:
        result += 1
      elif z == 7:
        result += 100
      elif z == 8:
        result += 1000

  return result


def reconcile_dupes(unused_argv):
  """Adds found bond topology ids to the output from topology_from_geom."""
  del unused_argv

  # For each input file, a dataframe.
  df_list: List[pd.Dataframe] = []

  for filepath in gfile.glob(FLAGS.input):
    logging.info("Opening %s", filepath)
    with gfile.GFile(filepath, "r") as f:
      df_list.append(
          pd.read_csv(
              f,
              names=[
                  "Smiles", "StartSmi", "id", "Fate", "NBts", "RingAtoms",
                  "IsStart"
              ]))

  data = pd.concat(df_list)
  del df_list
  logging.info(data.shape)

  # Convert conformer_ids to bond_topology_id by dividing by 1000
  # Expect many dupes to be overwritten here.
  smiles_to_id = {k: v for k, v in zip(data["StartSmi"], data["id"] // 1000)}

  # We only care about the cases where there is a BondTopology mismatch.
  interesting = data.loc[not data["IsStart"]]
  logging.info(interesting.shape)
  # Convert the two smiles columns to molecules.
  mstart = [Chem.MolFromSmiles(x) for x in interesting["StartSmi"]]
  mfound = [Chem.MolFromSmiles(x) for x in interesting["Smiles"]]

  # Ring score for each of the molecules.
  rstart = [ring_atoms(m) for m in mstart]
  rfound = [ring_atoms(m) for m in mfound]
  same_ring_membership = 0
  different_ring_membership = 0
  no_smiles = 0
  print("FoundSmiles,StartSmi,StartId,FoundId,FoundScore,StartScore")
  for i, scores in enumerate(zip(rfound, rstart)):
    found_score = scores[0]
    start_score = scores[1]
    if found_score == start_score:
      same_ring_membership += 1
      continue

    different_ring_membership += 1
    found_smiles = interesting.iloc[i, 0]
    other_bt = smiles_to_id.get(found_smiles, "*")
    if other_bt == "*":
      message = f"smiles {found_smiles}, not known"
      logging.info(message)
      no_smiles += 1
    print(
        f"{interesting.iloc[i,0]},{interesting.iloc[i, 1]},{interesting.iloc[i,2]},{other_bt},{found_score},{start_score}"
    )

  logging.info("%d molecules different smiles but same ring membership",
               same_ring_membership)
  logging.info("%d of %d items have different ring membership",
               different_ring_membership, interesting.shape[0])
  logging.info("%d items had unrecognised smiles", no_smiles)

if __name__ == "__main__":
  flags.mark_flag_as_required("input")
  app.run(reconcile_dupes)
