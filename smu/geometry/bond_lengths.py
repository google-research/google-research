"""Extract bond lengths from SMU molecules."""

import apache_beam as beam

import utilities

from smu import dataset_pb2
from smu.parser import smu_utils_lib


MAX_DIST = 2.0



class GetBondLengthDistribution(beam.DoFn):
  """Generates a bond length distribution."""
  def process(self, conformer:dataset_pb2.Conformer):
    bt = conformer.bond_topologies[0]
    geom = conformer.optimized_geometry

    bonded = utilities.bonded(bt)

    natoms = len(bt.atoms)

    for a1 in range(0, natoms):
      atomic_number1 = smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[bt.atoms[a1]]
      for a2 in range(a1 + 1, natoms):
        atomic_number2 = smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[bt.atoms[a2]]
        # Do not process H-H pairs
        if atomic_number1 == 1 and atomic_number2 == 1:
          continue

        d = utilities.distance_between_atoms(geom, a1, a2)
        if d > MAX_DIST:
          continue

        discretized = int(d * utilities.DISTANCE_BINS)
        yield (min(atomic_number1, atomic_number2),
               int(bonded[a1, a2]),
               max(atomic_number1, atomic_number2), discretized), 1
