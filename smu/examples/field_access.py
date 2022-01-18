# This examples shows how to do some basic field access in SMU.
# The dataset.proto file has complete documentation for the available fields
# and structure.

from smu import smu_sqlite

db = smu_sqlite.SMUSQLite('20220104_standard.sqlite', 'r')

# This is an arbitrary choice of the conformers to use.
conformer = db.find_by_conformer_id(57001)

print('We will examine conformer with id', conformer.conformer_id)

print('The computed properties are generally in the .properties field')

print('Scalar values are access by name (note the .value suffix),',
      'like this single point energy: ',
      conformer.properties.single_point_energy_atomic_b5.value)

print('Fields with repeated values',
      'like harmonic_intensities and excitation_energies_cc2)',
      'use an index with [] on the repeated values')

print('The 0th and 6th harmonic_intensities:',
      conformer.properties.harmonic_intensities.value[0],
      conformer.properties.harmonic_intensities.value[6])

print('Or you can iterate over all of them')
for value in conformer.properties.excitation_energies_cc2.value:
    print('Excitation energy:', value)

print('Or just ask how many excitation_energies_cc2 there are:',
      len(conformer.properties.excitation_energies_cc2.value))

print('Some fields like rotational_constants have explicit x,y,z components')

print(conformer.properties.rotational_constants.x,
      conformer.properties.rotational_constants.y,
      conformer.properties.rotational_constants.z)

print('A couple of important fields are not inside "properties"')

geometry = conformer.optimized_geometry
print('The geometry has positions for',
      len(geometry.atom_positions),
      'atoms and the first atom x-coordinate is',
      geometry.atom_positions[0].x)

print('In addition to looking at dataset.proto for field documentation,',
      'you can just print a given conformer to see what fields are available')

print(conformer)
