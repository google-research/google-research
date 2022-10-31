#!/bin/sh

GEOMS="""
initial_geometry_energy initial_geometry_energy_deprecated
initial_geometry_gradient_norm initial_geometry_gradient_norm_deprecated
optimized_geometry_energy optimized_geometry_energy_deprecated
optimized_geometry_gradient_norm optimized_geometry_gradient_norm_deprecated
rotational_constants rotational_constants_deprecated
nuclear_repulsion_energy nuclear_repulsion_energy_deprecated
"""

# Done manually on just a few files
#molecule rdkit_molecule
# Molecule RDKitMolecule
M_TO_RM="""
SmuMolecule TopologyMolecule
smu_molecule topology_molecule
"""

CONFORMER="""
Conformer Molecule
conformer molecule
CONFORMER MOLECULE
Chem.Molecule Chem.Conformer
AddMolecule AddConformer
.GetMolecule .GetConformer
"""

CONFORMER_PART2="""
confid molid
conf_id mol_id
cid mid
conf mol
molidence confidence
molig config
amid acid
mollicting conflicting
mollict conflict
molormer conformer
"""

FATE_PART1="""
Molecule.FATE Properties.FATE
.fate .properties.errors.fate
"""

FATE_PART2="""
Molecule.FateCategory Properties.FateCategory
"""

FATE_PART3="""
FATE_GEOMETRY_OPTIMIZATION_PROBLEM FATE_FAILURE_GEO_OPT
FATE_DISASSOCIATED FATE_FAILURE_TOPOLOGY_CHECK
FATE_DISCARDED_OTHER FATE_FAILURE_STAGE2
FATE_NO_CALCULATION_RESULTS FATE_FAILURE_NO_RESULTS
FATE_CALCULATION_WITH_SERIOUS_ERROR FATE_ERROR_SERIOUS
FATE_CALCULATION_WITH_MAJOR_ERROR FATE_ERROR_MAJOR
FATE_CALCULATION_WITH_MODERATE_ERROR FATE_ERROR_MODERATE
FATE_CALCULATION_WITH_WARNING_SERIOUS FATE_SUCCESS_ALL_WARNING_SERIOUS
FATE_CALCULATION_WITH_WARNING_VIBRATIONAL FATE_SUCCESS_ALL_WARNING_MEDIUM_VIB
FATE_SUCCESS FATE_SUCCESS_ALL_WARNING_LOW
"""

FATE_PART4="""
FATE_SUCCESS_ALL_WARNING_LOW_ALL FATE_SUCCESS_ALL
"""

WHICH_DB="""
which_database which_database_deprecated
"""

THIS_REPLACE="${FATE_PART4}"
ARGS="-i"
#ARGS=""

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
FILES=$(find ${SCRIPT_DIR}/.. -type f -maxdepth 4 | grep -v venv | grep -v dataset_pb2.py | grep -v rename.sh | egrep -v '\.(cc|h|pyc)$' | grep -v '~$')

#for f in ${FILES}; do
#    echo "$f"
#done

while IFS= read -r line; do
    if [ -z "$line" ]; then
       continue
    fi
    repl=($line)
    #echo "X: ${repl[0]}"
    #echo "Y: ${repl[1]}"
    "${HOME}/bin/multi-repl" ${ARGS} "${repl[0]}" "${repl[1]}" $FILES < /dev/tty
done <<< "$THIS_REPLACE"
