# Copyright 2022 The Google Research Authors.
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

#!/bin/sh

set -e

SMUDIR="`cd $(dirname $0); pwd`"
ROOTDIR="`cd "${SMUDIR}/.."; pwd`"
VENV="${SMUDIR}/venv_smu_user"

echo "Running protocol compiler"
cd ${ROOTDIR}
protoc --experimental_allow_proto3_optional smu/dataset.proto --python_out=.

echo "Creating virtual environment"
python3 -m venv "${VENV}"
. ${VENV}/bin/activate
cd $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
echo ${ROOTDIR} > smu.pth

echo "Installing dependencies"
pip install -r "${SMUDIR}/requirements_user.txt"

echo "Running tests"
cd "${SMUDIR}"
# Parser test used gfile, so we exclude it
#python -m smu.parser.smu_parser_test
# Query test relies on parser, which uses gfile, so we exclude it
#python -m smu.query_sqlite_test
python -m smu.smu_sqlite_test
python -m smu.geometry.topology_from_geom_test
python -m smu.geometry.bond_length_distribution_test
python -m smu.geometry.utilities_test
python -m smu.geometry.smu_molecule_test

echo "============================================================="
echo "Success! Before running query_sqlite, please make sure to do:"
echo "source ${VENV}/bin/activate"
echo "Example command line:"
echo "python -m smu.query_sqlite --input_sqlite <path to standard.sqlite> --smiles NN"
