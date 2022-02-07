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
VENV="${SMUDIR}/venv_smu_dev"

echo "Running protocol compiler"
cd ${ROOTDIR}
protoc --experimental_allow_proto3_optional smu/dataset.proto --python_out=.

echo "Creating virtual environment"
python3 -m venv "${VENV}"
. ${VENV}/bin/activate
cd $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
echo ${ROOTDIR} > smu.pth

# On mac, rdkit-pypi couldn't be found without upgrading pip. No idea why.
echo "Upgrading pip"
python -m pip install --upgrade pip

echo "Installing dependencies"
pip install -r "${SMUDIR}/requirements.txt"

echo "Running tests"
cd "${SMUDIR}"
./run_tests.sh

echo "============================================================="
echo "SUCCESS"
