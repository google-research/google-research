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

#!/bin/bash
set -e

SMUDIR=$(dirname $0)
ROOTDIR=$(cd ${SMUDIR}/..; pwd)
VENV="${SMUDIR}/venv_smu_dev"

if [ -z "${VIRTUAL_ENV}" -a -d "${VENV}" ]; then
    echo "Activating virtual environment in ${VENV}"
    source ${VENV}/bin/activate
fi

set -u

for TESTFN in $(find $SMUDIR -maxdepth 3 -name '*_test.py')
do
    if [[ $TESTFN == *"$VENV"* ]]; then
        continue
    fi
    echo "Executing ${TESTFN}"
    python $TESTFN
done

echo "ALL TESTS PASSED"
