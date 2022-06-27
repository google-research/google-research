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
# Shell script for running all unit tests on GCP.
# Example usage: bash run_all_tests.sh

# Automatially stop if one of the unit tests fails.
set -e

# Set up a virtual environment to run in.
virtualenv -p python3 venv
source venv/bin/activate
pip install -r tunas/requirements.txt

# Get the absolute path of the source code base directory.
DIRNAME=$(readlink -f $(dirname $0))

# Set up the PYTHONPATH so that imports will work correctly.
export PYTHONPATH="${DIRNAME}:${PYTHONPATH}"

# Scan through the directory looking for unit tests.
echo "Using base directory ${DIRNAME}"
for filename in $(find "${DIRNAME}" -name \*_test.py -type f)
do
  # Find the filename relative to the parent directory. This means the
  # relative filename will starts with 'tunas/'.
  relative_filename="$(realpath --relative-to="${DIRNAME}/.." "${filename}")"

  # Strip off the ".py" suffix. For example, "tunas/foo_test.py" becomes
  # "tunas/foo_test".
  module="$(echo "${relative_filename}" | sed "s/\.py$//")"

  # Replace slashes with periods. For example, "tunas/foo_test" becomes
  # "tunas.foo_test".
  module="$(echo "${module}" | sed "s/\//./g")"

  # Run the test.
  echo "+" python3 -m "${module}"
  python3 -m "${module}"
  echo
  echo
  echo
done
