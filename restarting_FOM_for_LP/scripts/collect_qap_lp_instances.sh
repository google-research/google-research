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
# This script illustrates how to collect the instances used in the paper.
# Note it assumes that wget (https://www.gnu.org/software/wget/) and
# cc are avaliable (https://www.computerhope.com/unix/ucc.htm).
#
# This collects the instances:
# - nug08-3rd and nug20 from http://plato.asu.edu/ftp/lptestset/nug/
# - qap15 from http://plato.asu.edu/ftp/lptestset/
# - qap10 from https://miplib.zib.de/WebData/instances/

if [[ "$#" != 2 ]]; then
  echo "Usage: collect_qap_lp_instances.sh temporary_directory" \
    "output_directory" 1>&2
  exit 1
fi

TEMP_DIR="$1"
DEST_DIR="$2"

mkdir -p "${TEMP_DIR}" || exit 1
mkdir -p "${DEST_DIR}" || exit 1

# Download and compile the netlib tool for uncompressing files in the
# "compressed MPS" format.
wget --directory-prefix="${TEMP_DIR}" -nv http://www.netlib.org/lp/data/emps.c
cc -O3 -o "${TEMP_DIR}/emps" "${TEMP_DIR}/emps.c"

wget --directory-prefix="${DEST_DIR}" \
  https://miplib.zib.de/WebData/instances/qap10.mps.gz

for f in qap15; do
  wget -nv -O - "http://plato.asu.edu/ftp/lptestset/${f}.mps.bz2" | bzcat | \
       gzip > "${DEST_DIR}/${f}.mps.gz"
done

for f in nug08-3rd nug20; do
  instance="$(basename $f)"
  wget -nv -O - "http://plato.asu.edu/ftp/lptestset/nug/${f}.bz2" | bzcat | \
      "${TEMP_DIR}/emps" | gzip > "${DEST_DIR}/${instance}.mps.gz"
done

# Delete the temporary directory.
echo "This script created a temporary directory, to delete run:"
echo "rm -f -r ${TEMP_DIR}"
