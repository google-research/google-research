# Copyright 2020 The Google Research Authors.
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

# Download wikiextractor from Github and apply our patch that disables its link
# filtering.

# Usage: get_wikiextractor.sh [output_dir]
#   output_dir defaults to a local sub-directory if not provided.

set -eux

DEFAULT_OUTDIR="$(dirname $0)/wikiextractor_repo"
OUTDIR="${1:-${DEFAULT_OUTDIR}}"

PATCH="$(readlink -e $(dirname $0))/wikiextractor.patch"
if [[ ! -f "${PATCH}" ]]; then
  echo "! Failed to locate the patch file expected at ${PATCH}"
  exit 1
fi

echo ">Download external wikiextractor tool..."
OLDPWD=${PWD}
git clone https://github.com/attardi/wikiextractor.git "${OUTDIR}"
cd "${OUTDIR}"

echo
echo ">Apply custom patch.."
git checkout -b linkfilter_off 16186e290d
git apply "${PATCH}"
cd "${OLDPWD}"

echo
echo ">Done: ${OUTDIR}"
