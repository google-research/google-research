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

# Download WikiNews dumps from archive.org
#
# Usage: get_wikinews_dumps.sh [output_dir]

set -eux

DEFAULT_OUTDIR="$(dirname $0)/output/download"
OUTDIR="${1:-${DEFAULT_OUTDIR}}"

# Snapshot date and languages to target.
DATE="20190101"
LANG_LIST=(ar de en es fa ja pl ro ta tr uk)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKSUMS="${SCRIPT_DIR}/dump_checksums.txt"
CHECKSUM_TOOL="${SCRIPT_DIR}/../tools/md5_check.sh"

mkdir -p "${OUTDIR}"

for lang in ${LANG_LIST[@]}; do
  echo ">Download '${lang}'..."
  filename="${lang}wikinews-${DATE}-pages-articles.xml.bz2"
  url="https://archive.org/download/${lang}wikinews-${DATE}/${filename}"
  wget -a "${OUTDIR}/wget.log" -O "${OUTDIR}/${filename}" "${url}"
done

echo ">Verify..."
pushd "${OUTDIR}"
bash "${CHECKSUM_TOOL}" "${CHECKSUMS}"
popd

echo
echo ">Done: ${OUTDIR}"
