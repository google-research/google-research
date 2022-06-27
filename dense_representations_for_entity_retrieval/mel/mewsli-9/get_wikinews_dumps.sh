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

# Download WikiNews dumps from archive.org
#
# Usage: get_wikinews_dumps.sh [output_dir]

set -eux

DEFAULT_OUTDIR="$(dirname $0)/output/download"
OUTDIR="${1:-${DEFAULT_OUTDIR}}"

# Snapshot date and languages to target.
DATE="20190101"
LANG_LIST=(ar de en es fa ja sr ta tr)
CHECKSUMS="$(readlink -e $(dirname $0))/dump_checksums.txt"

mkdir -p "${OUTDIR}"

for lang in ${LANG_LIST[@]}; do
  echo ">Download '${lang}'..."
  filename="${lang}wikinews-${DATE}-pages-articles.xml.bz2"
  url="https://archive.org/download/${lang}wikinews-${DATE}/${filename}"
  wget -a "${OUTDIR}/wget.log" -O "${OUTDIR}/${filename}" "${url}"
done

echo ">Verify..."

if [[ ! -f "${CHECKSUMS}" ]]; then
  echo "! Failed to locate the checksums expected at ${CHECKSUMS}"
  exit 1
fi

OLDPWD="${PWD}"
cd "${OUTDIR}"
md5sum -c ${CHECKSUMS}
cd "${OLDPWD}"

echo
echo ">Done: ${OUTDIR}"
