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
