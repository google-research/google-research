# Download wikiextractor from Github and apply our patch that disables its link
# filtering.

# Usage: get_wikiextractor.sh [output_dir]
#   output_dir defaults to a local sub-directory if not provided.

set -eux

DEFAULT_OUTDIR="$(dirname $0)/wikiextractor_repo"
OUTDIR="${1:-${DEFAULT_OUTDIR}}"

if [[ ! -f "${OUTDIR}/WikiExtractor.py" ]]; then
  PATCH="$(cd "$(dirname "$0")" && pwd)/wikiextractor.patch"
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
else
  echo ">Use WikiExtractor already present in ${OUTDIR}"
  echo "(To start over, clear the above directory and rerun this script.)"
fi

echo
echo ">Done: ${OUTDIR}"
