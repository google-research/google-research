# Verify MD5 checksums of files, using md5sum (Linux) or md5 (MacOS).
#
# Usage: md5_check.sh CHECKSUMS_FILE
#  See documentation of 'md5sum -c' for the format of CHECKSUMS_FILE.

set -eu

if [[ $# -ne 1 ]]; then
  echo "Usage: md5sum.sh checksums.txt"
  exit 1
fi
CHECKSUMS="$1"

if [[ ! -f "${CHECKSUMS}" ]]; then
  echo "! Failed to locate the checksums expected at ${CHECKSUMS}"
  exit 1
fi

if [[ "$(uname)" == "Darwin" ]]; then
  # `md5` included on MacOS requires doing a manual check per listed file.
  #
  # Read the expected checksums and verify each matches the actual checksum of
  # the corresponding file.
  checksums_ok=1
  while IFS= read -r LINE || [[ -n "$LINE" ]]; do
    # LINE has the format <checksum> <filename>.
    expected_checksum="$(echo ${LINE} | cut -f1 -d' ')"
    filename="$(echo ${LINE} | cut -f2 -d' ')"
    if [[ ! -f "${filename}" ]]; then
      echo "${filename}: MISSING"
      checksums_ok=0
    else
      actual_checksum="$(md5 -q ${filename})"
      if [[ "${actual_checksum}" == "${expected_checksum}" ]]; then
        echo "${filename}: OK"
      else
        echo "${filename}: FAILED"
        checksums_ok=0
      fi
    fi
  done < ${CHECKSUMS}

  if [[ ${checksums_ok} -eq 0 ]]; then
    echo "! Checksums did NOT all match."
    exit 1
  fi
else
  # Otherwise assume `md5sum` is available.
  md5sum -c ${CHECKSUMS}
fi
