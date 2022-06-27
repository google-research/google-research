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
# Demo of the pair LM system.

set -eou pipefail

readonly TEMPDATA="$(mktemp --directory)"

# On exit, remove temporary directory and deactivate conda.
trap "rm -rf ${TEMPDATA}; conda deactivate" EXIT

# Hyperparameters and seeds.
readonly SEED=10037  # For reproducibility.
readonly BATCH_SIZE=128
readonly MAX_ITERS=10
readonly ORDER=6
readonly SIZE=100000

# File paths; none exist yet.
readonly PAIRS="${TEMPDATA}/pairs.tsv"
readonly LEXICON="${TEMPDATA}/lexicon.txt"
readonly TRAIN="${TEMPDATA}/train.tsv"
readonly DEV="${TEMPDATA}/dev.tsv"
readonly TEST="${TEMPDATA}/test.tsv"
readonly PLM="${TEMPDATA}/plm.fst"

setup() {
  conda env create --file environment.yml
  eval "$(conda shell.bash hook)"
  conda activate pair_ngram
}

download() {
  echo -n "Downloading pairs..."
  curl \
      --silent \
      --output "${PAIRS}" \
      "https://gist.githubusercontent.com/kylebgorman/01adff5799edb0edf3bcce20187c833a/raw/fb0e66d31e021fca7adec4c2104ffea0e879f2e4/pairs.tsv"
  printf "%'d lines\n" "$(wc -l < "${PAIRS}")"
  echo -n "Downloading lexicon..."
  curl \
      --silent \
      --output "${LEXICON}" \
      "http://cvsweb.netbsd.org/bsdweb.cgi/src/share/dict/web2?rev=1.54"
  printf "%'d lines\n" "$(wc -l < "${LEXICON}")"
  echo
}

split() {
  echo "Splitting data..."
  python -m split \
      --seed "${SEED}" \
      --input "${PAIRS}" \
      --train "${TRAIN}" \
      --dev "${DEV}" \
      --test "${TEST}"
  echo
}

train() {
  echo "Training pair LM..."
  python -m train \
     --seed "${SEED}" \
     --batch_size "${BATCH_SIZE}" \
     --max_iters "${MAX_ITERS}" \
     --order "${ORDER}" \
     --tsv "${TRAIN}" \
     --fst "${PLM}"
  echo
}

evaluate() {
  local -r GOLD="$(mktemp --dry-run -p "${TEMPDATA}" tmp.XXXXXX.o.txt)"
  local -r HYPO="$(mktemp --dry-run -p "${TEMPDATA}" tmp.XXXXXX.h.txt)"
  local -r INPUT="$(mktemp --dry-run -p "${TEMPDATA}" tmp.XXXXX.i.txt)"
  cut -f2 "${DEV}" > "${GOLD}"
  cut -f1 "${DEV}" > "${INPUT}"

  echo "Pair LM without lexicon constraint:"
  python -m predict \
      --rule "${PLM}" \
      --input "${INPUT}" \
      --output "${HYPO}"
  python -m error --gold "${GOLD}" --hypo "${HYPO}"
  echo

  echo "Pair LM with lexicon constraint:"
  python -m predict_lexicon \
    --rule "${PLM}" \
    --lexicon "${LEXICON}" \
    --input "${INPUT}" \
    --output "${HYPO}"
  python -m error --gold "${GOLD}" --hypo "${HYPO}"
  echo
}

main() {
  setup
  download
  split
  train
  evaluate
}

main
