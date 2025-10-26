#!/bin/sh
# Copyright 2025 The Google Research Authors.
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


#
# Trains a pair language transliteration model given a romanization lexicon.
#
# Example usage:
# LOCALAKDIR="/path/to/local/aksharantar/subdirectory"
# ./train_pair_lm.sh \
#   --input_lexicon="${LOCALAKDIR}"/brx.tsv \
#   --output_dir="${LOCALAKDIR}"/brx_model

INPUT_LEXICON=""
OUTPUT_DIR=""
NGORDER=3     # Default n-gram model order is 3; use --order flag to change.
UPDATES=20    # Default number of EM updates in Baum Welch training.
BATCH_SIZE=0  # Default training batch size (unused since this value is 0).
ALPHA=0       # Default alpha parameter for Baum Welch training.

# EDIT THESE VARIABLES to point to required binary executables.
# Paths to binary executables from the OpenFst library:
FARPATH=/path/to/binary/executables/far
FSTPATH=/path/to/binary/executables/fst
# Path to binary executables from the OpenGrm Baum Welch library:
BAUMWELCHPATH=/path/to/binary/executables/baumwelch
# Path to binary executables from the OpenGrm NGram library:
NGRAMPATH=/path/to/binary/executables/ngram

while [[ $# != 0 ]]; do
  # Parses 'option=optarg' word.
  OPT="$(awk 'BEGIN { split(ARGV[1], a, "="); print a[1] }' "${1}")"
  ARG="$(awk 'BEGIN { split(ARGV[1], a, "="); print a[2] }' "${1}")"
  shift
  case "${OPT}" in
    --input_lexicon|-input_lexicon)
      # Full path to romanization lexicon TSV for training.
      INPUT_LEXICON="${ARG}" ;;
    --output_dir|-output_dir)
      # Full path to local output directory.
      OUTPUT_DIR="${ARG}" ;;
    --order|-order)
      NGORDER="${ARG}" ;;
    --updates|-updates)
      UPDATES="${ARG}" ;;
    --batch_size|-batch_size)
      BATCH_SIZE="${ARG}" ;;
    --alpha|-alpha)
      ALPHA="${ARG}" ;;
    *)
      echo "bad option: ${OPT}"
      exit 1 ;;
  esac
done

if [[ -z ${INPUT_LEXICON} || -z ${OUTPUT_DIR} ]]; then
  echo "--input_lexicon and --output_dir options are required!"
  exit 1
fi

mkdir -p ${OUTPUT_DIR}
TMPDIR="$(mktemp -d -p ${OUTPUT_DIR})"

# Copies over specified romanization lexicons for use as training.
cat "${INPUT_LEXICON}" | shuf >"${TMPDIR}"/train.tsv
cat "${TMPDIR}"/train.tsv | awk '{print $1}' >"${TMPDIR}"/train.input.txt
cat "${TMPDIR}"/train.tsv | awk '{print $2}' >"${TMPDIR}"/train.output.txt

# Compiles input and output strings into FAR archives for training access.
"${FARPATH}/farcompilestrings" \
  --far_type=sttable \
  --token_type=utf8 \
  "${TMPDIR}"/train.input.txt "${TMPDIR}"/train.input.far
"${FARPATH}/farcompilestrings" \
  --far_type=sttable \
  --token_type=utf8 \
  "${TMPDIR}"/train.output.txt "${TMPDIR}"/train.output.far

# Extracts input and output vocabulary labels from compiled FAR archives.
"${FARPATH}/farprintstrings" "${TMPDIR}"/train.input.far |\
  awk '{for (i=1; i <= NF; i++) {print $i}}' |\
  sort -nu >"${TMPDIR}"/train.ilabels.txt
"${FARPATH}/farprintstrings" "${TMPDIR}"/train.output.far |\
  awk '{for (i=1; i <= NF; i++) {print $i}}' |\
  sort -nu >"${TMPDIR}"/train.olabels.txt

# Uses input and output vocabulary labels to compile an initial uniform model.
# This is essentially an unweighted finite-state transducer allowing all inputs
# to map to all outputs, as well as deletions and insertions of all input and
# output labels, respectively. This is a single state, cost-free transducer.
# Adds input label deletion arcs, mapping input indices to 0 (<epsilon>).
cat "${TMPDIR}"/train.ilabels.txt |\
  awk 'BEGIN {printf("0\n")} {printf("0\t0\t%d\t0\n",$1)}' \
  >"${TMPDIR}"/train.init_mod.txt
# Adds substitution and insertion arcs for each output label, mapping all
# input indices and 0 (<epsilon>) to the output label.
cat "${TMPDIR}"/train.olabels.txt |\
  while read OL; do
    cat "${TMPDIR}"/train.ilabels.txt |\
      awk -v OL=$OL '{printf("0\t0\t%d\t%d\n",$1,OL)}
      END {printf("0\t0\t0\t%d\n",OL)}'
  done >>"${TMPDIR}"/train.init_mod.txt
"${FSTPATH}/fstcompile" "${TMPDIR}"/train.init_mod.txt |\
  "${FSTPATH}/fstarcsort" >"${TMPDIR}"/train.init_mod.fst

# Creates a random starting model and then trains alignment via EM.
"${BAUMWELCHPATH}/baumwelchrandomize" \
  "${TMPDIR}"/train.init_mod.fst \
  "${TMPDIR}"/train.init_rand_mod.fst
"${BAUMWELCHPATH}/baumwelchtrain" \
  --batch_size="${BATCH_SIZE}" \
  --max_iters="${UPDATES}" \
  --alpha="${ALPHA}" \
  "${TMPDIR}"/train.input.far \
  "${TMPDIR}"/train.output.far \
  "${TMPDIR}"/train.init_rand_mod.fst \
  "${TMPDIR}"/train.baum_welch_mod.fst

# Gradually increases the order of the alignment model to desired order.
CURR_MOD="${TMPDIR}"/train.baum_welch_mod.fst
ORDER=1
while [[ "$ORDER" -le "${NGORDER}" ]]; do
  CURR_OUT="${TMPDIR}/train.baum_welch_mod.${ORDER}"
  # Decodes to a FAR of transducer strings mapping from source to target.
  "${BAUMWELCHPATH}/baumwelchdecode" \
    "${TMPDIR}"/train.input.far \
    "${TMPDIR}"/train.output.far \
    "${CURR_MOD}" \
    "${CURR_OUT}".far \
    "${CURR_OUT}".map

  # Trains n-gram language model from output alignments to FST automaton.
  "${NGRAMPATH}/ngramcount" \
    --order="${ORDER}" \
    --require_symbols=false \
    "${CURR_OUT}".far |\
    "${NGRAMPATH}/ngrammake" --method=witten_bell --witten_bell_k=15 \
    >"${CURR_OUT}".pairlm.fst

  # Converts FST automaton n-gram language model to pair LM transducer.
  "${FSTPATH}/fstencode" \
    --decode \
    "${CURR_OUT}".pairlm.fst \
    "${CURR_OUT}".map |\
    "${FSTPATH}/fstarcsort" >"${CURR_OUT}".pairlm.trans.fst
  CURR_MOD="${CURR_OUT}".pairlm.trans.fst
  ORDER=$((ORDER+1))
done

FINAL_TMP="${TMPDIR}/final.pair${ORDER}g"
FINAL_MOD="${OUTPUT_DIR}/final.pair${ORDER}g"

# Creates automaton format of pair LM with explicit source/target codepoints,
# and removes insertion/deletion pairs from unigram state of final model.
"${FSTPATH}/fstprint" "${CURR_OUT}".pairlm.trans.fst |\
  awk '{if (NF > 3 && ($3 != 0 || $4 != 0)) {printf("%d;%d\n",$3,$4)}}' |\
  "${NGRAMPATH}/ngramsymbols" >"${FINAL_TMP}".syms
"${FSTPATH}/fstprint" "${CURR_OUT}".pairlm.trans.fst |\
  awk '{if (NF < 3) {print} else {if ($3 == 0 && $4 == 0) {
       printf("%d\t%d\t%s\t%s\t%s\n",$1,$2,"<epsilon>","<epsilon>",$NF)}
       else {printf("%d\t%d\t%d;%d\t%d;%d\t%s\n",$1,$2,$3,$4,$3,$4,$NF)}}}' |\
  "${FSTPATH}/fstcompile" \
  --isymbols="${FINAL_TMP}".syms --keep_isymbols \
  --osymbols="${FINAL_TMP}".syms --keep_osymbols |\
  "${FSTPATH}/fstarcsort" |\
  "${FSTPATH}/fstprint" |\
  awk '{if (NR == 1) {unigram_st = $2};
  if ($1 != unigram_st) {print} else {
  if (NF < 3 ||
      (substr($3,1,2) != "0;" && substr($3,length($3)-1,2) != ";0")) {
  print}}}' >"${FINAL_TMP}".txt

# Compiles final model and outputs automaton to the desired output directory.
"${FSTPATH}/fstcompile" --isymbols="${FINAL_TMP}".syms --keep_isymbols \
    --osymbols="${FINAL_TMP}".syms --keep_osymbols \
    "${FINAL_TMP}".txt >"${FINAL_MOD}".fst

# Converts final automaton to a transducer and writes to output directory.
"${FSTPATH}/fstprint" "${FINAL_MOD}".fst |\
  sed 's/<epsilon>/0/g' |\
  sed 's/;.*;/ /g' |\
  "${FSTPATH}/fstcompile" |\
  "${FSTPATH}/fstarcsort" >"${FINAL_MOD}"-trans.fst

# Cleans up intermediate files created during the process.
rm -rf "${TMPDIR}"
