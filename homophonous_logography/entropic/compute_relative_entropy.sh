# Copyright 2021 The Google Research Authors.
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

# Obtain estimates of various information-theoretic measures based on entropy
# of n-grams. The first argument specifies the original training data in the
# format required for training the neural measures. This is a tab-separated text
# file. The second argument is the language name. The temporary files are
# created in the `/tmp` directory.
#
# Bazel build system should be installed locally for this tool to work.
#
# Examples:
# --------
# ./compute_relative_entropy.sh ${DATA_DIR}/korean-jamo.tsv Korean
# ./compute_relative_entropy.sh ${DATA_DIR}/japanese.tsv Japanese

set -euo pipefail

# N-gram order.
ORDER=2

# Language name.
LANGUAGE="$2"

# Check for Bazel installation.
which bazel > /dev/null 2>&1
if [ $? -ne 0 ] ; then
  echo "Please install Bazel to run this tool!"
  exit 1
fi

# Split the corpus into training/test written/pronounced/joint.
bazel build -c opt :split_corpus_main
bazel-bin/split_corpus_main experimental/nlp/sweet/logo/entropy/split_corpus.py --corpus="$1"

# Extract symbol table.
bazel build -c opt :ngramsymbols
NGRAM_SYMBOLS_TOOL=bazel-bin/external/org_opengrm_ngram/ngramsymbols
cat /tmp/wtrain.txt /tmp/wtest.txt > /tmp/xxx
${NGRAM_SYMBOLS_TOOL} \
  /tmp/xxx /tmp/written.syms
cat /tmp/ptrain.txt /tmp/ptest.txt > /tmp/xxx
${NGRAM_SYMBOLS_TOOL} \
  /tmp/xxx /tmp/phoneme.syms
cat /tmp/jtrain.txt /tmp/jtest.txt > /tmp/xxx
${NGRAM_SYMBOLS_TOOL} \
  /tmp/xxx /tmp/joint.syms

# Compile FST archives (FARs).
bazel build -c opt :farcompilestrings
FAR_COMPILE_STRINGS_TOOL=bazel-bin/external/org_openfst/farcompilestrings
${FAR_COMPILE_STRINGS_TOOL} \
  --fst_type=compact \
  --symbols=/tmp/written.syms \
  --keep_symbols \
  /tmp/wtrain.txt /tmp/wtrain.far
${FAR_COMPILE_STRINGS_TOOL} \
  --fst_type=compact \
  --symbols=/tmp/written.syms \
  --keep_symbols \
  /tmp/wtest.txt /tmp/wtest.far
${FAR_COMPILE_STRINGS_TOOL} \
  --fst_type=compact \
  --symbols=/tmp/phoneme.syms \
  --keep_symbols \
  /tmp/ptrain.txt /tmp/ptrain.far
${FAR_COMPILE_STRINGS_TOOL} \
  --fst_type=compact \
  --symbols=/tmp/phoneme.syms \
  --keep_symbols \
  /tmp/ptest.txt /tmp/ptest.far
${FAR_COMPILE_STRINGS_TOOL} \
  --fst_type=compact \
  --symbols=/tmp/joint.syms \
  --keep_symbols \
  /tmp/jtrain.txt /tmp/jtrain.far

# Accumulate n-gram counts.
bazel build -c opt :ngramcount
NGRAM_COUNT_TOOL=bazel-bin/external/org_opengrm_ngram/ngramcount
${NGRAM_COUNT_TOOL} \
  --order="${ORDER}" \
  /tmp/wtrain.far /tmp/wtrain.cnts
${NGRAM_COUNT_TOOL} \
  --order="${ORDER}" \
  /tmp/ptrain.far /tmp/ptrain.cnts
${NGRAM_COUNT_TOOL} \
  --order="${ORDER}" \
  /tmp/jtrain.far /tmp/jtrain.cnts

# Build n-gram models.
bazel build -c opt :ngrammake
NGRAM_MAKE_TOOL=bazel-bin/external/org_opengrm_ngram/ngrammake
${NGRAM_MAKE_TOOL} \
  /tmp/wtrain.cnts /tmp/wtrain.mod
${NGRAM_MAKE_TOOL} \
  /tmp/ptrain.cnts /tmp/ptrain.mod
${NGRAM_MAKE_TOOL} \
  /tmp/jtrain.cnts /tmp/jtrain.mod

# Compute perplexities.
bazel build -c opt :ngramperplexity
NGRAM_PERPLEXITY_TOOL=bazel-bin/external/org_opengrm_ngram/ngramperplexity
${NGRAM_PERPLEXITY_TOOL} \
  /tmp/wtrain.mod /tmp/wtest.far /tmp/wtest.perp
${NGRAM_PERPLEXITY_TOOL} \
  /tmp/ptrain.mod /tmp/ptest.far /tmp/ptest.perp

# Compute entropy difference/ratio.
bazel build -c opt :entropy_difference_main
bazel-bin/entropy_difference_main \
  --corpus=$1 \
  --wperp=/tmp/wtest.perp \
  --pperp=/tmp/ptest.perp

# Print models.
bazel build -c opt :ngramprint
NGRAM_PRINT_TOOL=bazel-bin/external/org_opengrm_ngram/ngramprint
${NGRAM_PRINT_TOOL} \
  /tmp/ptrain.mod /tmp/ptrain.mod.txt
${NGRAM_PRINT_TOOL} \
  /tmp/wtrain.mod /tmp/wtrain.mod.txt
${NGRAM_PRINT_TOOL} \
  /tmp/jtrain.mod /tmp/jtrain.mod.txt

# Compute (cross-)entropies/KL divergences and mutual information (MI) measures.
# bazel build -c opt :ngramcrossentropy
bazel build -c opt :ngramcrossentropy
bazel-bin/ngramcrossentropy \
  --info_header="["${LANGUAGE}"]:" \
  --ngram_joint_fst=/tmp/jtrain.mod \
  --ngram_source_fst=/tmp/ptrain.mod \
  --ngram_destination_fst=/tmp/wtrain.mod \
  --source_samples_far=/tmp/ptest.far \
  --destination_samples_far=/tmp/wtest.far
