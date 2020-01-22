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

#!/bin/bash

mosesdecoder=/mount/mosesdecoder
tok_gold_targets=newstest2014.tok.de

decodes_file=$1

# Replace unicode.
perl $mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl -l de  < $decodes_file > $decodes_file.n

# Tokenize.
perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < $decodes_file.n > $decodes_file.tok

# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $tok_gold_targets > $tok_gold_targets.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_file.tok > $decodes_file.tok.atat

# Get BLEU.
perl $mosesdecoder/scripts/generic/multi-bleu.perl $tok_gold_targets.atat < $decodes_file.tok.atat
