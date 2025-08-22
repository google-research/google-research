#!/bin/bash
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


set -e

DATAFILE=${1:-../data/performance.tsv}
TAG=${2:-unknown}
# Max test perplexity to show for n-grams
MAXNGRAMPERPLEX=${3:-300}

# Algorithm keeps the 1st header row
egrep '(Algorithm|NGram).*rain' $DATAFILE >$TAG-ngram-train.tsv
egrep '(Algorithm|KneyserNey).*rain' $DATAFILE >$TAG-kn-train.tsv
egrep '(Algorithm|Initial).*rain' $DATAFILE >$TAG-initial-train.tsv

egrep '(Algorithm|NGram).*est' $DATAFILE >$TAG-ngram-test.tsv
egrep '(Algorithm|KneyserNey).*est' $DATAFILE >$TAG-kn-test.tsv
egrep '(Algorithm|Initial).*est' $DATAFILE >$TAG-initial-test.tsv

# Filter intermediate results
egrep '(Algorithm|LAMP.*final).*rain' $DATAFILE >$TAG-lamp-train.tsv
egrep '(Algorithm|LAMP.*final).*est' $DATAFILE >$TAG-lamp-test.tsv

# LAMP optimizing the weights only
egrep '(Algorithm|LAMP.*first).*rain' $DATAFILE >$TAG-lamp-weights-only-train.tsv
egrep '(Algorithm|LAMP.*first).*est' $DATAFILE >$TAG-lamp-weights-only-test.tsv

for order in `seq 2 10`; do
  # LAMP progress broken down by iteration, order is 2nd column
  egrep "(Algorithm|LAMP	$order	).*rain" $DATAFILE >$TAG-lamp-order-$order-iter-train.tsv
  egrep "(Algorithm|LAMP	$order	).*est" $DATAFILE >$TAG-lamp-order-$order-iter-test.tsv
  # Weight curves
  # Find the single line with the weights and stdev, transpose the row vectors, and rejoin them.
  egrep "LAMP	$order	" $TAG-lamp-train.tsv | cut -f 14 | tr -d '[] ' | tr ',' '\n' >weights
  egrep "LAMP	$order	" $TAG-lamp-train.tsv | cut -f 15 | tr -d '[] ' | tr ',' '\n' >weights_stdev
  echo 'Weight	WeightStDev' >$TAG-lamp-order-$order-weights.tsv
  paste weights weights_stdev >>$TAG-lamp-order-$order-weights.tsv
  # The same for the LAMP that optimizes the weights only
  egrep "LAMP	$order	" $TAG-lamp-weights-only-train.tsv | cut -f 14 | tr -d '[] ' | tr ',' '\n' >weights
  egrep "LAMP	$order	" $TAG-lamp-weights-only-train.tsv | cut -f 15 | tr -d '[] ' | tr ',' '\n' >weights_stdev
  echo 'Weight	WeightStDev' >$TAG-lamp-weights-only-order-$order-weights.tsv
  paste weights weights_stdev >>$TAG-lamp-weights-only-order-$order-weights.tsv
done

mkdir -p ../figs
sed "s/TAG/$TAG/g" plotall.gnu | sed "s/MAXNGRAMPERPLEX/$MAXNGRAMPERPLEX/g" | gnuplot
