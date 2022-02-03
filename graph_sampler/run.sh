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
set -e

virtualenv -p python3 .
source ./bin/activate

pip install -e graph_sampler

HEAVY_ELEMENTS=C,N,O
NUM_HEAVY=2

mkdir graph_sampler/outputs
cd graph_sampler/outputs
mkdir stoichs weighted uniform

python -m enumerate_stoichiometries --output_prefix=stoichs/ \
  --num_heavy=$NUM_HEAVY --heavy_elements=$HEAVY_ELEMENTS

for stoich_file in $(ls stoichs); do
  prefix=${stoich_file%.*}
  python -m sample_molecules --min_samples=100 \
    --stoich_file=stoichs/$prefix.stoich \
    --out_file=weighted/$prefix.graphml
  python -m reject_to_uniform --in_file=weighted/$prefix.graphml \
    --out_file=uniform/$prefix.graphml
done
python -m stats_to_csv --output=weighted/stats.csv weighted/*.graphml
python -m stats_to_csv --output=uniform/stats.csv uniform/*.graphml

merged_filename="${NUM_HEAVY}_${HEAVY_ELEMENTS}_uniform"
python -m aggregate_uniform_samples --output=${merged_filename}.graphml \
    uniform/*.graphml
python -m graphs_to_smiles ${merged_filename}.graphml > ${merged_filename}.smi
cat ${merged_filename}.smi | uniq -c
