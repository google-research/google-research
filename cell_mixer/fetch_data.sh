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

DIR="$1"
echo "Storing the data to $DIR"

for variant in b_cells memory_t native_t cd56_nk cd14_monocytes cd4_t_helper regulatory_t naive_cytotoxic;do
    dst="$DIR/$variant"
    mkdir -p "$dst"
    wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/$variant/${variant}_filtered_gene_bc_matrices.tar.gz -O - |tar xzv --directory "$dst"
done
