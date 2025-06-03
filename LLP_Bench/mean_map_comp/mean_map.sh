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


#!/bin/bash
for split_no in $(seq 0 4)
do
  for pair_no in $(seq 0 51)
  do
    python3 mean_map_comp/pismatrix.py --which_split=$split_no --which_pair=$pair_no --which_size=0 --bags_type=feat
  done
done

for split_no in $(seq 0 4)
do
  for pair_no in $(seq 0 51)
  do
    for bag_size in 64 128 256 512
    do
      python3 mean_map_comp/pismatrix.py --which_split=$split_no --which_pair=$pair_no --which_size=$bag_size --bags_type=feat_rand
    done
  done
done

for split_no in $(seq 0 4)
do
  for bag_size in 64 128 256 512
  do
    python3 mean_map_comp/pismatrix.py --which_split=$split_no --which_size=$bag_size --which_pair=0 --bags_type=rand
  done
done

for split_no in $(seq 0 4)
do
  for pair_no in $(seq 0 51)
  do
    for seg_no in $(seq 0 39)
    do
      python3 mean_map_comp/partial_sums.py --which_split=$split_no --which_pair=$pair_no --which_seg=$seg_no --which_size=0 --bags_type=feat
    done
  done
done

list_num_segs=(40 20 10 5)

for split_no in $(seq 0 4)
do
  for pair_no in $(seq 0 51)
  do
    for size in $(seq 0 3)
    do
      num_segs=${list_num_segs[$size]}
      for seg_no in $(seq 0 $num_segs)
      do
        python3 mean_map_comp/partial_sums.py --which_split=$split_no --which_pair=$pair_no --which_seg=$seg_no --which_size=$size --bags_type=feat_rand
      done
    done
  done
done

for split_no in $(seq 0 4)
do
  for bag_size in 64 128 256 512
  do
    for seg_no in $(seq 0 39)
    do
      python3 mean_map_comp/partial_sums.py --which_split=$split_no --which_size=$bag_size --which_seg=$seg_no --which_pair=0 --bags_type=rand
    done
  done
done

for split_no in $(seq 0 4)
do
  for pair_no in $(seq 0 51)
  do
    python3 mean_map_comp/full_vecs.py --which_split=$split_no --which_pair=$pair_no --which_size=0 --bags_type=feat
  done
done

for split_no in $(seq 0 4)
do
  for pair_no in $(seq 0 51)
  do
    for bag_size in 64 128 256 512
    do
      python3 mean_map_comp/full_vecs.py --which_split=$split_no --which_pair=$pair_no --which_size=$bag_size --bags_type=feat_rand
    done
  done
done

for split_no in $(seq 0 4)
do
  for bag_size in 64 128 256 512
  do
    python3 mean_map_comp/full_vecs.py --which_split=$split_no --which_size=$bag_size --which_pair=0 --bags_type=rand
  done
done
