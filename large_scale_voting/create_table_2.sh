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
killall general_voter_6
sleep 2
cpus=`lscpu | egrep -m1 'CPU\(s\)' | gawk '{print $2}'`
bazel build -c opt --cxxopt='-std=c++17' //:general_voter_6dof_main
rm /tmp/gv_table2_eval_s*
rm gv_table2*
for i in {1..11}; do
echo "case $i"
bazel-bin/general_voter_6dof_main \
  -match_file ./Scene_1_Data61_2D3D/s1_`printf "%03d" ${i}` -table 2 -ext_factor 2 -spatial_min_oct 0.25 -angle_min_oct 0.02 -max_proj_distance 0.03 -use_all_in_verification \
  -debug_file `printf "/tmp/pose_debug_%03d" ${i}` \
  ${1}  > gv_table2_eval_s1_${i}.txt &
sleep 1
while [ `ps -e | grep general_vote | wc -l` -ge ${cpus} ]
do
  sleep 1
done
done
sleep 5
while [ `ps -e | grep general_vote | wc -l` -ge 1 ]
do
  sleep 1
done
./median.sh "gv_table2_eval_s1_*.txt"

