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
rm gv_table3*.txt
a=(18 14 15 8 8 8 1 12 1 13 16 1 16 9 9 9 9 9 17 9 8 23 18 23 22 22)
b=(2 1 1 4 3 1 3 1 1 1 1 2 2 2 3 1 5 6 1 4 2 1 1 2 1 2)
for i in $(seq 0 1 25); do
echo "case ${a[i]} ${b[i]}"
bazel-bin/general_voter_6dof_main \
  -match_file ./area3_2D-3D-S/s${a[i]}_`printf "%03d" ${b[i]}` -table 3 \
  -debug_file `printf "/tmp/pose_debug_%03d_%03d" ${a[i]} ${b[i]}` -ext_factor 0.5 -spatial_min_oct 0.25 -angle_min_oct 0.11 -max_proj_distance 0.14 -max_time_sec 180 ${1} > gv_table3_eval_s${a[i]}_${b[i]}.txt &
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
./median.sh "gv_table3*" 

