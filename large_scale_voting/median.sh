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

echo "translation error:" `grep error $1 | sort -k2 -n | gawk 'BEGIN{n=0;}{c[n++]=$2}END{if (n % 2) print c[(n-1)/2],n; else print c[n/2-1],n}'`
echo "rotation error:" `grep error $1 | sort -k3 -n | gawk 'BEGIN{n=0;}{c[n++]=$3}END{if (n % 2) print c[(n-1)/2],n; else print c[n/2-1],n}'`
echo "overall time:" `grep "overall_time" $1 | sort -k5 -n | gawk '{c[NR]=$5}END{if (NR % 2) print c[(NR-1)/2],NR; else print c[NR/2-1],NR}'`
