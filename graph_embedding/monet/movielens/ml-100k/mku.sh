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

#!/bin/sh

trap `rm -f tmp.$$; exit 1` 1 2 15

for i in 1 2 3 4 5
do
	head -`expr $i \* 20000` u.data | tail -20000 > tmp.$$
	sort -t"	" -k 1,1n -k 2,2n tmp.$$ > u$i.test
	head -`expr \( $i - 1 \) \* 20000` u.data > tmp.$$
	tail -`expr \( 5 - $i \) \* 20000` u.data >> tmp.$$
	sort -t"	" -k 1,1n -k 2,2n tmp.$$ > u$i.base
done

allbut.pl ua 1 10 100000 u.data
sort -t"	" -k 1,1n -k 2,2n ua.base > tmp.$$
mv tmp.$$ ua.base
sort -t"	" -k 1,1n -k 2,2n ua.test > tmp.$$
mv tmp.$$ ua.test

allbut.pl ub 11 20 100000 u.data
sort -t"	" -k 1,1n -k 2,2n ub.base > tmp.$$
mv tmp.$$ ub.base
sort -t"	" -k 1,1n -k 2,2n ub.test > tmp.$$
mv tmp.$$ ub.test

