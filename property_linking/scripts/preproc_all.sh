# Copyright 2019 The Google Research Authors.
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
python3 fbjson_to_wdtsv.py dev500.json mid_to_wd.tsv rels_fb_to_wd.tsv yago_s_cats.tsv yago_s_cats_dev_0.tsv
python ../scripts/overlap.py yago_s_cats_dev_0.tsv ../property_linking/yago_s_names.tsv yago_s_cats_dev_1.tsv
python ../scripts/add_scored_links.py ../soft_matches/yago_matches.txt yago_s_cats_dev_1.tsv yago_s_cats_dev_2.tsv
mv yago_s_cats_dev_2.tsv ../property_linking/yago_s_cats_dev.tsv
sed -n -e '101,500p' ../property_linking/yago_s_cats_dev.tsv > ../property_linking/yago_s_cats_dev_1.tsv
sed -n -e '1,100p' ../property_linking/yago_s_cats_dev.tsv > ../property_linking/yago_s_cats_dev_1_test.tsv
sed -n -e '1,100p' -e '201,500p' ../property_linking/yago_s_cats_dev.tsv > ../property_linking/yago_s_cats_dev_2.tsv
sed -n -e '101,200p' ../property_linking/yago_s_cats_dev.tsv > ../property_linking/yago_s_cats_dev_2_test.tsv
sed -n -e '1,200p' -e '301,500p' ../property_linking/yago_s_cats_dev.tsv > ../property_linking/yago_s_cats_dev_3.tsv
sed -n -e '201,300p' ../property_linking/yago_s_cats_dev.tsv > ../property_linking/yago_s_cats_dev_3_test.tsv
sed -n -e '1,300p' -e '401,500p' ../property_linking/yago_s_cats_dev.tsv > ../property_linking/yago_s_cats_dev_4.tsv
sed -n -e '301,400p' ../property_linking/yago_s_cats_dev.tsv > ../property_linking/yago_s_cats_dev_4_test.tsv
sed -n -e '1,400p' ../property_linking/yago_s_cats_dev.tsv > ../property_linking/yago_s_cats_dev_5.tsv
sed -n -e '401,500p' ../property_linking/yago_s_cats_dev.tsv > ../property_linking/yago_s_cats_dev_5_test.tsv
