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
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r numbert/requirements.txt

echo "The Charnel House (Le Charnier) is a c.1944–1948 painting by Spanish \
artist Pablo Picasso (1881–1973), purportedly dealing with the Nazi genocide \
of the Holocaust.
The black and white 'grisaille' composition centres on a massed pile of \
corpses, and was based primarily upon film and photographs of a slaughtered \
family during the Spanish Civil War.
It is considered to be Picasso's second major anti-war painting, the first \
being the monumental Guernica (1937), but it is not thought to be as \
significant as that work because the art \
ist left it unfinished." > numbert/example.txt

python -m numbert.create_pretraining_data --input_file=numbert/example.txt \
--output_file=numbert/output.tfrecord --vocab_file=numbert/model/vocab.txt
