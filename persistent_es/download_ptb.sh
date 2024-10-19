# Copyright 2024 The Google Research Authors.
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

# This script is copied from https://github.com/salesforce/awd-lstm-lm/blob/master/getdata.sh
echo "- Downloading Penn Treebank (PTB)"
wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz

mkdir -p data/pennchar
mv simple-examples/data/ptb.char.train.txt data/pennchar/train.txt
mv simple-examples/data/ptb.char.test.txt data/pennchar/test.txt
mv simple-examples/data/ptb.char.valid.txt data/pennchar/valid.txt

rm -rf simple-examples/
rm -rf simple-examples.tgz

echo "Happy language modeling on PTB :)"
