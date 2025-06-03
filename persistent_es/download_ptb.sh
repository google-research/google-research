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
