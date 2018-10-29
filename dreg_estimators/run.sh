#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install tensorflow
pip install tensorflow-probability
pip install -r dreg_estimators/requirements.txt

MNIST_LOCATION="/tmp/mnist"
rm -rf $MNIST_LOCATION
mkdir $MNIST_LOCATION
curl -c /tmp/cookie -s -L "https://drive.google.com/uc?export=download&id=1BaEWtwo3SQ8m7_Xs9VpTEPX10zpdbklX"
curl -o "$MNIST_LOCATION/train_xs.npy" -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' /tmp/cookie`&id=1BaEWtwo3SQ8m7_Xs9VpTEPX10zpdbklX"
wget -O $MNIST_LOCATION/valid_xs.npy "https://drive.google.com/uc?id=1Z4ItIhpUMXF_NIx3k_14pCTMyeQV8v69&export=download"
wget -O $MNIST_LOCATION/test_xs.npy "https://drive.google.com/uc?id=1OsyM_2tlZOoPGHYM7tQs8KTpAxSo5bYq&export=download"

OMNIGLOT_LOCATION="/tmp/omniglot.mat"
wget -O $OMNIGLOT_LOCATION "https://drive.google.com/uc?export=download&id=1ZgNzUjHskBbwZd4so0VxILkVFMCm0hbg"

python -m dreg_estimators.main_loop --max_steps 2000 \
          --MNIST_LOCATION $MNIST_LOCATION --OMNIGLOT_LOCATION $OMNIGLOT_LOCATION
