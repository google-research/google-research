# Efficient Optimization of Sparse User Encoder Recommenders

This code reproduces the results from the paper [Efficient Optimization of
Sparse User Encoder Recommenders](https://dl.acm.org/doi/10.1145/3651170).

## Instructions

1) Install packages `pip install -r requirements.txt`

2) Compile the code

- Download [Eigen](https://eigen.tuxfamily.org/):

```
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.zip
unzip eigen-3.3.9.zip
```
- Create the subdirectories

```
mkdir lib
mkdir bin
```
- Compile the binaries

```
make all
```

3) Download and process the data

```
python generate_data.py --output_dir ./
```

This will generate two sub-directories `ml-20m` and `msd` corresponding
respectively to the data sets
[MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) and the
[Million Song Data](http://millionsongdataset.com/tasteprofile/).

The evaluation follows the protocol from
[Liang et al., Variational Autoencoders for Collaborative Filtering, WWW '18](https://dl.acm.org/doi/10.1145/3178876.3186150).
The script generate_data.py was adapted from
https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb.


## Training and Evaluation

The binary bin/sue_main can reproduce the experimental results from the paper
"Efficient Optimization of Sparse User Encoder Recommenders". The
hyperparameters for each experiment can be found in Appendix B of the paper.

### Identity Encoder

```
./bin/sue_main --train_data ml-20m/train.csv \
  --test_train_data ml-20m/test_tr.csv --test_test_data ml-20m/test_te.csv \
  --encoder identity --regularization 500
```

### Encoder with Crosses

```
./bin/sue_main --train_data ml-20m/train.csv \
  --test_train_data ml-20m/test_tr.csv --test_test_data ml-20m/test_te.csv \
  --encoder crosses --regularization 500,5000 --num_features 20108,10000
```
The flag num_features specifies how many items and pairwise crosses are used.
This dataset has 20108 items, so this example uses all items and 10000
additional pairs. Pairs and items are selected by frequency. The
regularization flag has one entry for items (here 500) and one for pairs (here
5000).

### Hashed Encoder

```
./bin/sue_main --train_data ml-20m/train.csv \
  --test_train_data ml-20m/test_tr.csv --test_test_data ml-20m/test_te.csv \
  --encoder hashing --regularization 512 --num_buckets 4096 \
  --num_hash_functions 8
```

### Encoder with Features

```
./bin/sue_main --train_data ml-20m/train.csv \
  --test_train_data ml-20m/test_tr.csv --test_test_data ml-20m/test_te.csv \
  --encoder features --regularization 512 \
  --features genres.csv
```

### Restricting Labels

All encoders support restricting the labels during training. For example:

```
./bin/sue_main --train_data ml-20m/train.csv \
  --test_train_data ml-20m/test_tr.csv --test_test_data ml-20m/test_te.csv \
  --encoder identity --regularization 512 --filter_labels 4096
```
This example trains an identity encoder where only the 4096 most frequent items
can be predicted.


### Input Drop-Out

All encoders support input dropout. For example:

```
./bin/sue_main --train_data ml-20m/train.csv \
  --test_train_data ml-20m/test_tr.csv --test_test_data ml-20m/test_te.csv \
  --encoder identity --regularization 500 \
  --dropout_keep_prob 0.8 --dropout_num_cases 32
```

### Frequency Regularization

All encoders support frequency regularization. For example:

```
./bin/sue_main --train_data ml-20m/train.csv \
  --test_train_data ml-20m/test_tr.csv --test_test_data ml-20m/test_te.csv \
  --encoder crosses --regularization 512,4096 --num_features 20180,8192 \
  --frequency_regularization 0.125
```

