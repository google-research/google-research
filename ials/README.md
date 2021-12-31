# Reproducing iALS benchmark results

This code reproduces the results from the papers:

* Revisiting the Performance of IALS on Item Recommendation Benchmarks
* iALS++: Speeding up Matrix Factorization with Subspace Optimization

## NCF benchmarks

This follows the evaluation protocol and uses the datasets from
[He et al., Neural Collaborative Filtering, WWW '17](https://dl.acm.org/doi/10.1145/3038912.3052569).

Note: This requires the code and datasets from https://github.com/hexiangnan/neural_collaborative_filtering, which requires a Python 2.7 runtime.

### Instructions

1) Navigate to `ncf_benchmarks/` and install packages `pip install -r requirements.txt`

2) Download the NCF code and data.

```
wget https://github.com/hexiangnan/neural_collaborative_filtering/archive/master.zip
unzip master.zip
mv neural_collaborative_filtering-master/* ./
```

3) Run the code. Example usage:

**MovieLens 1M**

```
python ials_simple.py --data Data/ml-1m --epochs 12 --embedding_dim 192 \
    --regularization 0.007 --unobserved_weight 0.3 --stddev 0.1
```

**Pinterest**

```
python ials_simple.py --data Data/pinterest-20 --epochs 16 --embedding_dim 192 \
    --regularization 0.02 --unobserved_weight 0.007 --stddev 0.1
```
This will reproduce the iALS numbers in Table 5 of the paper "Revisiting the
Performance of iALS on Item Recommendation Benchmarks". Note that the table
reports the mean over 10 repetitions of the experiment.


### Hyperparameter tuning

To generate a holdout set for hyper-parameter tuning, we use the script provided
in https://github.com/google-research/google-research/tree/master/dot_vs_learned_similarity

Example usage:

```
./create_hold_out.pl --in Data/ml-1m.train.rating \
                     --out_train Data/ml-1m.holdout.train.rating \
                     --out_test Data/ml-1m.holdout.test.rating \
                     --out_test_neg Data/ml-1m.holdout.test.negative
```


## VAE benchmarks

This follows the evaluation protocol and uses the datasets from
[Liang et al., Variational Autoencoders for Collaborative Filtering, WWW '18](https://dl.acm.org/doi/10.1145/3178876.3186150).

### Instructions

1) Navigate to `vae_benchmarks/` and install packages `pip install -r requirements.txt`

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

This will generate two sub-directories `ml-20m` and `msd` corresponding respectively to the data sets [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) and the [Million Song Data](http://millionsongdataset.com/tasteprofile/).

Note: this code is adapted from https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb which requires a Python 3 runtime.


4) Run the training and evaluation code. Example usage:

**MovieLens 20M (ML20M)**

```
./bin/ialspp_main --train_data ml-20m/train.csv --test_train_data ml-20m/test_tr.csv \
  --test_test_data ml-20m/test_te.csv --embedding_dim 256 --stddev 0.1 \
  --regularization 0.003 --regularization_exp 1.0 --unobserved_weight 0.1 \
  --epochs 16 --block_size 128 --eval_during_training 0
```

**Million Song Data (MSD)**

```
./bin/ialspp_main --train_data msd/train.csv --test_train_data msd/test_tr.csv \
  --test_test_data msd/test_te.csv --embedding_dim 256 --stddev 0.1 \
  --regularization 0.002 --regularization_exp 1.0 --unobserved_weight 0.02 \
  --epochs 16 --block_size 128 --eval_during_training 0
```

Setting the flag `--eval_during_training` to 1 will run evaluation after each epoch.

### Reproducing results from "Revisiting the Performance of iALS on Item Recommendation Benchmarks".

* For Figure 2: Vary the `embedding_dim`. The plots contain values from
  {64, 128, 256, 512, 1024, 2048} for ML20M and for {128, 256, 512, 1024, 2048,
  4096, 8192} for MSD.
* For Table 3 (ML20M): run the commands above with `embedding_dim 2048`.
* For Table 4 (MSD): run the commands above with `embedding_dim 8192`.

All reported iALS results in the paper are averages over 10 runs.

### Reproducing results from "iALS++: Speeding up Matrix Factorization with Subspace Optimization"

Additional to the iALS++ solver (binary ialspp_main), the experiments also
require running the vanilla iALS solver (binary ials_main) and an iCD solver
(binary icd_main). The calls are exactly the same as outlined in step 4, just
switch ./bin/ialspp_main to ./bin/ials_main or ./bin/icd_main. The parameter
block_size will be ignored by both solvers.

* For Figure 1 (top): the results vary embedding_dim from
  {64, 128, 256, 512 1024, 2048}. The results for iALS++ contain three choices
  of block_size: {32, 64, 128}. The training time for one epoch is reported.
* For Figure 1 (bottom): all combinations of embedding_dim
  {64, 128, 256, 512, 1024, 2048} and block_size {2, 4, 8, ..., embedding_dim}.
  The results for block_size=1 are generated by icd_main and the results for
  block_size=embedding_dim by ials_main. The training time for one epoch is
  reported.
* For Figure 2: all combinations of embedding_dim
  {64, 128, 256, 512, 1024, 2048} and block_size {2, 4, 8, ..., embedding_dim}.
  The results for block_size=1 are generated by icd_main and the results for
  block_size=embedding_dim by ials_main. The quality after 16 training epochs is
  reported.
* For Figure 3: the results vary embedding_dim from
  {128, 512, 2048}. The results for iALS++ contain three choices of
  block_size: {32, 64, 128}. The quality after each training epoch is reported.
* For Figure 4: Same setup as Figure 4 but the x-axis is not epochs but training
  time.

### [Optional] Validation of eval code

We also include a 'most-popular' recommender that can be used for testing the
correctness of the evaluation code. Results for this recommender have been
generated by H. Steck in "Embarrassingly Shallow Autoencoders for Sparse Data".
The same recommender can be tested in our code base:

```
./bin/popularity_main --train_data ml-20m/train.csv --test_train_data ml-20m/test_tr.csv \
  --test_test_data ml-20m/test_te.csv
```

and for MSD

```
./bin/popularity_main --train_data msd/train.csv --test_train_data msd/test_tr.csv \
  --test_test_data msd/test_te.csv
```

The produced numbers will match the ones from prior work exactly. The most
popular recommender is deterministic.
