# Neural Collaborative Filtering vs. Matrix Factorization Revisited
Code to reproduce the experiments from the paper:
Rendle, Krichene, Zhang, Anderson (2020): [Neural Collaborative Filtering vs.
Matrix Factorization Revisited](https://arxiv.org/abs/2005.09683)

## Experiment 1: "Revisiting NCF Experiments"

Implementation of a simple matrix factorization model using the datasets and
evaluation protocol from the NCF paper.

Requires the code and datasets from
https://github.com/hexiangnan/neural_collaborative_filtering
(this code assumes a Python 2 runtime)

### Instructions
- Download https://github.com/hexiangnan/neural_collaborative_filtering/archive/master.zip
- Copy mf_simple.py into the same directory

### Final experiments

The results in the plots of Figure 2 were created with the following
hyperparameters.

Movielens:

```
python mf_simple.py --data Data/ml-1m --epochs 256 --embedding_dim 16 \
  --regularization 0.005 --negatives 8 --learning_rate 0.002 --stddev 0.1
```

Pinterest:

```
python mf_simple.py --data Data/pinterest-20 --epochs 256 --embedding_dim 16 \
  --regularization 0.01 --negatives 10 --learning_rate 0.007 --stddev 0.1
```

* We varied the embedding dimension from 16 to 192. Running the larger embedding
  dimension takes the longest but results in the highest quality.
* We repeated each experiment 8 times and report the mean value.
* The code is not optimized for speed but rather for simplicity.

### Hyperparameter tuning
The hyperparameters above were tuned on a holdout set. The holdout set for
hyperparameter tuning can be created with:

```
./create_hold_out.pl --in Data/ml-1m.train.rating \
                     --out_train Data/ml-1m.holdout.train.rating \
                     --out_test Data/ml-1m.holdout.test.rating \
                     --out_test_neg Data/ml-1m.holdout.test.negative
```

More details about the experiments can be found in appendix A.

## Experiment: Learning a Dot Product with MLP

The plots in Figure 3 were created with:

```
python approx_dot.py --embedding_dim {16,32,64,128} \
   --num_users {4000,8000,16000,32000,64000,128000} \
   --num_items {4000,8000,16000,32000,64000,128000} \
   --first_layer_mult {1,2,4} --learning_rate 0.001
```

* The three different plots in Figure 3 correspond to different choices of
first_layer_mult {1,2,4}.
* The y-axis is the number of users {4000,8000,16000,32000,64000,128000}. We set
num_items=num_users.
* We repeated the experiment 5 times and report the mean value.
