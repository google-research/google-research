# Persistent Evolution Strategies

This repository contains code used for the paper [Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies](https://arxiv.org/abs/2112.13835).


## Requirements

* JAX
* Haiku
* ruamel.yaml
* tensorflow_datasets


## Experiments

### Influence Balancing

Run the influence balancing experiment with different gradient estimates {TBPTT, RTRL, UORO, ES, PES}:
```
python influence_balancing.py --K=1 --estimate=tbptt
python influence_balancing.py --K=10 --estimate=tbptt
python influence_balancing.py --K=100 --estimate=tbptt

python influence_balancing.py --K=1 --estimate=es
python influence_balancing.py --K=1 --estimate=pes

python influence_balancing.py --K=1 --lr=1e-5 --estimate=uoro
python influence_balancing.py --K=1 --estimate=rtrl
```

Then plot the resulting loss curves:
```
python plot_influence_balancing.py
```


### Toy 2D Regression Task

The following commands run the toy 2D regression task with different gradient estimates {TBPTT, RTRL, UORO, ES, PES}:
```
python toy_regression.py --estimate=tbptt
python toy_regression.py --estimate=rtrl
python toy_regression.py --estimate=uoro
python toy_regression.py --estimate=es --sigma=1.0
python toy_regression.py --estimate=pes
```

To visualize the meta-optimization trajectories of each method, first run a fine-grained grid search to use as the background heatmap:
```
python toy_regression_grid.py
```

After running all the training commands above, as well as the grid command, plot the visualization:
```
python plot_toy_regression.py
```

### Variance Measurement with a Toy LSTM

To reproduce the experiment measuring the empirical variance of PES with a toy LSTM on a subset of the Penn TreeBank data, first download the PTB dataset (this will create the directory `data/pennchar`):
```
./download_ptb.sh
```

Run the variance measurement script:
```
python rnn_variance.py --scenario=real
python rnn_variance.py --scenario=random
python rnn_variance.py --scenario=repeat
```

Then, plot the variance curves:
```
python plot_variance_combined.py
```


### MNIST Hyperparameter Optimization

#### Meta-Objective: Training Loss

**ES**
```
# K=10
python hyperopt.py \
    --dataset=mnist \
    --model=mlp \
    --objective=train_sum_loss \
    --tune_params=lr:inverse-time-decay \
    --outer_optimizer=adam \
    --outer_lr=1e-2 \
    --outer_iterations=5000 \
    --log_every=2 \
    --print_every=10 \
    --eval_every=500 \
    --lr0=0.003 \
    --lr1=30.0 \
    --T=5000 \
    --K=10 \
    --sigma=0.1 \
    --n_chunks=1 \
    --n_per_chunk=1000 \
    --estimate=es \
    --save_dir=saves/mnist_lr_decay

# K=100
python hyperopt.py \
    --dataset=mnist \
    --model=mlp \
    --objective=train_sum_loss \
    --tune_params=lr:inverse-time-decay \
    --outer_optimizer=adam \
    --outer_lr=1e-2 \
    --outer_iterations=5000 \
    --log_every=2 \
    --print_every=10 \
    --eval_every=500 \
    --lr0=0.003 \
    --lr1=30.0 \
    --T=5000 \
    --K=100 \
    --sigma=0.1 \
    --n_chunks=1 \
    --n_per_chunk=1000 \
    --estimate=es \
    --save_dir=saves/mnist_lr_decay
```

**PES**
```
# K=10
python hyperopt.py \
    --dataset=mnist \
    --model=mlp \
    --objective=train_sum_loss \
    --tune_params=lr:inverse-time-decay \
    --outer_optimizer=adam \
    --outer_lr=1e-2 \
    --outer_iterations=5000 \
    --log_every=2 \
    --print_every=10 \
    --eval_every=500 \
    --lr0=0.003 \
    --lr1=30.0 \
    --T=5000 \
    --K=10 \
    --sigma=0.1 \
    --n_chunks=1 \
    --n_per_chunk=1000 \
    --estimate=pes \
    --save_dir=saves/mnist_lr_decay

# K=100
python hyperopt.py \
    --dataset=mnist \
    --model=mlp \
    --objective=train_sum_loss \
    --tune_params=lr:inverse-time-decay \
    --outer_optimizer=adam \
    --outer_lr=1e-2 \
    --outer_iterations=5000 \
    --log_every=2 \
    --print_every=10 \
    --eval_every=500 \
    --lr0=0.003 \
    --lr1=30.0 \
    --T=5000 \
    --K=100 \
    --sigma=0.1 \
    --n_chunks=1 \
    --n_per_chunk=1000 \
    --estimate=pes \
    --save_dir=saves/mnist_lr_decay
```

#### Meta-Objective: Validation Accuracy

**ES**
```
# K=10
python hyperopt.py \
    --dataset=mnist \
    --model=mlp \
    --objective=val_sum_acc \
    --tune_params=lr:inverse-time-decay \
    --outer_optimizer=adam \
    --outer_lr=1e-2 \
    --outer_iterations=5000 \
    --log_every=2 \
    --print_every=10 \
    --eval_every=500 \
    --lr0=0.003 \
    --lr1=30.0 \
    --T=5000 \
    --K=10 \
    --sigma=0.1 \
    --n_chunks=1 \
    --n_per_chunk=1000 \
    --estimate=es \
    --save_dir=saves/mnist_lr_decay_val_acc

# K=100
python hyperopt.py \
    --dataset=mnist \
    --model=mlp \
    --objective=val_sum_acc \
    --tune_params=lr:inverse-time-decay \
    --outer_optimizer=adam \
    --outer_lr=1e-2 \
    --outer_iterations=5000 \
    --log_every=2 \
    --print_every=10 \
    --eval_every=500 \
    --lr0=0.003 \
    --lr1=30.0 \
    --T=5000 \
    --K=100 \
    --sigma=0.1 \
    --n_chunks=1 \
    --n_per_chunk=1000 \
    --estimate=es \
    --save_dir=saves/mnist_lr_decay_val_acc
```

**PES**
```
# K=10
python hyperopt.py \
    --dataset=mnist \
    --model=mlp \
    --objective=val_sum_acc \
    --tune_params=lr:inverse-time-decay \
    --outer_optimizer=adam \
    --outer_lr=1e-2 \
    --outer_iterations=5000 \
    --log_every=2 \
    --print_every=10 \
    --eval_every=500 \
    --lr0=0.003 \
    --lr1=30.0 \
    --T=5000 \
    --K=10 \
    --sigma=0.1 \
    --n_chunks=1 \
    --n_per_chunk=1000 \
    --estimate=pes \
    --save_dir=saves/mnist_lr_decay_val_acc

# K=100
python hyperopt.py \
    --dataset=mnist \
    --model=mlp \
    --objective=val_sum_acc \
    --tune_params=lr:inverse-time-decay \
    --outer_optimizer=adam \
    --outer_lr=1e-2 \
    --outer_iterations=5000 \
    --log_every=2 \
    --print_every=10 \
    --eval_every=500 \
    --lr0=0.003 \
    --lr1=30.0 \
    --T=5000 \
    --K=100 \
    --sigma=0.1 \
    --n_chunks=1 \
    --n_per_chunk=1000 \
    --estimate=pes \
    --save_dir=saves/mnist_lr_decay_val_acc
```


To plot the meta-optimization trajectories, first run a grid to use as the background heatmap:
```
python search.py \
    --search_type=grid \
    --num_points=40 \
    --dataset=mnist \
    --model=mlp \
    --inner_optimizer=sgdm \
    --tune_params=lr:inverse-time-decay \
    --T=5000 \
    --K=100 \
    --save_dir=saves/mnist_grid_sgdm
```

Then, plot the visualization:
```
python plot_mnist_lr_decay.py
```


### UCI Regression

These commands must be run within the `hyperopt` directory.

**ES**
```
for INIT_THETA in 5 3 1 -1 -3 -5 ; do
    python uci.py \
        --estimate=es \
        --K=1 \
        --outer_lr=0.003 \
        --lr=0.001 \
        --init_theta=$INIT_THETA \
        --save_dir=saves_uci &
done
```

**PES**
```
for INIT_THETA in 5 3 1 -1 -3 -5 ; do
    python uci.py \
        --estimate=pes \
        --K=1 \
        --outer_lr=0.003 \
        --lr=0.001 \
        --init_theta=$INIT_THETA \
        --save_dir=saves_uci &
done
```

**Plot the results**
```
python plot_uci.py
```


### Tuning Several Continuous and Discrete Hyperparameters

**Random Search**
```
for SEED in 3 5 7 11 ; do
    python search.py \
        --dataset=fashion_mnist \
        --objective=val_sum_loss \
        --model=mlp \
        --nlayers=5 \
        --search_type=random \
        --num_points=10000 \
        --chunk_size=1 \
        --inner_optimizer=sgdm \
        --tune_params=mask:fixed,lr:fixed-pl,mom:fixed-pl \
        --T=1000 \
        --K=100 \
        --num_eval_runs=10 \
        --seed=$SEED \
        --save_dir=saves/many_hparams/random &
done
```

**ES**
```
for SEED in 3 5 7 11 ; do
    python hyperopt.py \
        --dataset=fashion_mnist \
        --batch_size=100 \
        --model=mlp \
        --nlayers=5 \
        --inner_optimizer=sgdm \
        --objective=val_sum_loss \
        --tune_params=mask:fixed,lr:fixed-pl,mom:fixed-pl \
        --outer_optimizer=adam \
        --outer_lr=1e-2 \
        --outer_iterations=50000 \
        --log_every=2 \
        --print_every=10 \
        --eval_every=100 \
        --T=1000 \
        --K=10 \
        --sigma=0.3 \
        --n_chunks=1 \
        --n_per_chunk=10 \
        --random_hparam_init \
        --estimate=es \
        --num_eval_runs=10 \
        --seed=$SEED \
        --save_dir=saves/many_hparams/es &
done
```

**PES K=10**
```
for SEED in 3 5 7 11 ; do
    python hyperopt.py \
        --dataset=fashion_mnist \
        --batch_size=100 \
        --model=mlp \
        --nlayers=5 \
        --inner_optimizer=sgdm \
        --objective=val_sum_loss \
        --tune_params=mask:fixed,lr:fixed-pl,mom:fixed-pl \
        --outer_optimizer=adam \
        --outer_lr=1e-2 \
        --outer_iterations=50000 \
        --log_every=2 \
        --print_every=10 \
        --eval_every=100 \
        --T=1000 \
        --K=10 \
        --sigma=0.3 \
        --n_chunks=1 \
        --n_per_chunk=10 \
        --random_hparam_init \
        --estimate=pes \
        --num_eval_runs=10 \
        --seed=$SEED \
        --save_dir=saves/many_hparams/pes &
done
```

**Plot the Results**
```
python plot_hyperopt_comparison.py
```


### Control

To run the control experiments, you need to have Gym installed (experiments were performed with gym version 0.12.5).

**Full-Unroll ES**
```
for SEED in 3 5 7 11 13 23 ; do
    python control_pes.py \
        --save_dir=saves/control/es-K1000 \
        --N=10 \
        --estimate=es \
        --divide_by_variance \
        --horizon=1000 \
        --K=1000 \
        --noise=0.3 \
        --lr=0.3 \
        --seed=$SEED &
done
```

**Truncated ES**
```
for SEED in 3 5 7 11 13 23 ; do
    python control_pes.py \
        --save_dir=saves/control/es-K100 \
        --N=10 \
        --estimate=es \
        --divide_by_variance \
        --horizon=1000 \
        --K=100 \
        --noise=0.1 \
        --lr=0.1 \
        --seed=$SEED &
done
```

**PES**
```
for SEED in 3 5 7 11 13 23 ; do
    python control_pes.py \
        --save_dir=saves/control/pes-K100 \
        --N=10 \
        --estimate=pes \
        --divide_by_variance \
        --horizon=1000 \
        --K=100 \
        --noise=0.3 \
        --lr=0.1 \
        --seed=$SEED &
done
```

**Plot the results**
```
python plot_control.py
```


## Tips

If you encounter any out-of-memory issues, you can try setting the following environment variable (based on https://github.com/google/jax/issues/788):
```
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```


## Citation

If you find this repository useful, please cite:

* `Paul Vicol, Luke Metz, Jascha Sohl-Dickstein. Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies, 2021.`

```
@inproceedings{vicol2021unbiased,
  title={{Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies}},
  author={Vicol, Paul and Metz, Luke and Sohl-Dickstein, Jascha},
  booktitle={International Conference on Machine Learning},
  pages={10553--10563},
  year={2021},
  organization={PMLR}
}
```
