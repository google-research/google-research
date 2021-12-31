# Persistent Evolution Strategies

This repository contains code used for the paper "Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies."


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
python toy_regression.py --K=10 --estimate=tbptt
python toy_regression.py --K=10 --estimate=rtrl
python toy_regression.py --K=10 --estimate=uoro
python toy_regression.py --K=10 --estimate=es
python toy_regression.py --K=10 --estimate=pes
```

To visualize the meta-optimization trajectories of each method, first run a fine-grained grid search to use as the background heatmap:
```
python toy_regression_grid.py --T=100
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

Then, run the variance measurement script:
```
python rnn_variance.py
```


### MNIST Hyperparameter Optimization

To run ES and PES:
```
# ES
# --
python meta_opt.py \
    --dataset=mnist \
    --batch_size=100 \
    --shuffle=True \
    --model=mlp \
    --nlayers=2 \
    --nhid=100 \
    --inner_optimizer=sgdm \
    --sgdm_type=0 \
    --objective=train_sum_loss \
    --resample_fixed_minibatch=True \
    --tune_params=lr:inverse-time-decay \
    --outer_optimizer=adam \
    --outer_lr=3e-2 \
    --outer_iterations=50000 \
    --log_every=5000 \
    --print_every=100 \
    --random_hparam_init=False \
    --lr0=0.01 \
    --lr1=10.0 \
    --T=5000 \
    --K=10 \
    --sigma=0.3 \
    --N=100 \
    --estimate=pes \
    --save_dir=saves/mnist

# PES
# ---
python meta_opt.py \
    --dataset=mnist \
    --batch_size=100 \
    --shuffle=True \
    --model=mlp \
    --nlayers=2 \
    --nhid=100 \
    --inner_optimizer=sgdm \
    --sgdm_type=0 \
    --objective=train_sum_loss \
    --resample_fixed_minibatch=True \
    --tune_params=lr:inverse-time-decay \
    --outer_optimizer=adam \
    --outer_lr=3e-2 \
    --outer_iterations=50000 \
    --log_every=5000 \
    --print_every=100 \
    --random_hparam_init=False \
    --lr0=0.01 \
    --lr1=10.0 \
    --T=5000 \
    --K=10 \
    --sigma=0.3 \
    --N=100 \
    --estimate=pes \
    --save_dir=saves/mnist
```

To plot the meta-optimization trajectories, first run a grid to use as the background heatmap:
```
python search.py \
    --search_type=grid \
    --num_points=100 \
    --chunk_size=100 \
    --tune_params=lr:inverse-time-decay \
    --dataset=mnist \
    --model=mlp \
    --nlayers=2 \
    --nhid=100 \
    --T=5000 \
    --K=100 \
    --inner_optimizer=sgdm \
    --sgdm_type=0 \
    --save_dir=saves/mnist_grid
```

Then, plot the visualization:
```
python plot_mnist_lr_decay.py
```


### Control

**Full-Unroll ES**
```
for SEED in 3 5 13 ; do
    srun -p gpu --gres=gpu:0 --mem=12G python control_pes.py \
        --N=10 \
        --estimate=es \
        --divide_by_variance \
        --horizon=1000 \
        --K=1000 \
        --noise=0.1 \
        --lr=0.1 \
        --seed=$SEED \
        --save_dir=es-K1000 &
done
```

**Truncated ES**
```
for SEED in 3 5 13 ; do
    python control_pes.py \
        --N=10 \
        --estimate=es \
        --divide_by_variance \
        --horizon=1000 \
        --K=100 \
        --noise=0.1 \
        --lr=0.1 \
        --seed=$SEED \
        --save_dir=es-K100 &
done
```

**PES**
```
for SEED in 3 5 13 ; do
    python control_pes.py \
        --N=10 \
        --estimate=pes \
        --divide_by_variance \
        --horizon=1000 \
        --K=100 \
        --noise=0.3 \
        --lr=0.1 \
        --seed=$SEED \
        --save_dir=pes-K100 &
done
```

**Plot the results**
```
python plot_control.py
```


## Citation

If you find this repository useful, please cite:

* `Paul Vicol, Luke Metz, Jascha Sohl-Dickstein. Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies, 2021.`

```
@inproceedings{pes-unbiased,
  title={Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies},
  author={Paul Vicol and Luke Metz and Jascha Sohl-Dickstein},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2021}
}
```
