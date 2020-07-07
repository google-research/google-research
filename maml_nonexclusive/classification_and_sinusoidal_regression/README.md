This code is a modified version of the code from the MAML codebase from [https://github.com/cbfinn/maml](https://github.com/cbfinn/maml),
which is the code accompaning the paper,
 	[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (Finn et al., ICML 2017)](https://arxiv.org/abs/1703.03400).
This includes code for running MAML and pretraining experiments on classification datasets
 (Omniglot, MiniImagenet and Dclaw) and regression datasets (sinusoidal regression).

### Dependencies
This code requires the following:
* python 2.\* or python 3.\*
* TensorFlow v1.0+

##Commands for running MAML and pretraining (supervised learning on all tasks) experiments

### Train
`python main.py --logdir=logs/log_dir --expt_name=expt_name`

### MAML
`--metatrain_iterations=60000 --pretrain_iterations=0`

### Pretrained baseline
`--metatrain_iterations=0 --pretrain_iterations=60000`

### Test
`--train=False --test_set=True`

### Test with random initialization of the neural network
`--rand_init=True --train=False --test_set=True`

### Miniimagenet
`--datasource=miniimagenet --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --num_filters=32 --max_pool=True`

### Omniglot
`--datasource=omniglot --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1`

### DClaw
`--datasource=dclaw --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=2 --num_filters=32 --max_pool=True --dclaw_pn=1`

### Sinusoidal regression
`--datasource=sinusoid --norm=None --update_batch_size=10 --sine_seed=1 --meta_batch_size=7`

## Task settings
### For Omniglot, Miniimagenet and Dclaw:
Non-mutually-exclusive: `--expt_name=non_exclusive`  
Intrashuffle: `--expt_name=intrashuffle`  
Intershuffle: `--expt_name=intershuffle`  

### For sinusoid:
Non-mutually-exclusive: `--expt_name=non_exclusive`  
Meta-augmentation with uniform noise: `--expt_name=uniform_noise`

## Path to datasets
### Omniglot: 
path: `./data/omniglot_resized` (train, val, test split happens within the code)  
Preprocessing: `./data/omniglot_resized/resize_images.py`
### Miniimagenet:
path: `./data/miniImagenet/train` (train or val or test)    
preprocessing: `./data/miniImagenet/proc_images.py`   
### Dclaw:
path: `./data/dclaw/train` (train or val or test)  
preprocessing: `./data/dclaw/proc_images.py`

### Sinsuoid:
data generated and processed within the code
