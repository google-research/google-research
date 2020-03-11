# Hierarchical Visual Foresight ([HVF](https://sites.google.com/stanford.edu/hvf))

This directory contains code for the paper
["Hierarchical Foresight: Self-Supervised Learning of Long-Horizon Tasks via Visual Subgoal Generation"](https://arxiv.org/abs/1909.05829). 
Suraj Nair, Chelsea Finn. ICLR 2020.

For the most recent version of the code, please see the up to date repo [here](https://github.com/suraj-nair-1/google-research/tree/master/hierarchical_foresight)

## Setup

This code base uses Python 3.5 and Tensorflow 1.14.
From the google_research directory, run:

```
python3 -m venv hvf
source hvf/bin/activate
pip install -r hierarchical_foresight/requirements.txt
```

You will also need to clone the open source [tensor2tensor](https://github.com/tensorflow/tensor2tenso) library to run video prediction. Specififcally you will need to clone `tensor2tensor==1.13.4` and follow instructions under "Adding a Dataset".

## Generate Data
Run `python -m hierarchical_foresight.generate_data --savepath=SAVEPATH`

## Train [SV2P](https://arxiv.org/abs/1710.11252)
Train a video prediction model using the open source tensor2tensor library.
Run `git clone https://github.com/tensorflow/tensor2tensor` and follow instructions
under "Adding a Dataset".

Once you have a trained model on your problem, modify `hierarchical_foresight/env/subgoal_env.py` to use your model/problem. 

## Train VAE
Train the conditional variational autoencoder
`python -m hierarchical_foresight.train_vae --beta=0.1 --datapath=DATAPATH --savedir=SAVEDIR`

## Run HVF

Run trials on new tasks using HVF by running
`python -m hierarchical_foresight.meta_cem --difficulty=h --cost=pixel --numsg=1 --horizon=50 --gt_goals=1 --phorizon=5 --envtype=maze --vaedir=VAEDIR`.

This will run 100 trials with randomly generated tasks of the specified difficulty, and report the success rate, while also logging the identified subgoals. Using the default planning parameters each trial takes ~5 minutes, and can be adjusted to search over more or less samples by setting the hyperparams `metacem_samples, metacem_iters, cem_samples, cem_iters`.

## Train TDM, TAP (Optional)

Train the temporal distance cost function
`python -m hierarchical_foresight.train_tdm --datapath=DATAPATH --savedir=SAVEDIR`

Train the time agnostic prediction baseline
`python -m hierarchical_foresight.train_tap --datapath=DATAPATH --savedir=SAVEDIR`

Then you can run HVF with 
`python -m hierarchical_foresight.meta_cem --difficulty=h --cost=pixel --numsg=1 --horizon=50 --gt_goals=1 --phorizon=5 --envtype=maze --vaedir=VAEDIR --tdmdir=TDMDIR --tapdir=TAPDIR`


