# Hierarchical Visual Foresight (HVF)

This directory contains code for the paper
"Hierarchical Foresight: Self-Supervised Learning of Long-Horizon Tasks via Visual Subgoal Generation" Suraj Nair, Chelsea Finn

## Usage
From the [google_research] directory, run:
```
virtualenc -p python3.6 hvf
source hvf/bin/activate
pip install -r HVF/requirements.txt
```

## Generate Data
Run `python -m HVF.generate_data --savepath=SAVEPATH`

## Train SV2P
Train a video prediction model using the open source tensor2tensor library.
Run `git clone https://github.com/tensorflow/tensor2tensor` and follow instructions
under "Adding a Dataset".

## Train VAE, TDM, TAP
Train the conditional variation autoencoder
`python -m HVF.train_vae --datapath=DATAPATH --savedir=SAVEDIR`

Train the temporal distance cost function
`python -m HVF.train_tdm --datapath=DATAPATH --savedir=SAVEDIR`

Train the time agnostic prediction baseline
`python -m HVF.train_tap --datapath=DATAPATH --savedir=SAVEDIR`

## Run HVF
`python -m HVF.meta_cem --difficulty=m --cost=pixel --numsg=1 --horizon=50 --gt_goals=1 --phorizon=15 --envtype=maze --vaedir=VAEDIR --tdmdir=TDMDIR --tapdir=TAPDIR`
