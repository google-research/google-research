# Koopman Neural Forecaster

Rui Wang, Yihe Dong, Sercan O. Arik, Rose Yu; [Koopman Neural  Forecaster for  Time Series with Temporal Distribution Shifts](https://arxiv.org/abs/2210.03675)

## Abstract
Temporal distributional shifts, with underlying dynamics changing over time, frequently occur in real-world time series, and pose a fundamental challenge for deep neural networks (DNNs). In this paper, we propose a novel deep sequence model based on the Koopman theory  for time series forecasting: Koopman Neural Forecaster (KNF) that leverages DNNs to learn the linear Koopman space and the coefficients of chosen measurement functions. KNF imposes appropriate inductive biases for improved robustness against distributional shifts, employing both a global operator to learn shared characteristics, and a local operator to capture changing dynamics, as well as a specially-designed feedback loop to continuously update the learnt operators over time for rapidly varying behaviors. To the best of our knowledge, this is the first time that Koopman theory is applied to real-world chaotic time series without known governing laws. We demonstrate that KNF achieves the superior performance compared to the alternatives, on multiple time series datasets that are shown to suffer from distribution shifts.

## Requirements
To install requirements
```
pip install -r requirements.txt
```

## Description of Folders
1. data/: download raw datasets and preprocess raw data.
    - M4/m4_data_gen.py: download M4 data from the M4 git repo and generate M4 train/test sets.
    - Cryptos/cryptos_data_gen.py: generate cryptos train/test sets.
    - PlayerTraj/traj_data_gen.py: generate basketball trajectory train/test sets.
    - sample_data: a small subset of M4-weekly data for testing purposes.
    - data_analysis: functions for forecastability, trend and seasonality analysis.

2. modules/: pytorch modules for the KNF.
    - data_classes.py: pytorch datasets for M4, Cryptos, PlayerTraj and sample data.
    - train_utils.py: training and evaluation functions.
    - normalizer.py: pytorch implementation of reversible instance normalization.
    - models.py: main KNF modules.
    - eval_metrics.py: three different evaluation metrics for three different datasets.

3. run_koopman.py: training script for KNF.

4. args.py: hyperparameters.

5. run.sh: train KNF on the small sample dataset.

6. run_exp.sh: train KNF on all M4, Cryptos, PlayerTraj dataset.

7. evaluation.py: evaluate KNF on all M4, Cryptos, PlayerTraj dataset.


## Instructions
### Dataset and Preprocessing
- M4: Download and preprocess M4 data
```
python data/M4/m4_data_gen.py
```

- Cryptos: Download `train.csv` and `asset_details.csv` from [kaggle](https://www.kaggle.com/competitions/g-research-crypto-forecasting/data) to current `data/Cryptos` folder. Run `cryptos_data_gen.py` to preprocess Cryptos data
```
python data/Cryptos/cryptos_data_gen.py
```

- Traj: Download [NBA basketball player trajectory data](https://github.com/linouk23/NBA-Player-Movements/tree/master/data) and unzip all .7z files in `data/PlayerTraj/json_data`. Run `traj_data_gen.py` to preprocess Trajectory data. Since we didn't fix random seed when we sampled trajectory, to reproduce the results, please download [the same traj dataset we used](https://drive.google.com/drive/folders/1N_wo1I7G62HglyML5yL4FTEzbfekh4vZ?usp=sharing).
```
python data/PlayerTraj/traj_data_gen.py
```


### Training
- run `run.sh` to train a small KNF on a small subset of M4-weekly data in `data/sample_data`
```
sh run.sh
```

- run `run_exp.sh` to train three KNF models on all three datasets.
```
sh run_exp.sh
```

### Train KNF on a new dataset.
- Step1: Save the training and test sets of the new dataset separately as numpy arrays into two npy files. Both should have the shape of (number of time series, length, number of features).
- Step2: Use the `CustomDataset` class in `modules/data_classes.py` to load the data and specify the paths to train and test data files in arguments `direc` and `direc_test`.
- Step3: Change the default `num_feats` in `args.py` accordingly.
- Step4: Please do hyperparameter tuning, especially for `input_dim`, `input_length`, `num_steps` and `train_output_length`.

## Well-trained Models and Their Predictions
The well-trained models on all datasets and their prediction files, which generate the numbers in the paper, can be found [here](https://drive.google.com/drive/folders/1N_wo1I7G62HglyML5yL4FTEzbfekh4vZ?usp=sharing).

## Citation

If you find this repo useful, please cite [our paper](https://arxiv.org/abs/2210.03675).

```
@inproceedings{wang2023koopman,
title={Koopman Neural Operator Forecaster for Time-series with Temporal Distributional Shifts},
author={Rui Wang and Yihe Dong and Sercan O Arik and Rose Yu},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=kUmdmHxK5N}
}
```
