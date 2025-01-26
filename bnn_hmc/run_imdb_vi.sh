#!/bin/bash

arg="imdb"

if [[ "$arg" == "imdb" ]]; then
    DATASET="imdb"
    MODEL="cnn_lstm"
    SGD_WEIGHT_DECAY="3."
    MFVI_WEIGHT_DECAY="5"
    COMMON_HYPERPARAMS="--dataset_name=$DATASET --model_name=$MODEL"
    SGD_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=3e-7 --num_epochs=500 --eval_freq=20 --batch_size=80 --save_freq=500"
    VI_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=1e-4 --num_epochs=300 --eval_freq=10 --batch_size=80 --save_freq=150 --optimizer=Adam --vi_sigma_init=0.01 --temperature=1. --vi_ensemble_size=50"
else
    echo "Please specify cifar10 or imdb"
    exit 1
fi

# Ensure a seed value is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

SEED=$1
echo "Running on seed $SEED..."

STRATIFIED_ARGS=""
DIRSUFFIX=""
# Check for optional -pca argument
if [[ "$2" == "-pca" ]]; then
    STRATIFIED_ARGS="--stratified_folds=pca"
    DIRSUFFIX="_pca"
fi

EXPERIMENT_DIR="$HOME/Projects/bnn_seq_vi/bnn_hmc/.runs/multiseed/$DATASET"
RUNWD="$HOME/Projects/bnn_seq_vi"
cd $RUNWD
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/gabor/miniforge3/envs/bnn
# PYTHON=`which python`
# echo "Using python: $PYTHON"
export PYTHONPATH="$RUNWD/:$PYTHONPATH"

echo "SGD..."
python bnn_hmc/run_sgd.py --seed=$SEED --weight_decay=$SGD_WEIGHT_DECAY $SGD_HYPERPARAMS --dir=$EXPERIMENT_DIR/sgd$DIRSUFFIX/

echo "MFVI..."
python bnn_hmc/run_vi.py --seed=$SEED --weight_decay=$MFVI_WEIGHT_DECAY $VI_HYPERPARAMS --dir=$EXPERIMENT_DIR/mfvi$DIRSUFFIX/ \
    --mean_init_checkpoint=$EXPERIMENT_DIR/sgd$DIRSUFFIX/sgd_mom_0.9__lr_sch_i_3e-07___epochs_500_wd_3.0_batchsize_80_temp_1.0__seed_$SEED/model_step_499.pt

