#!/bin/bash

# Ensure a seed value is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

SEED=$1
echo "Running on seed $SEED..."

EXPERIMENT_DIR="/mnt/disks/checkpoints/martingale/pretrain"
RUNWD="$HOME/bnn_hmc/"
cd $RUNWD
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/miniforge3/envs/bnn"
export PYTHONPATH="$RUNWD/:$PYTHONPATH"

echo "Pretraining SGD..."
python bnn_hmc/run_sgd.py \
    --seed=$SEED \
    --weight_decay=10 \
    --dir=$EXPERIMENT_DIR/sgd/cifar10/ \
    --model_name=resnet20_frn_swish \
    --dataset_name=cifar10 \
    --subset_train_to=4080 \
    --init_step_size=3e-7 \
    --num_epochs=500 \
    --eval_freq=10 \
    --batch_size=80 \
    --save_freq=500

echo "Pretraining VI..."
python bnn_hmc/run_vi.py \
    --seed=$SEED \
    --weight_decay=5. \
    --dir=$EXPERIMENT_DIR/vi/cifar10/ \
    --model_name=resnet20_frn_swish \
    --dataset_name=cifar10 \
    --subset_train_to=4080 \
    --save_actual_dataset \
    --mean_init_checkpoint=$EXPERIMENT_DIR/sgd/cifar10/sgd_mom_0.9__lr_sch_i_3e-07___epochs_500_wd_10.0_batchsize_80_temp_1.0__seed_$SEED/model_step_499.pt \
    --init_step_size=1e-4 \
    --optimizer=Adam \
    --vi_sigma_init=0.01 \
    --temperature=1. \
    --vi_ensemble_size=20 \
    --num_epochs=300 \
    --eval_freq=10 \
    --batch_size=80 \
    --save_freq=300

