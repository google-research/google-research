#!/bin/bash

# Ensure a seed value is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

SEED=$1
PRETRAINED_SAMPLING_SEED="1"
echo "Running on sample seed $PRETRAINED_SAMPLING_SEED, retraining seed $SEED..."

EXPERIMENT_DIR="/mnt/disks/checkpoints/martingale"
BASE_POSTERIOR_DIR="/mnt/disks/checkpoints/martingale/pretrain/vi/cifar10/mfvi_initsigma_0.01_meaninit__opt_adam__lr_sch_i_0.0001___epochs_300_wd_5.0_batchsize_80_temp_1.0__seed_$PRETRAINED_SAMPLING_SEED"
RUNWD="$HOME/bnn_hmc/"
cd $RUNWD
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/miniforge3/envs/bnn"
export PYTHONPATH="$RUNWD/:$PYTHONPATH"

echo "Sampling synthetic labels..."
python bnn_hmc/sample_synthetic_labels.py \
    --seed=$SEED \
    --dir=$EXPERIMENT_DIR/synthdata/vi/cifar10 \
    --model_name=resnet20_frn_swish \
    --dataset_name=cifar10 \
    --subset_train_to=8160 \
    --sequential_training \
    --num_sequential_training_folds=2 \
    --index_sequential_training_fold=1 \
    --vi_checkpoint=$BASE_POSTERIOR_DIR/model_step_299.pt \
    --append_synthetic_dataset_to=$BASE_POSTERIOR_DIR/data_subset_4080.npz

# Find the output file produced by SGD for the second half
SYNTHDATADIR=$(ls $EXPERIMENT_DIR/synthdata/vi/cifar10 2>/dev/null | grep -E "^sample_from_vi_.*_subset_8160_split_2_of_2__seed_$SEED$")
if [ -z "$SYNTHDATADIR" ]; then
    echo "Error: No synthetic sampling runs were found with seed $SEED:\n$SYNTHDATADIR"
    exit 1
elif [ $(echo "$SYNTHDATADIR" | wc -l) -gt 1 ]; then
    echo "Error: Multiple synthetic sampling runs were found with seed $SEED:\n$SYNTHDATADIR"
    exit 2
fi

SYNTHDATAFILE=$(ls $EXPERIMENT_DIR/synthdata/vi/cifar10/$SYNTHDATADIR 2>/dev/null | grep -E "^synth_appended_.*.npz$")
if [ -z "$SYNTHDATAFILE" ]; then
    echo "Error: No synthetic sampling run files were found with seed $SEED:\n$SYNTHDATAFILE"
    exit 3
elif [ $(echo "$SYNTHDATAFILE" | wc -l) -gt 1 ]; then
    echo "Error: Multiple synthetic sampling run files were found with seed $SEED:\n$SYNTHDATAFILE"
    exit 4
fi

echo "Retraining SGD..."
python bnn_hmc/run_sgd.py \
    --seed=$SEED \
    --weight_decay=10 \
    --dir=$EXPERIMENT_DIR/retrain/sgd_for_vi/cifar10/ \
    --dataset_name=$EXPERIMENT_DIR/synthdata/vi/cifar10/$SYNTHDATADIR/$SYNTHDATAFILE \
    --subset_train_to=8160 \
    --model_name=resnet20_frn_swish \
    --init_step_size=3e-7 \
    --num_epochs=500 \
    --eval_freq=10 \
    --batch_size=80 \
    --save_freq=500

echo "Retraining VI..."
python bnn_hmc/run_vi.py \
    --seed=$SEED \
    --weight_decay=5. \
    --dir=$EXPERIMENT_DIR/retrain/vi/cifar10/ \
    --dataset_name=$EXPERIMENT_DIR/synthdata/vi/cifar10/$SYNTHDATADIR/$SYNTHDATAFILE \
    --subset_train_to=8160 \
    --model_name=resnet20_frn_swish \
    --mean_init_checkpoint=$EXPERIMENT_DIR/retrain/sgd_for_vi/cifar10/sgd_mom_0.9__lr_sch_i_3e-07___epochs_500_wd_10.0_batchsize_80_temp_1.0__seed_$SEED/model_step_499.pt \
    --init_step_size=1e-4 \
    --optimizer=Adam \
    --vi_sigma_init=0.01 \
    --temperature=1. \
    --vi_ensemble_size=20 \
    --num_epochs=300 \
    --eval_freq=10 \
    --batch_size=80 \
    --save_freq=300

