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

echo "Pretraining SGLD..."
python bnn_hmc/run_sgmcmc.py \
    --seed=$SEED \
    --weight_decay=5. \
    --dir=$EXPERIMENT_DIR/sgld/cifar10/ \
    --model_name=resnet20_frn_swish \
    --dataset_name=cifar10 \
    --subset_train_to=4080 \
    --save_actual_dataset \
    --init_step_size=1e-6 \
    --final_step_size=1e-6 \
    --num_epochs=10000 \
    --num_burnin_epochs=1000 \
    --eval_freq=10 \
    --batch_size=80 \
    --save_freq=10 \
    --momentum=0.

