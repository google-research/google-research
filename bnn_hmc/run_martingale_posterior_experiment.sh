#!/bin/bash

# Ensure a seed value is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <sgld-sample>"
    exit 1
fi

SAMPLE=$1
# Re-using SGLD sample as seed for sampling
SEED=$SAMPLE
echo "Running on seed $SEED..."

EXPERIMENT_DIR="$HOME/Projects/bnn_seq_vi/bnn_hmc/.runs/sgld_retraining/cifar10"
BASE_POSTERIOR_DIR="$HOME/Projects/bnn_seq_vi/bnn_hmc/.runs/sgmcmc_small/cifar10/sgld_mom_0.0_preconditioner_None__lr_sch_constant_i_1e-06_f_1e-06_c_50_bi_1000___epochs_10000_wd_5.0_batchsize_80_temp_1.0__seed_1"
RUNWD="$HOME/Projects/bnn_seq_vi"
cd $RUNWD
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/gabor/miniforge3/envs/bnn
export PYTHONPATH="$RUNWD/:$PYTHONPATH"

echo "Sampling synthetic labels..."
python bnn_hmc/sample_synthetic_labels.py \
    --seed=$SEED \
    --dir=$EXPERIMENT_DIR/synthdata/ \
    --model_name=resnet20_frn_swish \
    --dataset_name=cifar10 \
    --subset_train_to=8160 \
    --sequential_training \
    --num_sequential_training_folds=2 \
    --index_sequential_training_fold=1 \
    --params_checkpoint=$BASE_POSTERIOR_DIR/model_step_$SEED.pt \
    --append_synthetic_dataset_to=$BASE_POSTERIOR_DIR/data_subset_4080.npz

# Find the output file produced by SGD for the second half
SYNTHDATADIR=$(ls $EXPERIMENT_DIR/synthdata 2>/dev/null | grep -E "^sample_from_checkpoint_.*_subset_8160_split_2_of_2__seed_$SEED$")

if [ -z "$SYNTHDATADIR" ]; then
    echo "Error: No synthetic sampling runs were found with seed $SEED:\n$(ls $EXPERIMENT_DIR/sgd_2_of_2$DIRSUFFIX)"
    exit 1
fi

if [ $(echo "$SYNTHDATADIR" | wc -l) -gt 1 ]; then
    echo "Error: Multiple synthetic sampling runs were found with seed $SEED:\n$SYNTHDATADIR"
    exit 2
fi

echo "Running SGLD for the synthetic dataset"
python bnn_hmc/run_sgmcmc.py \
    --seed=$SEED \
    --weight_decay=5. \
    --dir=$EXPERIMENT_DIR \
    --dataset_name=$EXPERIMENT_DIR/synthdata/$SYNTHDATADIR/synth_appended_fba11d1.npz \
    --dataset_name=cifar10 \
    --model_name=resnet20_frn_swish \
    --init_step_size=1e-6 \
    --final_step_size=1e-6 \
    --num_epochs=10000 \
    --num_burnin_epochs=1000 \
    --eval_freq=10 \
    --batch_size=80 \
    --save_freq=100 \
    --momentum=0.

