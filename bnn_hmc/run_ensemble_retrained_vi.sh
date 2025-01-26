#!/bin/bash

# Ensure a seed value is provided
if [[ "$1" == "-bootstrap" ]]; then
	BOOTSTRAP=true
	if [ -n "$2" ]; then
		 RANDOM="$2"
	else
		$RANDOM=$(date '+%s')
	fi
	echo "Running on bootstrap shuffle seed $RANDOM"
elif [ -n "$1" ]; then
	echo "Usage: $0 <seed> [-pca]"
	exit 1
else
	BOOTSTRAP=false
fi

# Define input filenames and number of bootstrap samples
B=100  # Number of bootstrap samples
EXPERIMENT_DIR="/mnt/disks/checkpoints/martingale"
RUNWD="$HOME/bnn_hmc/"
cd $RUNWD
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $HOME/miniforge3/envs/bnn
# PYTHON=`which python`
# echo "Using python: $PYTHON"
export PYTHONPATH="$RUNWD/:$PYTHONPATH"

readarray -d '' input_checkpoints < <(find $EXPERIMENT_DIR/retrain/vi/cifar10 -wholename "*/mfvi_initsigma_0.01_meaninit__opt_adam__lr_sch_i_0.0001___epochs_300_wd_5.0_batchsize_80_temp_1.0__seed_*/model_step_299.pt" -print0)
# Get the number of input files
num_files="${#input_checkpoints[@]}"

if [ "$BOOTSTRAP" = true ]; then
	# Perform resampling B times
	for ((i=1; i<=B; i++)); do
		# Generate a bootstrap sample with replacement
		bootstrap_sample=()
		for ((j=0; j<num_files; j++)); do
			random_index=$((RANDOM % num_files))
			bootstrap_sample+=("${input_checkpoints[random_index]}")
		done

		# Call your script with the resampled filenames
		echo "Bootstrap ensemble $i... length ${#bootstrap_sample[@]}"
		python bnn_hmc/ensemble_checkpoints.py \
		    --dir=$EXPERIMENT_DIR/ensemble/vi/cifar10/bootstrap \
		    --model_name=resnet20_frn_swish \
		    --dataset_name=cifar10 \
		    --subset_train_to=4080 \
		    --vi_checkpoints \
		    -- ${bootstrap_sample[@]}
	done
else
	echo "Ensembling checkpoints..."
	python bnn_hmc/ensemble_checkpoints.py \
	    --dir=$EXPERIMENT_DIR/ensemble/vi/cifar10 \
	    --model_name=resnet20_frn_swish \
	    --dataset_name=cifar10 \
	    --subset_train_to=4080 \
	    --vi_checkpoints \
	    -- ${input_checkpoints[@]}
fi

