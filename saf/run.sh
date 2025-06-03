set -e

mkdir -p logs

EXPERIMENT_VERSION=v0

SYN_OPTION=1
DATASET=synthetic_autoregressive
LEN_TOTAL=100
MODEL_NAME=lstm_seq2seq
python3 -m experiment_${DATASET} --model_type=${MODEL_NAME} --gpu_index=-1 --synthetic_data_option=${SYN_OPTION} --len_total=${LEN_TOTAL} --num_trials=1 --filename=experiment_${DATASET}_${MODEL_NAME}_${EXPERIMENT_VERSION}
