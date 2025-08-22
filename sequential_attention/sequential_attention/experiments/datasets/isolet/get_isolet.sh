DATA_DIR='.'

ISOLET_TRAIN_URL=https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet1+2+3+4.data.Z
ISOLET_EVAL_URL=https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet5.data.Z
ISOLET_TRAIN_FILE=isolet1+2+3+4.data
ISOLET_EVAL_FILE=isolet5.data

curl "${ISOLET_TRAIN_URL}" --output "${ISOLET_TRAIN_FILE}.Z"
curl "${ISOLET_EVAL_URL}" --output "${ISOLET_EVAL_FILE}.Z"
uncompress "${ISOLET_TRAIN_FILE}.Z"
uncompress "${ISOLET_EVAL_FILE}.Z"