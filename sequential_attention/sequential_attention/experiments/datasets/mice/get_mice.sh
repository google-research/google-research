# https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression
MICE_URL=https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls
MICE_FILE='./Data_Cortex_Nuclear'
curl "${MICE_URL}" --output "${MICE_FILE}.xls"
python xls2csv.py --xls_file="${MICE_FILE}.xls" --csv_file="${MICE_FILE}.csv"