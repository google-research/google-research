DATA_DIR='.'
# https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php
COIL_URL=https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip
COIL_FILE=coil-20-proc
curl "${COIL_URL}" --output "${COIL_FILE}.zip"
unzip "${COIL_FILE}.zip"