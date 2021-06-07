#!/bin/bash

set -e

SMUDIR=$(dirname $0)
ROOTDIR=$(realpath ${SMUDIR}/..)
VENV="${SMUDIR}/venv"

if [ -z "${VIRTUAL_ENV}" -a -d "${VENV}" ]; then
    echo "Activating virtual environment in ${VENV}"
    source ${VENV}/bin/activate
fi

export PYTHONPATH=${PYTHONPATH}:${ROOTDIR}

set -u

for TESTFN in $(find $SMUDIR -name *_test.py) 
do
    if [[ $TESTFN == *"$VENV"* ]]; then
        continue
    fi
    echo "Executing ${TESTFN}"
    python $TESTFN
done

echo "ALL TESTS PASSED"