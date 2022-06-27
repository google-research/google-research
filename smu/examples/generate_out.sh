#!/bin/sh

# This script regenerates all the .out files for the examples.
# It should be run in the directory where the database files are

set -eux

EXAMPLEDIR=$(dirname $0)

for FN in $(ls ${EXAMPLEDIR}/*.py)
do
    BASE=$(basename ${FN})
    STEM=${BASE%.*}

    python -m smu.examples.${STEM} > ${EXAMPLEDIR}/${STEM}.out
done
