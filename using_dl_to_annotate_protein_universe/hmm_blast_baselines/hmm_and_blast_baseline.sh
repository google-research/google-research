# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# See supplement of https://doi.org/10.1101/626507 for more information.
# It's not recommended to run this script as it will take a long time (it
# includes a long BLASTP search). This script is instead useful as
# documentation.

# First, create a GCP instance, like this:
# PROJECT=YOUR_GCP_PROJECT
# ZONE=YOUR_GCP_ZONE

# gcloud beta compute --project "${PROJECT}" instances create \
# "blast-hmmer-timing" --zone "${ZONE}" --machine-type "n1-standard-32"  \
# --image=ubuntu-1604-lts-drawfork-v20190424 --image-project=eip-images \
# --boot-disk-size "250" --boot-disk-type "pd-ssd"

# gcloud beta compute ssh blast-hmmer-timing

# make a location in memory so we can run everything in the machine's memory.
TIMING_DIR=/dev/shm/timing
sudo mkdir ${TIMING_DIR}
sudo chown ${USER} ${TIMING_DIR}

# install required software
sudo apt-get --yes install make gcc

# Install blast
sudo apt-get --yes install ncbi-blast+

# install hmmer version 3.2.1
cd ~
wget http://eddylab.org/software/hmmer/hmmer-3.2.1.tar.gz
tar zxf hmmer-3.2.1.tar.gz
pushd hmmer-3.2.1
./configure --enable-threads
make
make check
popd
HMMSEARCH=~/hmmer-3.2.1/src/hmmsearch
HMMSCAN=~/hmmer-3.2.1/src/hmmscan
HMMPRESS=~/hmmer-3.2.1/src/hmmpress

# Get the Pfam 32.0 hmm profiles.
cd ${TIMING_DIR}
wget ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam32.0/Pfam-A.hmm.gz
gunzip Pfam-A.hmm

# Create the compressed hmm db for hmmscan
${HMMPRESS} Pfam-A.hmm

PROTEINS_BUCKET=gs://brain-genomics-public/research/proteins/timing

# Grab seed_train.fasta, full_train.fasta and seed_test.fasta
for f in full_train.fasta seed_train.fasta seed_test.fasta; do
  echo "Downloading file $f"
  curl -o ${TIMING_DIR}/${f} \
    https://storage.googleapis.com/brain-genomics-public/research/proteins/timing/${f};
done


wc -l *.fasta
# Expect to see 252342 lines for seed_test.fasta,
# 2173482 for seed_train.fasta
# and 87283672 for full_train.fasta.

# Create a 10% subset of the seed_test.fasta
head -n 25234 seed_test.fasta > seed_test.10_percent.fasta

# Create blast databases for seed and full train
makeblastdb -in seed_train.fasta -dbtype prot
makeblastdb -in full_train.fasta -dbtype prot


# Use the 10% sample of seed_test so the programs finish in a shorter timespan.
timing_fasta=seed_test.10_percent.fasta

# Use the full seed_test.fasta for a more complete runtime estimate.
# timing_fasta=seed_test.fasta

# We are using three replicates.
N_REPLICATES=3
HMMER_NCORES=1

# Time hmmscan and hmmsearch of seed_test.fasta against Pfam-A.hmm.
for replicate in $(seq $N_REPLICATES); do
for binary in ${HMMSCAN} ${HMMSEARCH}; do
  echo "Profiling hmmer ${binary} [replicate ${replicate}]"
  name="hmmer.${timing_fasta}.${binary##*/}.cores_${HMMER_NCORES}.rep_${replicate}"
  (time ${binary} \
    --cpu ${HMMER_NCORES} \
    --tblout ${name}.txt \
    -o ${name}.log \
    Pfam-A.hmm ${timing_fasta}) &> ${name}.time.log
  cat ${name}.time.log
done
done

# We want to use a different number of cores for each blast calculation.
# For seed, we want to use a single core so it's more directly comparable
# to hmmer. But blastp running on the 10% subset against the full
# training database takes a really long time. So we'll use all cores for that.
declare -A blast_database_ncores
blast_database_ncores[seed_train.fasta]=1
blast_database_ncores[full_train.fasta]=32

# Time blast against seed_train
for replicate in $(seq $N_REPLICATES); do
for blast_database in seed_train.fasta full_train.fasta; do
  ncores=${blast_database_ncores[${blast_database}]}
  echo "Profiling blastp against database \ 
  ${blast_database} with ${ncores} cores [replicate ${replicate}]"
  name="blast.${timing_fasta}.${blast_database}.cores_${ncores}.rep_${replicate}"
  (time blastp \
    -query ${timing_fasta} \
    -db ${blast_database} \
    -outfmt 10 -max_hsps 1 -num_alignments 1 \
    -num_threads ${ncores} \
    -out ${name}.out ) &> ${name}.time.log
  cat ${name}.time.log
done
done

# grep out all of the results:
fgrep real *.time.log
