# Linear Additive Markov Processes

This directory contains implementation of and experimental code for the paper

[Ravi Kumar, Maithra Raghu, Tamás Sarlós, and Andrew Tomkins. "Linear Additive Markov
Processes", WWW 2017.](https://arxiv.org/abs/1704.01255)

## Reproducing the Experiments

On MacOS:

0) Install Homebrew from http://brew.sh

1) Install Google command line flags with

    brew install gflags

Linking sometimes fails with
"Linking /usr/local/Cellar/gflags/2.0...
Error: Could not symlink bin/gflags_completions.sh
/usr/local/bin is not writable.",
which is OK.

On Ubuntu Linux:

1) Install Google command line flags with

    sudo apt-get install libgflags2 libgflags-dev

Common steps on either Mac or Linux:

2) Download the input datasets

    mkdir -p data

See Section 6.1 on page 6 of the paper for the description and URLs of the various datasets.
E.g. download and unzip
https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz
into `data/`.

Create another directory, one level above, for the plots in the paper

    mkdir -p ../data

3) Compile it

    make all

4) Run it

To avoid errors on Mac like:
"dyld: Library not loaded: /usr/local/opt/gflags/lib/libgflags.2.dylib
  Referenced from: /Users/stamas/Documents/lamp/src/./lamp
  Reason: image not found"

    export DYLD_LIBRARY_PATH=/usr/local/Cellar/gflags/2.2.2/lib:$DYLD_LIBRARY_PATH

To reproduce figues in the paper run the following series of commands.
Since it takes a while for each to complete you may wish to decrease \*max\*iter and
\*max\*order params first, plot the results, and then repeat it with the settings
below.

    time ./lamp --max_outer_iter=3 --grad_max_weight_iter=30 --grad_max_transitions_iter=30 --max_train_time=2010 --dataset=brightkite --plot_file ../data/brightkite-performance.tsv --min_location_count 10 --max_lamp_order 7 2>&1 | tee brightkite.log
    nohup time ./lamp --max_outer_iter=3 --grad_max_weight_iter=10 --grad_max_transitions_iter=10 --max_train_time=2009-02 --dataset=lastfm --plot_file ../data/lastfm-performance.tsv --min_location_count 50 --lastfm_max_users 1000 --max_lamp_order 7 --max_ngram_order 4 &
    nohup time ./lamp --max_outer_iter=3 --grad_max_weight_iter=30 --grad_max_transitions_iter=30 --dataset=reuters --plot_file ../data/reuters-performance.tsv --min_location_count 10 --max_lamp_order 7 &
    time ./lamp --max_outer_iter=3 --grad_max_weight_iter=30 --grad_max_transitions_iter=30 --num_folds 10 --dataset=wiki --plot_file ../data/wiki-performance.tsv --min_location_count 10 --max_lamp_order 7 2>&1 | tee wiki.log

    ./plotall.sh ../data/brightkite-performance.tsv brightkite
    ./plotall.sh ../data/lastfm-performance.tsv lastfm 10000
    ./plotall.sh ../data/reuters-performance.tsv reuters 500
    ./plotall.sh ../data/wiki-performance.tsv wiki

