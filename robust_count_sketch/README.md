# On the Robustness of CountSketch to Adaptive Inputs

This directory contains the implementation and experimental code for

Edith Cohen, Xin Lyu, Jelani Nelson, Tamás Sarlós, Moshe Shechner, Uri Stemmer
["On the Robustness of CountSketch to Adaptive Inputs"](https://arxiv.org/abs/2202.13736), ICML 2022.

## Reproducing the Experiments

1) Run once

    pip3 install cityhash matplotlib numpy
    mkdir results

2) Attack CountSketch and the median estimator

    python3 sim_1_code.py

3) Attack round dependency of CountSketch

    python3 sim_2_code.py

4) Comparison of non-robust BucketCountSketch and sign alignment estimator with
the classic CountSketch and median estimator.
Edit num_trials etc. parametrs in bcs.py first then pass num_trials to plot.py

    python3 bcs.py
    python3 plot.py file_written_by_bcs.tsv $num_trials
