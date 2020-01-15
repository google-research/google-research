# Experimental Python Framework for Sparse Data Classification

This repository details a Python framework which exposes an API to conduct
sparse data classification experiments. The framework is provided as a Python 2
module (`exp_framework`); experiment scripts which call this module are included
as well (`*_experiment.py`).

**Author:** Ji-Sung Kim (RMI Intern; jisungkim@google.com)
--------------------------------------------------------------------------------

## Run Experiments.

List of available scripts:

*   `experiment.py`: general experiments.
*   `tuning.py`: parameter tuning.

python -m experiment -- --logtostderr --dataset=sim_xor