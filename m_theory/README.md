# M-Theory Research

This directory contains TensorFlow-based code to address some research problems
in M-Theory / Superstring Theory / Supergravity / Quantum Gravity.

Broadly speaking, M-Theory is all about the very rich mathematical structure
that arises if one tries to reconcile the physical principles of Quantum
Mechanics, General Relativity, and Supersymmetry. In terms of "inputs" and
"data", this research should be regarded as Pure Mathematics, i.e. there is no
dependency on "measurement" (or even "user") data.

Still, *despite* this research not using any data examples (for learning or
otherwise), Google's TensorFlow Machine Learning technology is sufficiently
generic to be a very useful tool to address some research questions in this
domain that can/should be studied numerically.

# Downloading and Installation

A simple (albeit somewhat strange) way to download only this part of the
google-research github repository is:

```shell
svn export https://github.com/google-research/google-research/trunk/m_theory
```


Then, the Python environment can be set up as follows:

```shell
virtualenv -p python3 env
source env/bin/activate

pip3 install -r m_theory_lib/requirements.txt
```


# Structure

  * `dim4/so8_supergravity_extrema/`

    Code for the scalar potential of the de Wit - Nicolai model,
    SO(8)-gauged N=8 Supergravity in 3+1-dimensional spacetime.

    Article: "SO(8) Supergravity and the Magic of Machine Learning"
    (https://arxiv.org/abs/1906.00207).

    Demo: This will run a small demo search for a few solutions,
    plus analysis of one of those obtained. Output (providing location data and
    particle properties) will be in the directory `EXAMPLE_SOLUTIONS`.

    `python3 -m dim4.so8_supergravity_extrema.code.extrema`


  * `wrapped_branes/`


    Code for analyzing the potentials of the models constructed in
    https://arxiv.org/abs/1906.08900 and https://arxiv.org/abs/1009.3805
    by wrapping M5-branes.

    Run via:
    `python3 -i -m wrapped_branes.wrapped_branes {problem_name}`
    with `{problem_name}` one of: `dim7`, `cgr-S2`, `cgr-R2`, `cgr-H2`,
    `dgkv-S3`, `dgkv-R3`, `dgkv-H3`.
