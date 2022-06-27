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

# Running Colab Notebooks

The easiest way to work with this code, which does not require any code to be
installed, is to import the loader notebook into an online Google Colab, and
then let that notebook load the module either from github or from an on-device
archive (potentially modified).

The steps for this are as follows:

1. Navigate to:
   https://colab.research.google.com/github/google-research/google-research/blob/master/m_theory/colab/SUGRA_Colab.ipynb

1. Use the menu in the first notebook cell to select whether you want to load
   the `m_theory` module from the repository on github (new users should do
   this) or alternatively upload a (potentially modified) `m_theory.zip` archive
   from your device. (Enabling the `reset_package` checkbox allows replacing an
   already-loaded module with a fresh upload by re-running the cell.)

1. For a demo, run the 2nd notebook cell.
   Otherwise, modify the cell and run own code.

1. (Optional) Use the "File -> Download" or "File -> Save a Copy in Drive"
   (for users with a gmail / google drive account) menu item to save your work.


# Downloading and Installing locally

A simple (albeit somewhat strange) way to download only this part of the
(large!) google-research github repository is (this requires the
`subversion` package to be installed):

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

  * `m_theory_lib/`

     Generic library code for numerically studying supergravity.

  * `dim(N)/`

    Subdirectories for N-dimensional physics. `dim4/` holds definitions
    for 3+1-dimensional supergravity.

  * `dim(N)/papers/(paper_tag)/`

    Code accompanying various publications about supergravity.

  * `dim4/so8_supergravity_extrema/`

     Earlier SO(8) code that was published alongside
     "SO(8) Supergravity and the Magic of Machine Learning"
     (https://arxiv.org/abs/1906.00207) and is kept in the
     original form for reference. This code simplifies analysis
     of the scalar potential of the de Wit - Nicolai model,
     SO(8)-gauged N=8 Supergravity in 3+1-dimensional spacetime,
     but in comparison to more recent parts of this module has
     substantial gaps.

     Demo: This will run a small demo search for a few solutions, plus analysis
     of one of those obtained. Output (providing location data and particle
     properties) will be in the directory `EXAMPLE_SOLUTIONS`.

     `python3 -m dim4.so8_supergravity_extrema.code.extrema`


  * `wrapped_branes/`

     Code for analyzing the potentials of the models constructed in
     https://arxiv.org/abs/1906.08900 and https://arxiv.org/abs/1009.3805
     by wrapping M5-branes.

     Run via:
     `python3 -i -m wrapped_branes.wrapped_branes {problem_name}`
     with `{problem_name}` one of: `dim7`, `cgr-S2`, `cgr-R2`, `cgr-H2`,
     `dgkv-S3`, `dgkv-R3`, `dgkv-H3`.
