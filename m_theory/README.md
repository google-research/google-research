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

# Structure

  * 4d/so8_supergravity_extrema/
    Code for the scalar potential of the de Wit - Nicolai model,
    SO(8)-gauged N=8 Supergravity in 3+1-dimensional spacetime.

    Preprint: "SO(8) Supergravity and the Magic of Machine Learning"
    (https://bit.ly/2EMz81M)

    Demo: This will install and run a small demo search for a few solutions,
    plus analysis of one of those obtained. Output (providing location data and
    particle properties) will be in the directory "EXAMPLE_SOLUTIONS".


```shell
virtualenv -p python3 env
source env/bin/activate

pip install -r dim4/so8_supergravity_extrema/code/requirements.txt
python -m dim4.so8_supergravity_extrema.code.extrema
```
