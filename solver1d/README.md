# Solver1d

## Overview

This is an implementation of numerical solver of Schrodinger equation for non-interacting 1d system.

Note both solver (EigenSolver, SparseEigenSolver) here are based on directly
diagonalizing the Hamiltonian matrix, which are straightforward to understand,
but not as accurate as other delicate numerical methods, like density matrix
renormalization group (DMRG).

## Potentials

The following 1d potentials are implemented

* Gaussian dips
* Harmonic oscillator
* Sinlge Poschl-Teller potential

The solver support any 1d potential on 1d uniform grids.

## Example

Poschl-Teller potential is a special class of potentials for which the
one-dimensional Schrodinger equation can be solved in terms of Special
functions.

https://en.wikipedia.org/wiki/P%C3%B6schl%E2%80%93Teller_potential

The general form of the potential is

v(x) = -\frac{\lambda(\lambda + 1)}{2} a^2 \frac{1}{\cosh^2(a x)}

It holds M=ceil(\lambda) levels, where \lambda is a positive float.

For example, \lambda=0.5 holds 1 electron. The exact eigen energy is -0.125.

```bash
python -m solver1d.scripts.solve_poschl_teller_potential \
--solver=EigenSolver \
--lam=0.5 \
--scaling=1 \
--grid_lower=-20 \
--grid_upper=20 \
--num_grids=1001 \
--num_electrons=1
```

## Installation

Clone this repository and install in-place:

```bash
git clone https://github.com/google-research/google-research.git
pip install -e google-research/solver1d
```
