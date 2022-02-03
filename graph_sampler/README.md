# Graph/Molecule sampler

## Overview

This project implements sequential importance sampling for connected graphs,
with a particular focus on molecular graphs.

It was built to sample small molecules *uniformly* at random without being able
to explicitly enumerate them. For example, suppose we want a uniform sampling of
molecules with a given number of heavy (i.e. non-hydrogen) atoms any any number
of hydrogens.

We first enumerate all possible choices of heavy atoms and numbers of hydrogens
(i.e. stoichiometries). For each stoichiometry, we generate importance-weighted
samples, and then use rejection to get to a uniform sampling. Finally, we
aggregate those uniform samples to a single uniform sampling of the space of all
molecules with the given number of heavy atoms.

```
virtualenv -p python3 .
source ./bin/activate
pip install -e .

HEAVY_ELEMENTS=C,N,O,F
NUM_HEAVY=3

mkdir outputs
cd outputs
mkdir stoichs weighted uniform

python -m enumerate_stoichiometries --output_prefix=stoichs/ \
  --num_heavy=$NUM_HEAVY --heavy_elements=$HEAVY_ELEMENTS

for stoich_file in $(ls stoichs); do
  prefix=${stoich_file%.*}
  python -m sample_molecules --stoich_file=stoichs/$prefix.stoich \
    --out_file=weighted/$prefix.graphml
  python -m reject_to_uniform --in_file=weighted/$prefix.graphml \
    --out_file=uniform/$prefix.graphml
done
python -m stats_to_csv --output=weighted/stats.csv weighted/*.graphml
python -m stats_to_csv --output=uniform/stats.csv uniform/*.graphml

merged_filename="${NUM_HEAVY}_${HEAVY_ELEMENTS}_uniform"
python -m aggregate_uniform_samples --output=${merged_filename}.graphml \
    uniform/*.graphml
python -m graphs_to_smiles ${merged_filename}.graphml > ${merged_filename}.smi
```

Comments:

*   This example is a little silly, because there are so few molecules that we
    could explicitly enumerate them and sample uniformly from the list. However,
    that approach quickly becomes infeasible as the number of heavy atoms (and
    the choices for what those atoms are) increases. In this small run, each
    molecule gets generated many times, so we can take the opportunity to see
    how uniform the final sampling is:

    ```
    sort ${merged_filename}.smi | uniq -c
    ```

*   This process is highly parallelizable, which is important with a greater
    number of heavy atoms. Parallelizing the loop over stoichiometries is the
    simplest and biggest win.

### Troubleshooting

#### The final uniform sample isn't as big as I want

Check to see if any of your individual stoichiometries are holding you back by
checking what proportion of the space they sample (look at the ratio of the
columns `num_after_rejection` and `estimated_num_graphs` in
`weighted/stats.csv`). If one of those is much smaller than the rest, a lot of
samples have to be thrown out to achieve uniformity. You can force a certain
minimum proportion of the space to be sampled by `sample_molecules.py`. For
example, setting `--min_uniform_proportion=1e-5` will ensure we keep sampling
until `num_after_rejection / estimated_num_graphs` is at least 1e-5.

#### I want a *really* uniform sample set, or a better estimate of the number of unique molecules

Reduce the value of `--relative_precision` when calling `sample_molecules.py`.
This parameter defaults to 0.01, and it measures our relative uncertainty in the
size of the space we're exploring. By default, we keep sampling until we're
confident we know the true number of graphs to within about 1%. The more
precisely we estimate the number of molecules for each stoichiometry, the better
a job we can do combining them into a single uniform sample set.

#### It's too slow

First, parallize over stoichiometries. It's possible to paralleize sampling,
rejection, and aggregation further, but we never needed to do that. If you need
help parallelizing further (e.g. you just care about a single really big
stoichiometry), email geraschenko@google.com and I'll help you.

If you're willing to accept a smaller final sample set and be less confident
about how uniform it is, you can reduce `--min_samples` (defaults to 10000) or
`--relative_precision` (defaults to 0.01) when calling `sample_molecules.py`.

## Installation

```
git clone https://github.com/google-research/google-research.git
pip install google-research/graph_sampler
```
