# Pokemon Synthetic Data generation
This directory contains scripts needed to generate the synthetic pokemon
data. It assumes a data/ directory is already populated.

## Make names and kb

```
python make_names.py data/pokemon.csv data/move_names.csv data/type_names.csv
                     data/ability_names.csv data/generation_names.csv
python make_kb.py
```

This creates the names.tsv and kb.tsv files in the current directory.

## Sample categories

```
$ python make_cats.py
(number sampled, number accepted)
```
This samples potential category names and rejects duplicates. For around 30K
examples, it takes around 100K samples.

## Convert to Wikidata

Adds "i/" in front of everything and outputs a corresponding dev split
file. Might not be needed anymore. Example usage:

```
python conform_to_wd.py curr_dir/ property_linking_data_dir/ pokemon
```
