# Scripts assocated with Property Linking

Each script also has an example usage. While the original scripts contained
hardcoded paths, those have been removed.

## Data exploration scripts

These scripts are used for data exploration. They are nondestructive, do not
write to files, and operate on specific files. These are often useful for
debugging (e.g. faster than grep).

histogram.py: This is a generic histogram computation file; here it takes the KB
and computes the top 600 most frequent values for the head entities, relations,
and tail entities.

```
$ python histogram.py kb.tsv names.tsv
(num_relations, num_head_entities, num_tail_entities)
[relations, sorted by frequency]
[head_entities, sorted by frequency]
[tail_entities, sorted by frequency]
```

wordcounts.py: This returns the frequency (top 1000) of individual word tokens
in the intersection of two KB files (takes the min). Useful for seeing overlap
between different domains.

```
$ python wordcounts.py zoo_names.tsv film_names.tsv
[sorted list of tokens]
```

## Bridge scripts

These scripts are used to bridge between the output of the Sling KB extractor
and the expected input to property_linker.py. These are intended to be run once
over the output of the Sling KB extractor and perform fairly simple tasks that
could be incorporated directly into the KB extraction itself.

add_i.py: The Sling KB cats file output is of the form (example)

`Q# Category:Placeholder Category Names Q#|Q#`

but both the KB and names file refers to the entities and category names as
i/Q#. This script adds an i/ before everything in *_cats.tsv so that the file
can correctly reference other files.

```
$ python add_i.py cats.tsv
(no output)
```
!! This overwrites cats.tsv.

reduce_names.py: This takes a kb and a (possibly large) names file and prunes
the names file down to just the entities in the kb. This is useful because the
names file is often large (O(100M) lines) and can be reduced significantly,
which speeds up file i/o.

```
$ python reduce_names.py kb.tsv names_large.tsv names_reduced.tsv
(# unique names)
(100 examples)
(strings that are duplicated in names_large)
(length of names_reduced, should equal # unique names)
(If they aren't equal, what's left over)
```
!! This writes into `names_reduced.tsv`.

restrict_to_choi.py: Take a large kb/cats/names set and restricts to to the
categories used by Choi et al., 2015.

```
python restrict_to_choi.py
```
!! Writes into `yago_s_cats.tsv`, `yago_s_names.tsv`, `yago_s_kb.tsv`
!! Expects `category`, `entity`, and `yagoTypes.clean` to be in `../yago_choi/`
!! Expects `entity_cats.tsv`, `entity_names.tsv`, `entity_kb.tsv` in current dir

## Vital preprocessing scripts

These scripts compute additional features, reduce the size of the KB, or enhance
the KB in some way. Some of them interface with the YAGO wiki category data from
Choi et al., 2015).

preproc_all.sh: Does all of the below: translates from fbjson to wdtsv, computes
overlap features and adds soft candidates, then appropriately splits the data.

```
$ ./preproc_all.sh
```
!! Expects a lot of files to be in the right place; i.e. everything below needs
to work on their own.

fbjson_to_wdtsv.py: This take the freebase json formatted gold parses and
converts it to a wikidata tsv based file matching the format of the training
examples. The gold properties are placed in column 4. This file depends on a
freebase <-> wikidata file, and a manually annotated list of translations
between freebase and wikidata.

```
$ python3 fbjson_to_wdtsv.py dev500.json mid_to_wd.tsv
         rels_fb_to_wd.tsv cats.tsv cats_dev.tsv
(coverage stats per example)
(statistics on what failed to convert)
(frequency counts on what failed to convert)
(total count of examples)
```
This is not a general purpose script. It is tailored specifically to target the
freebase semantic parses from Choi et al., 2015.
dev500.json - Provided by Choi et al., 2015
mid_to_wd.tsv - A mapping from mid to wd (e.g. `grep "^i/P646	" kb.tsv`)
rels_fb_to_wd.tsv - a manually curated mapping for the mids that weren't found.
cats.tsv - Original cats file with entity sets
cats_dev.tsv - Output. Empty entity set if category not in `cats.tsv`. Empty
               properties if Freebase is not properly parsed
!! cats_dev.tsv is overwritten.
!! This needs python3

overlap.py: This script is used to cache string overlap features. It takes an
examples (cats) file and a names file and for each example, determines which
names are a subset of the example, and adds a column to the tsv with the IDs
of the overlapping node.

```
$ python overlap.py cats.tsv names.tsv cats_output.tsv
(number of categories, number of names contained somewhere
(times to compute overlap for 10 examples)
```
!! `cats_output.tsv` is overwritten.

add_scored_links.py: [WIP] This script takes scored soft links predicted by a
entity linking candidate generation system and incorporates the top 5 as an
additional column (column 6) in the training examples file (cats file). It
should also include the scores, but that has not been implemented.

```
$ python add_scored_links.py matches.txt cats.tsv cats_output.tsv
(number of soft matches, 20 examples)
```
Adds another column to `cats.tsv` but writes in `cats_output.tsv` instea
!! `cats_output.tsv` is overwritten.

prune_entity: [old] This takes a large KG and iteratively removes the least
frequent k% sets (by size) and entities (by frequency). It was deprecated when
we started using sitelinks, but it might still be useful.

```
$ python prune_entity.py ~/path/to/files
(number of entities)
(stats on what is left)
..
```
!! Expects several files to be in the right place, and overwrites many files.

## Additional Information

Other scripts that may have been useful but do not fit one of these categories
are not included here as they do not help future development of the project and
are not needed for replication.


