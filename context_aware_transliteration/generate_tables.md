# Building the LaTeX tables from the paper.

This document provides instructions for generating latex tables in the paper
from raw system outputs. [Return to main page.](README.md)

The roughly 1GB tar file
[generate_single_word_tables.tar](https://storage.googleapis.com/gresearch/context_aware_transliteration/generate_single_word_tables.tar)
includes scripts to generate single word transliteration results from the paper,
along with the raw system outputs that the scripts operate on.

The roughly 5.2GB tar file
[generate_full_string_tables.tar](https://storage.googleapis.com/gresearch/context_aware_transliteration/generate_full_string_tables.tar)
includes scripts to generate full string transliteration results from the paper,
along with the raw system outputs that the scripts operate on.

The scripts rely on the [Nisaba](https://github.com/google-research/nisaba)
library being installed locally (see the instructions below).  Let `$MYNISABA`
be a variable pointing to the Nisaba library directory, which should have a
compiled `bazel-bin` subdirectory underneath it.

Let `$MYLOCALDIR` be a variable pointing to the local directory where the
above-linked tar files will be downloaded.  Note that the tar files and
resulting files are quite large, so allocate about 20G to run comfortably.
Also, because of the number of conditions, the time to produce the tables will
likely be hours.

To build the single word LaTeX tables (omit wget if tar files already
downloaded):

```
cd $MYLOCALDIR
wget https://storage.googleapis.com/gresearch/context_aware_transliteration/generate_single_word_tables.tar
tar -xf generate_single_word_tables.tar
./single_word_scripts/generate_single_word_tables.sh $MYNISABA
```

To build the full string LaTeX tables (omit wget if tar files already
downloaded):

```
cd $MYLOCALDIR
wget https://storage.googleapis.com/gresearch/context_aware_transliteration/generate_full_string_tables.tar
tar -xf generate_full_string_tables.tar
./full_string_scripts/generate_full_string_tables.sh $MYNISABA
```

Both scripts can be safely run simultaneously.  Also note that each of the above
scripts can be run in stages -- please see the scripts for details.  After
running the scripts, the LaTeX tables will be found in the
`single_word_latex_tables` and `full_string_latex_tables directories` in
`$MYLOCALDIR`.  Other intermediate results are also left for inspection in the
directory.

## Installing Nisaba

The [library](https://github.com/google-research/nisaba) is built using
[Bazelisk](https://github.com/bazelbuild/bazelisk), a wrapper over Bazel build
system. The following instructions assume the host machine is amd64 Linux, but
the library can be built on other architectures as well by downloading an
appropriate version of Bazelisk:

```
git clone https://github.com/google-research/nisaba.git
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
cd nisaba
USE_BAZEL_VERSION=6.5.0 ../bazelisk-linux-amd64 build -c opt nisaba/...
MYNISABA=${PWD}
```
