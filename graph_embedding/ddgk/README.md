
DDGK: Learning Graph Representations for Deep Divergence Graph Kernels
===============================

This is the implementation of the WWW 2019 paper, [DDGK: Learning Graph Representations for Deep Divergence Graph Kernels](https://ai.google/research/pubs/pub47867).

The included code creates a Deep Divergence Graph Kernel as introduced in the paper.
The implementation makes use of the data sets collected [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

A distributed version of this implementation is necessary for large data sets.

Example Usage:
--------------
First, create a fresh virtual environment and install the requirements.

    # From google-research/
    virtualenv -p python3 .
    source ./bin/activate

    pip3 install -r graph_embedding/ddgk/requirements.txt

Then run the code on a [data set](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets):

    # From google-research/
    python3 -m graph_embedding.ddgk.main --data_set=https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip --working_dir=~/tmp

The code replaces Google internal, distributed libraries with a single-machine implementation. The code makes use of [scikit-learn](https://scikit-learn.org/) for support vector machine grid search where [DDGK: Learning Graph Representations for Deep Divergence Graph Kernels](https://ai.google/research/pubs/pub47867) made use of a distributed grid search.

Discrepancies between the code's results and those reported in [DDGK: Learning Graph Representations for Deep Divergence Graph Kernels](https://ai.google/research/pubs/pub47867) may occur (e.g., the paper reports `91.58` for `MUTAG` but with this code we attained only `91.04`).

Citing
------
If you find _Deep Divergence Graph Kernels_ useful in your research, we ask that you cite
the following paper:

> Al-Rfou, R., Zelle, D., Perozzi, B., (2019).
> DDGK: Learning Graph Representations for Deep Divergence Graph Kernels.
> In _The Web Conference_.

    @inproceedings{47867,
    title = {DDGK: Learning Graph Representations for Deep Divergence Graph Kernels},
    author = {Rami Al-Rfou and Dustin Zelle and Bryan Perozzi},
    year = {2019},
    booktitle = {Proceedings of the 2019 World Wide Web Conference on World Wide Web}
    }

Contact Us
----------
For questions or comments about the implementation, please contact
<dzelle@google.com> or <rmyeid@google.com>.
