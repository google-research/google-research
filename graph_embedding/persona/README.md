Is a Single Embedding Enough? Learning Node Representations that Capture Multiple Social Contexts
===============================

This is the implementation accompanying the WWW 2019 paper, [_Is a Single Embedding Enough? Learning Node Representations that Capture Multiple Social Contexts_](https://ai.google/research/pubs/pub47956).
The code also allows to create persona graphs to obtain overlapping clusters as
defined in the KDD 2017 paper [_Ego-splitting Framework: from Non-Overlapping 
to Overlapping Clusters_](https://ai.google/research/pubs/pub46238).

We are releasing the code in two installments. This is the first and it allows
to compute and cluster the persona graphs.
The second commit will allow to compute the embedding of the graph.

Example Usage:
--------------
First, create a fresh virtual environment and install the requirements.

    # From google-research/
    virtualenv -p python3 .
    source ./bin/activate

    python3 -m pip install -r graph_embedding/persona/requirements.txt

Then run the code on a [NetworkX](ttps://networkx.github.io/) dataset, you
can run the code like the following.

    # From google-research/
    python3 -m graph_embedding.persona.persona --input_graph=${graph} \
       --output_clustering=${clustering_output}

Citing
------
If you find _Persona Embedding_ useful in your research, we ask that you cite
the following paper:

> Epasto, A., Perozzi, B., (2019).
> Is a Single Embedding Enough? Learning Node Representations that Capture
Multiple Social Contexts.
> In _The Web Conference_.

    @inproceedings{epasto2019learning,
     author={Epasto, Alessandro and Perozzi, Bryan}
     title={Is a Single Embedding Enough? Learning Node Representations that
     Capture Multiple Social Contexts},
     booktitle = {The Web Conference},
     year = {2019},
    }

Contact Us
----------
For questions or comments about the implementation, please contact
<aepasto@google.com> and <bperozzi@acm.org>.
