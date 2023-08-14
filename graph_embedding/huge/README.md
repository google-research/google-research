HUGE: Huge Unsupervised Graph Embeddings with TPUs
===============================

This is an implementation accompanying the KDD 2023 paper, [_HUGE: HUGE Unsupervised Graph Embeddings with TPUs_](https://arxiv.org/abs/2307.14490)

This is not an officially supported Google product.

HUGE-TPU requires a cluster of machines and access to Tensor Processing Units
(TPU)s to achieve the reported scale. However, the models and training routines
are compatible with CPU-only devices for debugging and experimentation purposes.
Using HUGE-TPU on CPU architectures will results in extreme scaling limitations
and may require hyper-parameter tuning to achieve the same results as the full
TPU system even on small scale data.

Since simulating random walks and generating co-occurrence counts can be done
offline with respect to training an embeddings, we will first make a port of
our HUGE-TPU embedding model and custom training loop (CTL) so that users can
provide their own co-occurrence count samples.

Eventually we will open-source a basic random walk sampler and co-occurrence
count generator in Beam. We will also provide mechanisms for running
Graph Embedding training on Google Cloud infrastructure.

Detailed installation instructions and examples will be updated at each part
continues through our internal review process and is pushed to GitHub.
Please be patient as both the technical aspects of porting the original
implementation to open-source systems and the corresponding reviews will take
some time. Even in the early stages of the evolution of this codebase,
we hope it is immediately useful of alternative applications for TPUs
and dedicated SparseCore chips.

Status
------

| Title | Description | Status |
|:-------| :--------- | -----: |
| I/O    | Initial I/O routines for sharded TfRecords of co-occurences. | COMPLETE |
| HUGE Model | Initial model definition for HUGE-TPU | COMPLETE |
| Training | Initial CTL and API for HUGE-TPU training | COMPLETE |
|Cloud Integration | Running training on Google-Cloud with TPU resources | IN-PROGRESS |
|Basic Sampling | Basic Beam sampler with Dataflow integration | Not Started |


Citing
------

If you find _HUGE_ useful in your research, we ask that you cite the following paper:

> Mayer, B.A, Tsitsulin, A., Fichtenberger, H., Halcrow, J. and Perozzi, B. (2023).
> HUGE: Huge Unsupervised Graph Embeddings with TPUs.
> In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23)_.

    @inproceedings{mayer2023huge,
     author={Mayer, Brandon A. and Tsitsulin, Anton and Fichtenberger, Hendrik and Halcrow, Jonathan and Perozzi, Bryan}
     title={HUGE: Huge Unsupervised Graph Embeddings with TPUs},
     booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23), August 6--10, 2023, Long Beach, CA, USA},
     year = {2023},
    }

Contact Us
----------
For questions or comments about the implementation, please contact [huge-public@google.com](mailto:huge-public@google.com).