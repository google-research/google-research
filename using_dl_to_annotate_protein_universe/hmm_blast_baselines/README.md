This code is intended to support the publication "Using Deep Learning to
Annotate the Protein Universe". [preprint link](https://doi.org/10.1101/626507)

# Short description of files

-   hmm_blast_baselines.sh: documentation for how to run HMMER or BLAST on
    our test data, using a sandboxed instance in Google Cloud Platform.
-   train_hmmer_model_for_paper.py: ties model building and inference together.
-   generate_hmmer_files.py: makes inputs to HMMER
-   hmmer.py: run HMMER commands like hmmbuild/hmmsearch
-   phmmer.py: run phmmer
