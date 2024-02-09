# Library to run the indidually-fair clustering algorithms of the paper
# Scalable Individually-Fair K-Means Clustering, AISTATS 2024

# Usage

First, create a fresh virtual environment and install the requirements.

    # From google_research/
    virtualenv -p python3 .
    source ./bin/activate

    python3 -m pip install -r individually_fair_clustering/requirements.txt

Then, run the algorithm. Example command:

python3 -m individually_fair_clustering.run_individually_fair_clustering \
    --input=path-to-input.tsv \
    --output=path-to-output.json \
    --k=10 \
    --algorithm="LSPP"

The input of consists of a file in tab separated format with each row being a
point and each column being a dimension of the point. All dimensions are float.
All points should have the same dimension.

The output is a json file in text format encoding a data frame with a single
row. The data frame contains statistics about the result of the algorithm
(e.g., the k-means cost).