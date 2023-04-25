Implementation of clustering algorithms used in
"Scalable Differentially Private Clustering via Hierarchically
Separated Trees" KDD'22 https://arxiv.org/abs/2206.08646

# Usage
From google_search root directory:

OUTPUT_DIR=some/directory
python3 -m hst_clustering.run_example --raw_data=hst_clustering/s1_random-standarized.txt --hst_data=hst_clustering/s1-cell_info.csv --output_dir=${OUTPUT_DIR}--alsologtostderr
