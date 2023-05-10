Implementation of clustering algorithms used in
"Scalable Differentially Private Clustering via Hierarchically
Separated Trees" KDD'22 https://arxiv.org/abs/2206.08646

# Usage
Example run on the s1_random dataset which has 2 features in [0, 2].

From google_search root directory:

OUTPUT_DIR=some/directory
python3 -m hst_clustering.run_clustering \
  --raw_data=hst_clustering/s1_random-standarized.txt \
  --output_dir=${OUTPUT_DIR} \
  --dimensions=2 \
  --min_value_entry=0 \
  --max_value_entry=2 \
  --k_params=10 \
  --alsologtostderr
