# End-to-end script to reconstruct clean text for Mewsli-9 dataset from publicly
# available data sources.

# Run this from the mel/ directory:
#   bash get-mewsli-9.sh

set -eux

# Final output location.
DATASET_DIR="./mewsli-9/output/dataset"
mkdir -p ${DATASET_DIR}

# Download the dataset descriptors.
wget https://storage.googleapis.com/gresearch/mewsli/mewsli-9.zip

# Extract dataset descriptors archive
unzip -d ${DATASET_DIR} mewsli-9.zip

# Download WikiNews dumps for 9 languages from archive.org.
bash mewsli-9/get_wikinews_dumps.sh

# Download the external wikiextractor tool and patch it.
bash tools/get_wikiextractor.sh

# Process the WikiNews dumps into lightly marked-up JSON format.
bash mewsli-9/run_wikiextractor.sh

# Parse clean text from the processed dumps according to the Mewsli-9 dataset
# descriptors.
bash mewsli-9/run_parse_wikinews_i18n.sh

# Summary.
tail -n4 ${DATASET_DIR}/??/log
