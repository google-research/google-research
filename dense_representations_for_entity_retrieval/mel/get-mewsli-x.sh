# End-to-end script to reconstruct article text for Mewsli-X dataset from
# publicly available data sources.

# Run this from the mel/ directory:
#   bash get-mewsli-x.sh

set -eux

# Final output location.
DATASET_DIR="${PWD}/mewsli_x/output/dataset"
mkdir -p ${DATASET_DIR}

# Download and extract the dataset archive.
VERSION="20220518_6"
ARCHIVE_NAME="mewsli-x_${VERSION}.zip"
if [[ ! -e ${ARCHIVE_NAME} ]]; then
  wget "https://storage.googleapis.com/gresearch/mewsli/${ARCHIVE_NAME}"
fi
unzip -d ${DATASET_DIR} ${ARCHIVE_NAME}

# Download WikiNews dumps for 11 languages from archive.org.
bash mewsli_x/get_wikinews_dumps.sh

# Download the external wikiextractor tool and patch it.
bash tools/get_wikiextractor.sh

# Process the WikiNews dumps into lightly marked-up JSON format.
bash mewsli_x/run_wikiextractor.sh

# Parse clean text from the processed dumps according to the Mewsli-X dataset
# descriptors.
bash mewsli_x/run_parse_wikinews_i18n.sh

# Summary.
tail -n4 ${DATASET_DIR}/??/log

# Restore WikiNews article text into those released JSONL files that omitted it.
pushd ../../
for split in "dev" "test"; do
  python -m \
    dense_representations_for_entity_retrieval.mel.mewsli_x.restore_text \
    --index_dir="${DATASET_DIR}" \
    --input="${DATASET_DIR}/wikinews_mentions_no_text-${split}.jsonl" \
    --output="${DATASET_DIR}/wikinews_mentions-${split}.jsonl"
done
popd

set +x
echo
echo "The Mewsli-X dataset is now ready in ${DATASET_DIR}/:"
ls -lh ${DATASET_DIR}/*.jsonl