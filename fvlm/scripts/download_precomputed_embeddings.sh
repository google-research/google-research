SOURCE="https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fvlm/embeddings.zip"

wget "${SOURCE}"
unzip "./embeddings.zip"
rm "./embeddings.zip"
rm -rf "__MACOSX"