# Call from the root as
# `./hypertransformer_tf/scripts/omniglot_5shot_v1.sh` with flags
# "--data_cache_dir=<omniglot_cache> --train_log_dir=<output_path>"

./scripts/omniglot_1shot_v1.sh --samples_transformer=100 --samples_cnn=100 $@
