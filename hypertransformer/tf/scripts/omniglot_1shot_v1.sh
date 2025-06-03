# Call from the root as
# `./hypertransformer_tf/scripts/omniglot_1shot_v1.sh` with flags
# "--data_cache_dir=<omniglot_cache> --train_log_dir=<output_path>"

python -m hypertransformer.tf.train --num_layerwise_features=8 --default_num_channels=8 \
  --samples_transformer=20 --samples_cnn=60 --num_labels=20 --learning_rate=0.02 \
  --learning_rate_decay_steps=100000.0 --learning_rate_decay_rate=0.95 \
  --train_steps=4000000 --steps_between_saves=50000 --lw_key_query_dim=1.0 \
  --lw_value_dim=1.0 --lw_inner_dim=1.0 --cnn_model_name='maxpool-4-layer' \
  --embedding_dim=32 --num_layers=3 --stride=1 --heads=2 \
  --shared_feature_extractor='4-layer' --shared_features_dim=32 \
  --shared_feature_extractor_padding=same --layerwise_generator=joint \
  --nolw_use_nonlinear_feature --lw_weight_allocation=output --nolw_generate_bias \
  --nolw_generate_bn --nouse_decoder --noadd_trainable_weights --image_size=28 \
  --balanced_batches --per_label_augmentation --rotation_probability=0.0 \
  --boundary_probability=0.0 --smooth_probability=0.0 --contrast_probability=0.0 \
  --resize_probability=0.0 --negate_probability=0.0 --roll_probability=0.0 \
  --angle_range=30.0 --random_rotate_by_90 --train_dataset='omniglot:0-1149' \
  --test_dataset='omniglot:1200-1622' --eval_datasets='omniglot' \
  --num_task_evals=1024 --num_eval_batches=16 --eval_batch_size=100 \
  --shuffle_labels_seed=2022 --test_rotation_probability=0.0 \
  --test_smooth_probability=0.0 --test_contrast_probability=0.0 \
  --test_resize_probability=0.0 --test_negate_probability=0.0 \
  --test_roll_probability=0.0 --test_angle_range=-1.0 $@
