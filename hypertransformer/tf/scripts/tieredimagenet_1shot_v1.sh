# Call from the root as
# `./hypertransformer_tf/scripts/tieredimagenet_1shot_v1.sh` with flags
# "--data_cache_dir=<omniglot_cache> --train_log_dir=<output_path>"

python -m hypertransformer.tf.train --num_layerwise_features=8 --default_num_channels=8 \
  --samples_transformer=5 --samples_cnn=100 --num_labels=5 --learning_rate=0.01 \
  --learning_rate_decay_steps=100000.0 --learning_rate_decay_rate=0.95 \
  --train_steps=4000000 --steps_between_saves=50000 --lw_key_query_dim=0.5 \
  --lw_value_dim=0.5 --lw_inner_dim=0.5 --cnn_model_name='maxpool-4-layer' \
  --embedding_dim=32 --num_layers=3 --stride=1 --heads=8 \
  --shared_feature_extractor='4-layer' --shared_features_dim=16 \
  --shared_feature_extractor_padding=same --layerwise_generator=joint \
  --nolw_use_nonlinear_feature --lw_weight_allocation=output --nolw_generate_bias \
  --transformer_nonlinearity=lrelu --cnn_activation=lrelu \
  --noseparate_evaluation_bn_vars --apply_image_augmentations \
  --nolw_generate_bn --nouse_decoder --noadd_trainable_weights --image_size=84 \
  --balanced_batches --noper_label_augmentation --rotation_probability=0.0 \
  --boundary_probability=0.0 --smooth_probability=0.0 --contrast_probability=0.0 \
  --resize_probability=0.0 --negate_probability=0.0 --roll_probability=0.0 \
  --angle_range=30.0 --norandom_rotate_by_90 --train_dataset='tieredimagenet:0-350' \
  --test_dataset='tieredimagenet:448-607' --eval_datasets='tieredimagenet' \
  --num_task_evals=1024 --num_eval_batches=16 --eval_batch_size=100 \
  --test_rotation_probability=0.0 --test_smooth_probability=0.0 \
  --test_contrast_probability=0.0 --test_resize_probability=0.0 \
  --test_negate_probability=0.0 --notest_random_rotate_by_90 \
  --notest_per_label_augmentation --test_roll_probability=0.0 \
  --test_angle_range=-1.0 $@
