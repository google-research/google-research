model_name=scratch_svhn

python3 finetuning.py \
    --target_dataset=svhn_cropped_small \
    --train_steps=1200 \
    --model_dir=trained_models/$model_name \
    --train_batch_size=8 \
    --target_base_learning_rate=0.005 \
    --src_num_classes=5

python3 evaluate.py \
    --ckpt_path=trained_models/${model_name}/model.ckpt-1200 \
    --src_num_classes=5 \
    --target_dataset=svhn_cropped_small \
    --cls_dense_name=final_dense_dst
