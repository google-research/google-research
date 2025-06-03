model_name=mnist_pretrain

python3 finetuning.py \
     --target_dataset=mnist \
     --train_steps=2000 \
     --target_base_learning_rate=0.01 \
     --model_dir=trained_models/${model_name}\
     --train_batch_size=128

python3 evaluate.py \
    --target_dataset=mnist \
    --ckpt_path=trained_models/${model_name}/model.ckpt-2000 \
    --cls_dense_name=final_dense_dst
