export DYLD_LIBRARY_PATH="/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib"
python optimize_qed.py --model_dir="./save/optimize_qed/5/" --hparams="./configs/bootstrap_dqn.json"
