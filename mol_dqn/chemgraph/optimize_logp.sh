export DYLD_LIBRARY_PATH="/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib"
python optimize_logp.py --model_dir="./save/optimize_logp/2/" --hparams="./configs/bootstrap_dqn.json"
