source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate plenoxels

export COMMON_ARGS="--debug False --use-pytorch False"
# python opt.py -t checkpoints/lego data/nerf_synthetic/lego -c configs/syn.json $COMMON_ARGS
# python opt.py -t checkpoints/lego_no_view_dirs data/nerf_synthetic/lego -c configs/syn_no_view_dir.json $COMMON_ARGS
python opt.py -t checkpoints/lego_test_low_res data/nerf_synthetic/lego -c configs/syn_testing.json $COMMON_ARGS
# python opt.py -t checkpoints/lego_test_low_res_no_viewdir data/nerf_synthetic/lego -c configs/syn_testing_no_view_dir.json $COMMON_ARGS
