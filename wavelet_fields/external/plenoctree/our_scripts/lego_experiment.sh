export DATA_ROOT=./data/NeRF/nerf_synthetic
export CKPT_ROOT=./data/Plenoctree/checkpoints/syn_sh16
export SCENE=lego
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.extraction \
    --train_dir $CKPT_ROOT/$SCENE/ --is_jaxnerf_ckpt \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree.npz

python -m octree.optimization \
    --input $CKPT_ROOT/$SCENE/octrees/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree_opt.npz

python -m octree.evaluation \
    --input $CKPT_ROOT/$SCENE/octrees/tree_opt.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/

# [Optional] Only used for in-browser viewing.
echo python -m octree.compression \
    $CKPT_ROOT/$SCENE/octrees/tree_opt.npz \
    --out_dir $CKPT_ROOT/$SCENE/ \
    --overwrite