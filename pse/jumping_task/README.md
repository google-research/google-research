Code for the jumping task experiments in [Contrastive behavioral similarity embeddings
for generalization in reinforcement learning](https://agarwl.github.io/pse/), published in ICLR 2021.

<img src="https://i.imgur.com/3uY9ZXK.png" width="95%" alt="Jumping Task" >

To test whether the code runs correctly, go to the base `google_research` directory
and run: `bash pse/jumping_task/run.sh`.

To run (from inside the `pse` directory):
```
python -m jumping_task.train --train_dir {TRAIN_DIR} --training_epochs {EPOCHS}
```

To reproduce the main results for `RandConv + PSEs` on different grid
configurations, run the following commands (with `$SEED` varying from 1 to 100):

```
# Launch command for the "wide" grid
python -m jumping_task.train --training_epochs 2000  --seed $SEED \
--soft_coupling_temperature 0.01 --alpha 5.0 --temperature 0.5 \
--learning_rate 0.0026 --no_validation --rand_conv

# Launch command for the "narrow" grid:
python -m jumping_task.train --training_epochs 2000 --seed $SEED \
--min_obstacle_position 28 --max_obstacle_position 38 --min_floor_height 13 \
--max_floor_height 17 --positions_train_diff 2 --heights_train_diff 2 \
--soft_coupling_temperature 0.01 --alpha 5.0 --temperature 0.5 \
--learning_rate 0.0026 --no_validation --rand_conv

# Launch command for the random grid:
python -m jumping_task.train --training_epochs 2000 --seed $SEED \
--soft_coupling_temperature 0.01 --alpha 5.0 --temperature 0.5 \
--learning_rate 0.0026 --no_validation --rand_conv --random_tasks
```

To reproduce the results for `PSEs` on jumping task with colored obstacles,
run the command (with `$SEED` varying from 1 to 100):

```
python -m jumping_task.train --training_epochs 2000 --seed $SEED \
--soft_coupling_temperature 0.01 --alpha 5 --l2_reg 0.00007 \
--learning_rate 0.006 --temperature 0.5 --no_validation --use_colors
```

For more details about hyperparameters for baselines and ablations, refer to Appendix G.3 in the [paper](https://arxiv.org/pdf/2101.05265.pdf).
