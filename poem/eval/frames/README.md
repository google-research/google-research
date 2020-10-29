# Cross-View Pose Retrieval Evaluation

In the `frames` folder, we provide the keys to the frames we used in [1] for
cross-view pose retrieval evaluation on Human3.6M [2] and MPII-3DHP [3] dataset.
Please refer to the original datasets for more details.

## Frame Key Format
### Human3.6M
In the `frames/h36m` folder, we provide 4 text files for the validation split
[4], one for each camera view. In each text file, one frame key per row can be
found. The frame keys are in format
`${subject}:${action}:${camera_id}:${frame_index}`. For example, frame key `S11:Directions 1:54138969:000020` corresponds to the **21st** frame in the
video of action `Direction 1` by Subject `11` under Camera `54138969`.

**Notes:**

- The `action` strings follow those in the original dataset and may contain
whitespace.
- Frame index starts at 0.

### MPII-3DHP
In the `frames/3dhp` folder, we provide 11 text files for the training split
[3], one for each camera view. In each text file, one frame key per row can be
found. The frame keys are in format
`${subject}:${sequence_id}:${video_id}:${camera_id}:${frame_index}`. For
example, frame key `S1:Seq1:V0:C0:003370` corresponds to the **3371st** frame in
the video of Sequence `2` by Subject `1` under Camera `0` (Video `0`).

**Notes:**

- We use the **training** split of the dataset only for **testing** purposes. We
do not use MPII-3DHP for any training.

- `Video_id` is the same as `camera_id`.

- We do not use ceiling-mounted camera `11`, `12`, and `13`.

- Frame index starts at 0.


##Reference

[1] J.J. Sun, J. Zhao, L.-C. Chen, F. Schroff, H. Adam, T. Liu. View-invariant
probabilistic embedding for human pose. ECCV 2020.

[2] C. Ionescu, D. Papava, V. Olaru, C. Sminchisescu. Human3.6M: Large scale
datasets and predictive methods for 3D human sensing in natural environments.
IEEE TPAMI, 2013.

[3] D. Mehta, H. Rhodin, D. Casas, P. Fua, O. Sotnychenko, W. Xu, C. Theobalt.
Monocular 3D human pose estimation in the wild using improved CNN supervision.
3DV 2017.

[4] J. Martinez, R. Hossain, J. Romero, J.J. Little. A simple yet effective
baseline for 3D human pose estimation. ICCV 2017.
