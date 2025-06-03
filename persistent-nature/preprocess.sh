# prepare the dataset -- filter LHQ 
python -m preprocessing.filter_lhq # adjust source directory as necessary
# at the end of this step, there should be a folder structure like this:
# dataset/lhq_processed/
# 	img/
# 	dpt_sky/
# 	dpt_depth/


# generate camera poses
python -m preprocessing.generate_poses 
# at the end of this step, there should be a file like this:
# poses/width38.4_far16_noisy_height.pth

# prepare dataset for triplane variation
# need to run the previous two steps first
python -m preprocessing.prepare_triplane_data
# at the end of this step, there should be a folder structure like this:
# dataset/lhq_processed_for_triplane_cam050/
# 	img/
# 	disp/
# 	sky/
# 	dataset.json
