#!/bin/bash


# Base path for videos
# base_path='./64x1_topk_4096_6_vids/'
base_path='./generated_vids/'

# base_path='./zeroseed_vids_checkpoint/'


# Define the model list
models=("wan21_1.3b_480x832x81_baseline2" "wan21_1.3b_480x832x81_cached" "wan21_1.3b_720x1280x81_baseline" "wan21_1.3b_720x1280x81_bitmaskcached" "wan21_14b_480x832x81_baseline" "wan21_14b_480x832x81_bitmaskcached" "wan21_14b_720x1280x81_baseline" "wan21_14b_720x1280x81_bitmaskcached")

# Define the dimension list
dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")

# dimensions=("subject_consistency")  #
# dimensions=("imaging_quality") 
# dimensions=("object_class") 
# dimensions=("multiple_objects") 
# dimensions=("color") 
# dimensions=("spatial_relationship") 
# dimensions=("scene") #
# dimensions=("temporal_style") #
# dimensions=("overall_consistency") #
# dimensions=("human_action") 
# dimensions=("temporal_flickering") #
# dimensions=("motion_smoothness") 
# dimensions=("dynamic_degree") #
# dimensions=("appearance_style") #
# dimensions=("background_consistency") #
# dimensions=("aesthetic_quality") #


# dimensions=("subject_consistency"color" "spatial_relationship" "multiple_objects" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree")
# dimensions=("temporal_flickering")

# Corresponding folder names
# folders=("subject_consistency" "scene" "overall_consistency" "overall_consistency" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "subject_consistency" "subject_consistency" "appearance_style")
# folders=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "subject_consistency" "subject_consistency" "appearance_style")

# folders=("temporal_flickering")


# Loop over each model
dev=$1
echo "Device = " $dev
model=${models[dev]}

export CUDA_VISIBLE_DEVICES=$dev
echo $model
# Loop over each dimension
for i in "${!dimensions[@]}"; do
    # Get the dimension and corresponding folder
    dimension=${dimensions[i]}
    folder=${dimensions[i]}

    # Construct the video path
    videos_path="${base_path}${model}/${folder}"
    echo "$dimension $videos_path"

    # Run the evaluation script
    /home/sdurvasula/miniconda3/envs/vbench2/bin/python evaluate.py --videos_path $videos_path --dimension $dimension --output_path ./vbench/evaluation_results_${model}/
done




