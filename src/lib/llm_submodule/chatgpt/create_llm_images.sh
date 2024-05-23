#!/bin/bash

# Prompt for the timestamp at which your input descriptions were generated
read -p "Enter the timestamp (folder name in chatgpt_data) at which your input descriptions were generated: " timestamp
read -p "Enter the number of images input JSON files: " n_desc

echo "Number of images to create: $n_desc"
echo "Timestamp: $timestamp"

input_dir="chatgpt_data/${timestamp}" #directory of the input JSON files
output_dir="chatgpt_output/${timestamp}" #directory of the output results

# Ensure the output directory exists
mkdir -p "$output_dir"

#loop over the range of input JSONs and generate an output for each
for ((i=1; i<=n_desc; i++))
do
  input_file="${input_dir}/chatgpt_descriptions_bboxes${i}.json"
  
  # Check if the input file exists before running the command
  if [ -f "$input_file" ]; then
    python ../../instancediffusion/inference.py \
      --num_images 1 \
      --output "$output_dir" \
      --input_json "$input_file" \
      --ckpt "../../instancediffusion/pretrained/instancediffusion_sd15.pth" \
      --test_config "../../instancediffusion/configs/test_box.yaml" \
      --guidance_scale 7.5 \
      --alpha 0.8 \
      --seed 0 \
      --mis 0.2 \
      --cascade_strength 0.4
  else
    echo "Warning: ${input_file} not found."
  fi
done

echo "Inference completed for all input files."
