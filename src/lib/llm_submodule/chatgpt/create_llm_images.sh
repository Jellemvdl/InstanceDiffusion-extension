#!/bin/bash

# Get the absolute path to the script itself
script_dir=$(dirname "$(realpath "$0")")

# Prompt for the timestamp at which your input descriptions were generated
read -p "Enter the timestamp (folder name in chatgpt_data) at which your input descriptions were generated: " timestamp

# Construct dynamic paths based on the script location
input_dir="${script_dir}/chatgpt_data/${timestamp}" #directory of the input JSON files
output_dir="${script_dir}/chatgpt_output/${timestamp}" #directory of the output results

# Ensure the output directory exists
mkdir -p "$output_dir"
if [ ! -d "$input_dir" ]; then
  echo "Error: Input directory ${input_dir} does not exist."
  exit 1
fi

n_desc=$(ls "$input_dir"/*.json 2>/dev/null | wc -l)

echo "Number of images to create: $n_desc"
echo "Timestamp: $timestamp"

# Loop over the range of input JSONs and generate an output for each
for ((i=1; i<=n_desc; i++))
do
  input_file="${input_dir}/chatgpt_descriptions_bboxes${i}.json"
  
  # Check if the input file exists before running the command
  if [ -f "$input_file" ]; then
    python "${script_dir}/../../instancediffusion/inference.py" \
      --num_images 1 \
      --output "$output_dir" \
      --input_json "$input_file" \
      --ckpt "${script_dir}/../../instancediffusion/pretrained/instancediffusion_sd15.pth" \
      --test_config "${script_dir}/../../instancediffusion/configs/test_box.yaml" \
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
