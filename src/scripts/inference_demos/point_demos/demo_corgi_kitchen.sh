#!/bin/bash

sh_file_name="demo_corgi_kitchen.sh"
gpu="0"

base_path="InstanceDiffusion-extension/src/lib/instancediffusion"

python inference.py \
  --num_images 8 \
  --output InstanceDiffusion-extension/src/data/demo_outputs \
  --input_json ${base_path}/demos/demo_corgi_kitchen.json \
  --ckpt ${base_path}/pretrained/instancediffusion_sd15.pth \
  --test_config ${base_path}/configs/test_point.yaml \
  --guidance_scale 7.5 \
  --alpha 0.8 \
  --seed 0 \
  --mis 0.2 \
  --cascade_strength 0.4 \