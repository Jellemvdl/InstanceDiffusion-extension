#!/bin/bash

sh_file_name="demo_iterative_r1.sh"
gpu="0"

base_path="InstanceDiffusion-extension/src/lib/instancediffusion"

python inference.py \
  --num_images 8 \
  --output OUTPUT/ \
  --input_json ${base_path}/demos/demo_iterative_r1.json \
  --ckpt ${base_path}/pretrained/instancediffusion_sd15.pth \
  --test_config ${base_path}/configs/test_box.yaml \
  --guidance_scale 7.5 \
  --alpha 0.8 \
  --seed 0 \
  --mis 0.2 \
  --cascade_strength 0.4 \