#!/bin/bash

sh_file_name="demo_cat_dog_robin.sh"
gpu="0"

base_path="src/lib/instancediffusion"

python ${base_path}/inference.py \
  --num_images 8 \
  --output src/data/demo_outputs \
  --input_json ${base_path}/demos/demo_cat_dog_robin.json \
  --ckpt ${base_path}/pretrained/instancediffusion_sd15.pth \
  --test_config ${base_path}/configs/test_box.yaml \
  --guidance_scale 7.5 \
  --alpha 0.8 \
  --seed 0 \
  --mis 0.36 \
  --cascade_strength 0.4
